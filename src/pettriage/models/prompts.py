"""프롬프트 템플릿 — 학습과 추론이 **같은 문자열**을 쓴다.

설계 근거: docs/03_모델-멀티태스크학습.md §2 "템플릿 통일"

    학습 때와 추론 때 프롬프트가 다르면 성능 하락의 원인을 영원히 못 찾는다.
    그래서 두 곳이 이 모듈 하나를 부른다. 태스크 지시문을 다른 데 적지 않는다.

Qwen3는 chat template을 쓰므로 **역할 분리 메시지**로 만든다 (D-42).
토크나이저의 `apply_chat_template` 에 그대로 넣을 수 있는 형태다.
"""

from __future__ import annotations

from .tasks import SPECS, Task

#: 전 태스크 공통 규칙. **모델이 지어내지 못하게 막는 문장들이다.**
_COMMON = (
    "너는 반려동물 헬스케어 보조 시스템의 구성 요소다.\n"
    "진단하지 않는다. 주어진 입력 밖의 사실을 만들어내지 않는다.\n"
    "확신할 수 없으면 추측하지 말고 정해진 미확인 값을 출력한다."
)

_INSTRUCTIONS: dict[Task, str] = {
    Task.CLASSIFY: (
        "사용자 발화의 의도와 위험 성격을 분류한다.\n"
        "아래 라벨 중 **하나만**, 적힌 문자열 그대로 출력한다. 다른 문자를 덧붙이지 않는다.\n"
        "판단이 서지 않으면 `unknown` 을 출력한다."
    ),
    Task.SLOT: (
        "발화에서 슬롯을 추출해 **JSON 객체 하나만** 출력한다. 설명을 덧붙이지 않는다.\n"
        "값이 발화에 없으면 **추정하지 말고 null** 로 둔다.\n"
        "종은 명시된 경우에만 채운다 — 품종명이나 이름에서 추측하지 않는다."
    ),
    # ③ 은 **기간 리포트**를 요약한다 (D-83). 2026-08-03 까지 이 지시문은
    # *"검색된 문서들을 질문에 필요한 내용만 남겨 압축한다"* 였고, 질의 경로의
    # 압축 노드를 향하고 있었다. 그 노드가 없어졌으므로 지시문도 함께 옮긴다 —
    # **이 파일은 학습과 추론이 같이 읽는다.** 여기가 안 바뀌면 이서은 팀원이
    # 만드는 ③ 학습 데이터가 없어진 노드를 향해 생성된다 (03 §5 템플릿 통일).
    Task.COMPRESS: (
        "반려동물 일일 기록 여러 건을 보호자가 읽을 기간 요약으로 만든다.\n"
        "**기록에 없는 증상·수치·날짜를 추가하지 않는다.**\n"
        "수치는 단위까지 기록 그대로 옮긴다.\n"
        "진단하지 않는다 — 관찰된 것과 변화만 적는다."
    ),
    Task.VERIFY: (
        "답변의 각 문장이 근거 문서에 의해 뒷받침되는지 판정한다.\n"
        "문장마다 아래 라벨 중 하나를 적힌 그대로 출력한다.\n"
        "**애매하면 `근거없음` 쪽으로 판정한다** — 놓친 환각이 나가는 것보다 낫다."
    ),
    Task.SIMPLIFY: (
        "수의학 용어를 보호자가 이해할 표현으로 바꾼다.\n"
        "**의미를 바꾸거나 위험도를 낮추는 완곡 표현을 쓰지 않는다.**\n"
        "수치와 단위는 그대로 둔다."
    ),
    Task.TRANSLATE: (
        "원문을 한국어로 옮긴다.\n**학명·수치·단위는 원문 표기를 그대로 유지한다** — 검증 앵커다."
    ),
}


def _label_block(task: Task) -> str:
    """**허용 라벨을 프롬프트에 싣는다** (D-73).

    ⚠️ 여기에 라벨을 손으로 적지 않는다. `SPECS[task].labels` 가 단일 출처이고,
    `graph/nodes/classify.py` 의 검증기도 **같은 것**을 본다 (D-22).

    적지 않았을 때 무슨 일이 났는지 — 모델이 보기를 모르니 `'위험성우려'` 처럼
    그럴듯한 말을 냈고, 코드가 전부 `unknown` 으로 걸러 거절로 보냈다.
    **진짜 LLM 을 붙였더니 키워드 폴백보다 성적이 나빠졌다** (2026-08-02).
    """
    spec = SPECS[task]
    if not spec.labels:
        return ""
    lines = [f"\n[라벨] 다음 중 하나를 **그대로** 출력한다 — {' · '.join(spec.labels)}"]
    for label in spec.labels:
        hint = (spec.label_hints or {}).get(label)
        if hint:
            lines.append(f"  {label:<14} {hint}")
    return "\n".join(lines)


def _schema_block(task: Task) -> str:
    """**출력 스키마를 프롬프트에 싣는다** (D-86 · 03 §5).

    ⚠️ 여기에 키를 손으로 적지 않는다. `SPECS[task].output_keys` 가 단일 출처이고,
    `graph/nodes/slots.py` 의 파서도 **같은 것**을 본다 (D-22 · D-73).

    적지 않았을 때 무슨 일이 났는지 — 모델이 키 이름을 모르니 `concern`·`item`
    같은 것을 지어냈고, 코드는 `substance` 만 보므로 **뽑아 놓고 버렸다.**
    ②슬롯이 사실상 키워드 폴백으로 돌고 있었는데 폴백 집계는 100% 로 찍혔다
    (2026-08-03 실측).
    """
    spec = SPECS[task]
    if not spec.output_keys:
        return ""
    hints = spec.key_hints or {}
    width = max(len(k) for k in spec.output_keys)
    rules = "\n".join(f"  {k:<{width}}  {hints.get(k, '')}".rstrip() for k in spec.output_keys)
    # **형식 예시도 키 목록에서 만든다.** 손으로 적으면 키가 하나 늘 때 낡는다.
    shape = "{" + ", ".join(f'"{k}": null' for k in spec.output_keys) + "}"
    return (
        "\n\n[출력 스키마] **아래 키만** 쓴다. 다른 이름의 키를 만들지 않는다.\n"
        "값을 모르면 그 키를 `null` 로 둔다 — 키를 빼지 않는다.\n\n" + rules + f"\n\n형식: {shape}"
    )


def system_prompt(task: Task) -> str:
    return f"{_COMMON}\n\n[과제] {_INSTRUCTIONS[task]}{_label_block(task)}{_schema_block(task)}"


def build_messages(task: Task, user_input: str) -> list[dict[str, str]]:
    """학습·추론 공통. `tokenizer.apply_chat_template()` 에 그대로 넣는다."""
    return [
        {"role": "system", "content": system_prompt(task)},
        {"role": "user", "content": user_input},
    ]


def build_sample(task: Task, user_input: str, target: str) -> list[dict[str, str]]:
    """학습 샘플 1건. assistant 턴만 손실 계산 대상이 된다."""
    return [*build_messages(task, user_input), {"role": "assistant", "content": target}]
