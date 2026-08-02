"""프롬프트 템플릿 — 학습과 추론이 **같은 문자열**을 쓴다.

설계 근거: docs/03_모델-멀티태스크학습.md §2 "템플릿 통일"

    학습 때와 추론 때 프롬프트가 다르면 성능 하락의 원인을 영원히 못 찾는다.
    그래서 두 곳이 이 모듈 하나를 부른다. 태스크 지시문을 다른 데 적지 않는다.

Qwen3는 chat template을 쓰므로 **역할 분리 메시지**로 만든다 (D-42).
토크나이저의 `apply_chat_template` 에 그대로 넣을 수 있는 형태다.
"""

from __future__ import annotations

from .tasks import Task

#: 전 태스크 공통 규칙. **모델이 지어내지 못하게 막는 문장들이다.**
_COMMON = (
    "너는 반려동물 헬스케어 보조 시스템의 구성 요소다.\n"
    "진단하지 않는다. 주어진 입력 밖의 사실을 만들어내지 않는다.\n"
    "확신할 수 없으면 추측하지 말고 정해진 미확인 값을 출력한다."
)

_INSTRUCTIONS: dict[Task, str] = {
    Task.CLASSIFY: (
        "사용자 발화의 의도와 위험 성격을 분류한다.\n"
        "다음 라벨 중 **하나만** 출력한다. 라벨 외의 문자를 덧붙이지 않는다.\n"
        # graph.nodes.classify.ALLOWED_INTENTS 와 같은 목록이어야 한다 —
        # 여기 없는 라벨을 출력하면 코드가 걸러 unknown 으로 떨어진다.
        "  intoxication (중독·오섭취) · symptom (증상) · nutrition (영양·급여) · general (그 외)\n"
        "판단이 서지 않으면 `unknown` 을 출력한다."
    ),
    Task.SLOT: (
        "발화에서 슬롯을 추출해 **다음 키를 정확히 그대로 쓴** JSON 객체 하나로 출력한다.\n"
        # graph.nodes.slots.extract_slots 가 이 키 이름으로 llm.get(...) 을 호출한다.
        # 다른 키(예: weight·toxic_food)로 내면 코드가 못 찾아 결측으로 처리된다.
        '  {"species": "dog"|"cat"|"bird"|null, "weight_kg": 숫자|null, '
        '"amount_g": 숫자|null, "substance": 문자열|null}\n'
        "값이 발화에 없으면 **추정하지 말고 null** 로 둔다.\n"
        "`species` 는 반드시 `dog`·`cat`·`bird` 중 하나(영문)로 쓴다 — "
        "한국어(개·강아지 등)나 품종명·이름에서 추측하지 않는다.\n"
        "`weight_kg`·`amount_g` 는 단위를 뺀 숫자만 쓴다(kg·g 단위로 이미 통일된 값).\n"
        # 실측(2026-08-03): "양파"가 종종 "onion"으로 나왔다 — 코퍼스 물질명은
        # 전부 한국어라 영문으로 나오면 매칭이 아예 안 된다(폐쇄 목록 이탈).
        "`substance` 는 **발화에 쓰인 한국어 표현을 그대로** 옮긴다 — "
        "번역하거나 학명·영문으로 바꾸지 않는다.\n"
        # 실측(2026-08-03): "설사를 해요"·"기침을 해요"·"깃털을 뽑아요" 같은
        # **증상·행동 묘사**를 substance 자리에 그대로 넣는 오류가 잦았다.
        # substance는 "먹었거나·핥았거나·접촉한 대상(음식·식물·화학물질 등)"만이다.
        "`substance` 는 **먹었거나 핥았거나 접촉한 대상**(음식·식물·화학물질 등)만 담는다 — "
        "구토·설사·기침·가려움·깃털 뽑기처럼 **증상이나 행동을 나타내는 표현은 "
        "substance 가 아니다.** 그런 문장은 무엇을 먹었는지 나와 있지 않으므로 "
        "substance 를 null 로 둔다.\n"
        "  예: '강아지가 설사를 해요' → substance: null (증상만 있고 물질 언급 없음)\n"
        "  예: '강아지가 초콜릿을 먹고 설사를 해요' → substance: '초콜릿' (물질이 실제로 있음)"
    ),
    Task.COMPRESS: (
        "검색된 문서들을 질문에 필요한 내용만 남겨 압축한다.\n"
        "**원문에 없는 수치·단위·종을 추가하지 않는다.**\n"
        "수치는 단위까지 원문 그대로 옮긴다."
    ),
    Task.VERIFY: (
        "답변의 각 문장이 근거 문서에 의해 뒷받침되는지 판정한다.\n"
        "문장마다 `근거있음` / `근거없음` / `모순` 중 하나를 출력한다.\n"
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


def system_prompt(task: Task) -> str:
    return f"{_COMMON}\n\n[과제] {_INSTRUCTIONS[task]}"


def build_messages(task: Task, user_input: str) -> list[dict[str, str]]:
    """학습·추론 공통. `tokenizer.apply_chat_template()` 에 그대로 넣는다."""
    return [
        {"role": "system", "content": system_prompt(task)},
        {"role": "user", "content": user_input},
    ]


def build_sample(task: Task, user_input: str, target: str) -> list[dict[str, str]]:
    """학습 샘플 1건. assistant 턴만 손실 계산 대상이 된다."""
    return [*build_messages(task, user_input), {"role": "assistant", "content": target}]
