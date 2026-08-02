"""멀티태스크 5종 정의.

설계 근거: docs/03_모델-멀티태스크학습.md §2 · docs/06 D-05

    공지 예시(분류·요약·번역)를 그대로 쓰지 않고 **파이프라인이 실제로
    필요로 하는 태스크로 재구성**했다. 각 태스크는 02 §6 그래프의
    특정 노드에 대응하며, 그 노드가 없으면 태스크도 없다.

    ④ 근거 검증이 이 구성의 핵심이다 — 과제 목표 1번(환각 방지)을
    파인튜닝 모델이 직접 담당하게 만드는 지점이다.

**출력 길이가 태스크마다 다른 것은 의도다.** 라벨 한 단어(①)부터
문단(③)까지 섞어 **태스크 간섭이라는 실제 논점**을 확보한다 (04 E4).
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum


class Task(StrEnum):
    """태스크 ID. 학습 샘플·평가 지표·프롬프트가 모두 이 값으로 묶인다."""

    CLASSIFY = "classify"  # ① 의도·위험 분류
    SLOT = "slot"  # ② 슬롯 추출
    COMPRESS = "compress"  # ③ 컨텍스트 압축
    VERIFY = "verify"  # ④ 근거 검증  ← 핵심
    SIMPLIFY = "simplify"  # ⑤ 평이화
    # ⑥ 번역 — **미채택 확정** (2026-08-02 · D-19 후속).
    #   경로 ②(D-37·D-38·D-45)로 벡터DB 청크가 전부 한국어가 되어 번역 대상이 사라졌다.
    #   비교군을 만들 수 없어 04 E2 도 폐기했다. **멀티태스크는 5종이다** (D-05).
    #   값을 지우지 않고 남기는 이유 — 과거 학습 샘플·설정에 문자열이 남아 있을 수 있고,
    #   여기서 없애면 로딩이 KeyError 로 죽는다. `DEFAULT_TASKS` 에 없으므로 학습에는 안 들어간다.
    TRANSLATE = "translate"


@dataclass(frozen=True)
class TaskSpec:
    task: Task
    graph_node: str  # 02 §6 그래프의 대응 노드
    output_kind: str  # 출력 형태 — 간섭 분석의 축
    verified_by: str  # 05 §4 — LLM 출력에 붙는 검증 코드
    metric: str  # 04 §3 태스크별 지표

    #: **이 태스크가 낼 수 있는 라벨.** 비어 있으면 자유 출력이다.
    #:
    #: ⚠️ 여기가 단일 출처다 (D-22 · D-73). 2026-08-02 실측에서 —
    #:
    #:     ① 프롬프트는 *"허용된 라벨 중 하나만 출력한다"* 라고만 적었다.
    #:        **그 라벨이 무엇인지는 안 적혀 있었다.**
    #:     ② 코드는 `ALLOWED_INTENTS` 로 정확히 대조했다.
    #:
    #:   모델은 보기를 모르니 `'위험성우려'`·`'high_risk'` 같은 그럴듯한 말을 냈고,
    #:   코드가 전부 `unknown` 으로 걸러 **거절**로 보냈다. 결과가 뒤집혔다 —
    #:   **키워드 폴백(통과 10%)보다 진짜 LLM(3.3%)이 더 나빴다.**
    #:   *(이서은 팀원 발견)*
    #:
    #:   프롬프트와 검증기가 **같은 목록을 봐야 한다.** 프롬프트에 손으로 다시 적으면
    #:   라벨이 하나 늘어날 때 같은 사고가 반복된다.
    labels: tuple[str, ...] = ()

    #: 라벨의 뜻. 프롬프트에 함께 실어 **모델이 경계를 알게** 한다.
    #: `general` 이 *"우리가 다루지 않는 질문"* 이라는 것은 설명 없이는 알 수 없다 (D-46).
    label_hints: dict[str, str] | None = None


SPECS: dict[Task, TaskSpec] = {
    Task.CLASSIFY: TaskSpec(
        task=Task.CLASSIFY,
        graph_node="classify_intent",
        output_kind="단일 라벨",
        verified_by="허용목록 검증 · 미분류 시 폴백",
        metric="macro F1",
        labels=("intoxication", "symptom", "nutrition", "general"),
        label_hints={
            "intoxication": "물질을 먹었거나 핥았거나 접촉했다",
            "symptom": "증상만 말하고 물질은 말하지 않았다",
            "nutrition": "급여·영양·사료에 대한 질문이다",
            "general": "우리가 다루지 않는 질문 — 이름 짓기·훈련·보험·브랜드 추천",
        },
    ),
    Task.SLOT: TaskSpec(
        task=Task.SLOT,
        graph_node="extract_slots",
        output_kind="JSON 객체",
        verified_by=(
            "JSON 스키마 검증 · **폐쇄 목록 정규화**(D-59 ① · `vocabulary.resolve_substance`) "
            "· 결측 판정 · 되묻기 상한 2회"
        ),
        metric="슬롯 단위 정확도 · 결측 탐지율",
    ),
    Task.COMPRESS: TaskSpec(
        task=Task.COMPRESS,
        graph_node="compress_context",
        output_kind="문단",
        verified_by="길이 임계 · 원문 포함 검사",
        metric="압축률 대비 근거 보존율",
    ),
    Task.VERIFY: TaskSpec(
        task=Task.VERIFY,
        graph_node="verify_grounding",
        output_kind="문장별 3값 라벨",
        verified_by="판정에 따른 게이트 · 재검색 트리거 · 문장 제거",
        metric="근거없음 탐지 재현율 (**놓치면 환각이 나간다**)",
        labels=("근거있음", "근거없음", "모순"),
    ),
    Task.SIMPLIFY: TaskSpec(
        task=Task.SIMPLIFY,
        graph_node="simplify_terms",
        output_kind="문장",
        verified_by="용어집 준수 검증",
        metric="용어집 적용률 · 의미 보존",
    ),
    Task.TRANSLATE: TaskSpec(
        task=Task.TRANSLATE,
        graph_node="(없음 — 미채택)",
        output_kind="문장",
        verified_by="학명·수치 앵커 대조",  # 미채택이라 실제로 도는 검증은 아니다
        metric="수치·학명 보존율",
    ),
}

#: 기본 학습 대상 **5종이 최종이다** (D-05 · D-19 후속).
#: ⑥ 번역은 편입되지 않는다 — 번역할 원문이 인덱스에 없다 (2026-08-02).
DEFAULT_TASKS: tuple[Task, ...] = (
    Task.CLASSIFY,
    Task.SLOT,
    Task.COMPRESS,
    Task.VERIFY,
    Task.SIMPLIFY,
)
