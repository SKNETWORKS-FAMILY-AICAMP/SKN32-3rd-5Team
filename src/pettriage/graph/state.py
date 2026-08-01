"""LangGraph 상태 (02 §6.1).

설계 근거: docs/02_시스템-아키텍처.md §6 · docs/05 §3

    이 State 는 **되묻기 세션 상태**다 — 조각 3. 휘발성이고 한 질의 안에서만 산다.
    반려동물 일일 기록은 여기 들어오지 않는다. 그건 조각 4(RAG)의 검색 대상이다 (05 §3).

노드는 State 를 받아 **바뀐 키만** 돌려준다. LangGraph 가 병합한다.
"""

from __future__ import annotations

from typing import Any, Literal, TypedDict

Intent = Literal["intoxication", "symptom", "nutrition", "general", "unknown"]


class Slots(TypedDict, total=False):
    """② 슬롯 추출 결과. **없는 값은 키를 두지 않는다** — 추정 금지 (D-10)."""

    species: str  # dog · cat · bird
    breed: str
    weight_kg: float
    age_month: int
    substance: str
    amount_g: float
    elapsed_hours: float
    signs: list[str]


class GraphState(TypedDict, total=False):
    """질의 파이프라인 전체가 공유하는 상태."""

    # ── 입력 ────────────────────────────────────────────────
    question: str
    session_id: str
    pet_id: str

    # ── ① 분류 ──────────────────────────────────────────────
    intent: Intent
    risk: str

    # ── ② 슬롯 ──────────────────────────────────────────────
    slots: Slots
    missing_slots: list[str]
    clarify_turns: int
    clarify_question: str

    # ── 검색 · 계산 ─────────────────────────────────────────
    where: dict[str, Any]  # 검색 필터 — 코드가 만든다 (05 §4)
    hits: list[Any]  # retrieval.Hit
    computed: dict[str, Any]  # 계산 노드 결과 (체중당 섭취량 등)

    # ── ③ 압축 · 생성 ───────────────────────────────────────
    context: str
    draft: str

    # ── 트리아지 ────────────────────────────────────────────
    rule_level: int | None
    llm_level: int | None
    triage_level: int | None
    escalation_conditions: list[str]

    # ── ④ 근거 검증 ─────────────────────────────────────────
    verdicts: list[dict[str, str]]  # 문장별 근거있음/근거없음/모순
    retry_count: int

    # ── 출력 ────────────────────────────────────────────────
    status: Literal["answered", "clarify", "refused"]
    answer: str
    refusal_reason: str


def initial_state(question: str, session_id: str, **kw: Any) -> GraphState:
    """빈 상태. **카운터는 반드시 0으로 시작한다** — 없으면 루프 상한이 안 먹는다."""
    st: GraphState = {
        "question": question,
        "session_id": session_id,
        "slots": {},
        "missing_slots": [],
        "clarify_turns": 0,
        "retry_count": 0,
        "hits": [],
        "computed": {},
        "verdicts": [],
        "escalation_conditions": [],
        "rule_level": None,
        "llm_level": None,
        "triage_level": None,
    }
    st.update(kw)  # type: ignore[typeddict-item]
    return st
