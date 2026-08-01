"""② 슬롯 추출 · 되묻기 노드 — **WS2 구현 대기.**

설계 근거: 02 §2 · §9 · D-10 · 05 §4 (②)

    **종이 없으면 검색으로 넘어가지 않는다** (D-10).
    포유류 기준을 조류에 적용하는 것이 이 도메인에서 가장 치명적인 오류다.

    LLM 이 슬롯을 뽑고, **코드가 JSON 스키마로 검증**한다.
    발화에 없는 값을 채우면 그게 곧 환각이므로 `null` 로 둔다.
"""

from __future__ import annotations

from ..state import GraphState

#: 의도별 필수 슬롯. 조류는 정량 임계치가 0건이라 체중·섭취량을 요구하지 않는다 (D-09).
REQUIRED_SLOTS = {
    "intoxication": ("species", "substance"),
    "symptom": ("species",),
    "nutrition": ("species",),
    "general": ("species",),
}


def extract_slots(state: GraphState) -> GraphState:
    """자유서술에서 슬롯을 뽑는다.

    Returns:
        `{"slots": ..., "missing_slots": [...]}`.
        발화에 없는 값은 **채우지 않는다**.
    """
    raise NotImplementedError("WS2: tests/todo/test_graph_nodes.py::TestSlots 참조")


def ask_clarify(state: GraphState) -> GraphState:
    """되묻기 문구를 만든다. 상한은 설정값(`triage.max_clarify_turns`)이다.

    Returns:
        `{"clarify_question": ..., "clarify_turns": n}`.
        상한 초과면 `{"status": "refused", "refusal_reason": "되묻기상한"}`.
    """
    raise NotImplementedError("WS2: tests/todo/test_graph_nodes.py::TestSlots 참조")
