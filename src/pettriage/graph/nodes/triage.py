"""트리아지 판정 노드 — **WS2 구현 대기.**

설계 근거: 02 §7.2 · 05 §5 · D-09 · D-39

    🔒 **`max()` 를 직접 쓰지 않는다.** 반드시 `triage.gate.apply_gate` 를 부른다 —
    MONITOR 상승 조건 검사가 거기 붙어 있다.
"""

from __future__ import annotations

from ..state import GraphState


def decide_triage(state: GraphState) -> GraphState:
    """규칙 1차 → 미적용 시 LLM → 하향 금지 게이트.

    규칙 테이블 입력이 **종별로 다르다** (D-09 개정).
    개·고양이는 `물질 × 체중당 섭취량` 정량, 조류는 `물질 × 증상` 정성이다.
    조류에 수치를 요구하면 근거에 없는 값을 모델이 지어낸다.

    Returns:
        `{"rule_level": ..., "llm_level": ..., "triage_level": ...,
          "escalation_conditions": [...]}`
    """
    raise NotImplementedError("WS2: tests/todo/test_graph_nodes.py::TestTriage 참조")
