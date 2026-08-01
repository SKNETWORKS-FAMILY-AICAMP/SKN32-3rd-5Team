"""계산 노드 (비-RAG) — **WS2 구현 대기.**

설계 근거: 02 §7 · D-16 · 05 §4

    **수치는 벡터 검색으로 찾지 않는다** (D-16). 관계형 테이블을 조회하고
    계산은 코드가 한다. LLM 은 여기 관여하지 않는다.
"""

from __future__ import annotations

from ..state import GraphState


def compute_metrics(state: GraphState) -> GraphState:
    """체중당 섭취량·권장 열량 등을 계산한다.

    단위는 내부적으로 `weight_g` 하나로 통일한다 (D-17).
    kg↔g 혼용이 곧 용량 계산 오류다.

    Returns:
        `{"computed": {"dose_per_kg": ..., "unit": ...}}`
    """
    raise NotImplementedError("WS2: tests/todo/test_graph_nodes.py::TestCompute 참조")
