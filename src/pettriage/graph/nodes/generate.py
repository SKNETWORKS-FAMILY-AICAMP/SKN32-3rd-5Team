"""③ 압축 · 생성 · ⑤ 평이화 노드 — **WS2 구현 대기.**

설계 근거: 02 §2 · 05 §4 (③⑤)
"""

from __future__ import annotations

from ..state import GraphState


def compress_context(state: GraphState) -> GraphState:
    """검색 결과를 질문에 필요한 만큼으로 줄인다.

    **원문에 없는 수치·단위·종을 추가하지 않는다.** 압축 실행 여부는
    길이 임계로 코드가 정한다 (05 §4).

    Returns: `{"context": ...}`
    """
    raise NotImplementedError("WS2: tests/todo/test_graph_nodes.py::TestGenerate 참조")


def generate_draft(state: GraphState) -> GraphState:
    """근거를 바탕으로 초안을 만든다. 아직 사용자에게 나가지 않는다.

    Returns: `{"draft": ...}`
    """
    raise NotImplementedError("WS2: tests/todo/test_graph_nodes.py::TestGenerate 참조")


def simplify(state: GraphState) -> GraphState:
    """수의학 용어를 보호자 표현으로. **위험도를 낮추는 완곡 표현을 쓰지 않는다.**

    용어집 준수는 코드가 검증한다 (05 §4).

    Returns: `{"answer": ...}`
    """
    raise NotImplementedError("WS2: tests/todo/test_graph_nodes.py::TestGenerate 참조")
