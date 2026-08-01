"""검색 노드 — **WS2 구현 대기.**

설계 근거: 02 §8 · D-10 · 05 §4

    **필터 구성은 전부 코드가 한다.** 여기에 LLM 을 넣지 않는다 (05 §4).
    유사도 임계 미만이면 **검색 실패로 보고 거절**한다 (02 §8.3).
"""

from __future__ import annotations

from typing import Any

from ..state import GraphState


def build_filter(state: GraphState) -> GraphState:
    """슬롯 → 검색 필터. 결정론이다.

    `species` 는 반드시 들어간다. 종이 `dog`·`cat` 이면 `mammal`·`all` 문서도
    함께 봐야 4단계가 성립한다 (D-39 — 고양이 자체 자료는 2단계뿐이다).

    Returns:
        `{"where": {...}}`
    """
    raise NotImplementedError("WS2: tests/todo/test_graph_nodes.py::TestFilter 참조")


def retrieve(state: GraphState, store: Any = None) -> GraphState:
    """벡터 검색 + 임계값 적용.

    Returns:
        `{"hits": [...]}`. 임계 통과분이 0건이면 부르는 쪽이
        `refused / 근거없음` 으로 보낸다 — **빈 결과는 실패가 아니라 신호다.**
    """
    raise NotImplementedError("WS2: tests/todo/test_graph_nodes.py::TestRetrieve 참조")
