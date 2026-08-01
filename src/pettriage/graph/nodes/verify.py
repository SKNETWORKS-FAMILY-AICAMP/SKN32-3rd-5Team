"""④ 근거 검증 노드 — **WS2 구현 대기. 이 프로젝트의 핵심이다.**

설계 근거: 02 §2 · §9 · D-05 · 05 §4 (④)

    문장별로 `근거있음` / `근거없음` / `모순` 을 판정하고,
    **판정에 따른 조치는 코드가 한다** — 문장 제거 · 재검색 1회 · 거절.

    애매하면 `근거없음` 쪽으로 판정한다. 놓친 환각이 나가는 것보다 낫다.
"""

from __future__ import annotations

from ..state import GraphState

VERDICTS = ("근거있음", "근거없음", "모순")

#: 재검색은 1회까지 (02 §2). 무한 루프를 막는다.
MAX_RETRY = 1


def verify_grounding(state: GraphState) -> GraphState:
    """초안의 각 문장이 검색된 근거로 뒷받침되는지 검사한다.

    Returns:
        `{"verdicts": [{"sentence": ..., "verdict": ...}], "retry_count": n}`.
        전부 `근거없음` 이고 재검색도 끝났으면
        `{"status": "refused", "refusal_reason": "검증실패"}`.
    """
    raise NotImplementedError("WS2: tests/todo/test_graph_nodes.py::TestVerify 참조")
