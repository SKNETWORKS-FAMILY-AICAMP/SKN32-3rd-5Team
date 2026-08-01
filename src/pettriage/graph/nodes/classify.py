"""① 의도·위험 분류 노드 — **WS2 구현 대기.**

설계 근거: 02 §2 · 05 §4 (①)

    LLM 이 라벨을 내고, **코드가 허용목록으로 검증한다.**
    목록에 없는 라벨이 오면 지어낸 것이므로 `unknown` 으로 떨어뜨린다.
"""

from __future__ import annotations

from ..state import GraphState

#: 허용 라벨. LLM 출력이 여기 없으면 폴백한다 (05 §4).
ALLOWED_INTENTS = ("intoxication", "symptom", "nutrition", "general")


def classify_intent(state: GraphState) -> GraphState:
    """질문을 의도·위험으로 분류한다.

    Returns:
        `{"intent": ..., "risk": ...}` 만. 목록 밖이면 `intent="unknown"`.
    """
    raise NotImplementedError("WS2: tests/todo/test_graph_nodes.py::TestClassify 참조")
