"""① 의도·위험 분류 노드 — **WS2 구현 대기.**

설계 근거: 02 §2 · 05 §4 (①)

    LLM 이 라벨을 내고, **코드가 허용목록으로 검증한다.**
    목록에 없는 라벨이 오면 지어낸 것이므로 `unknown` 으로 떨어뜨린다.

⚠️ **이 노드가 도메인 밖 거절을 책임진다** (D-46).

    유사도 임계값이 막아줄 것으로 설계돼 있었으나 **실측에서 성립하지 않았다** —
    "고양이 이름 지어주세요"(0.550) · "강아지 배변 훈련"(0.553) 같은 도메인 밖 질의가
    근거 있는 질의의 최저점(0.547)보다 높은 점수를 받는다.

    **여기서 안 걸러지면 그대로 답으로 나간다.** 뒤에 받쳐줄 것이 없다.
"""

from __future__ import annotations

from ..state import GraphState

#: 허용 라벨. LLM 출력이 여기 없으면 폴백한다 (05 §4).
ALLOWED_INTENTS = ("intoxication", "symptom", "nutrition", "general")


def classify_intent(state: GraphState) -> GraphState:
    """질문을 의도·위험으로 분류한다.

    `general` 은 **"우리가 다루지 않는 질문"** 이다 — 이름 짓기·훈련·보험·브랜드 추천.
    이 경우 **검색하지 않고** `refused / 범위밖` 으로 보낸다 (D-46).
    검색해 봐야 관련 없는 청크가 0.5대로 딸려 오기 때문이다.

    Returns:
        `{"intent": ..., "risk": ...}` 만. 목록 밖이면 `intent="unknown"`.
    """
    raise NotImplementedError("WS2: tests/todo/test_graph_nodes.py::TestClassify 참조")
