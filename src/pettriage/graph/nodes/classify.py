"""① 의도·위험 분류 노드.

설계 근거: 02 §2 · 05 §4 (①) · D-46

    LLM 이 자연어 의도를 파악하고, **코드가 허용목록으로 검증한다.**
    LLM 출력이 목록 밖이면 지어낸 것이므로 폴백한다.

    폴백 순서:
      ① LLM 호출 실패 · API 키 없음 → 키워드 매칭
      ② 키워드도 매칭 안 됨          → "general"

⚠️ **이 노드가 도메인 밖 거절을 책임진다** (D-46).

    유사도 임계값이 막아줄 것으로 설계돼 있었으나 **실측에서 성립하지 않았다** —
    "고양이 이름 지어주세요"(0.550) · "강아지 배변 훈련"(0.553) 같은 도메인 밖 질의가
    근거 있는 질의의 최저점(0.547)보다 높은 점수를 받는다.

    `general` 은 **"우리가 다루지 않는 질문"** 이다 — 이름 짓기·훈련·보험·브랜드 추천.
    라우팅이 `general` 을 보고 검색을 건너뛰고 `refused / 범위밖` 으로 보낸다.
    **여기서 안 걸러지면 그대로 답으로 나간다.** 뒤에 받쳐줄 것이 없다.
"""

from __future__ import annotations

import logging

from ...models.tasks import Task
from ..state import GraphState

log = logging.getLogger(__name__)

#: 허용 라벨. LLM 출력이 여기 없으면 폴백한다 (05 §4).
ALLOWED_INTENTS = ("intoxication", "symptom", "nutrition", " ")

#: 키워드 폴백 — LLM 을 못 부를 때만 사용.
#: 순서가 중요하다: 중독 키워드가 증상 키워드보다 먼저 매칭돼야 한다
#: ("초콜릿 먹고 구토" 는 intoxication 이 우선).
_KEYWORDS: tuple[tuple[str, tuple[str, ...]], ...] = (
    ("intoxication", ("먹었", "먹은", "먹어", "섭취", "삼켰", "중독", "독",
                      "초콜릿", "포도", "양파", "자일리톨", "백합", "알로에")),
    ("symptom",      ("구토", "설사", "기침", "발작", "증상", "떨", "안먹",
                      "축 처", "아파", "헛구역질", "빵빵")),
    ("nutrition",    ("사료", "먹이", "급여", "열량", "영양", "간식")),
)


def _keyword_classify(question: str) -> str:
    """키워드 기반 폴백 분류. 매칭이 없으면 'general' — D-46 상 도메인 밖 신호."""
    for intent, keywords in _KEYWORDS:
        if any(kw in question for kw in keywords):
            return intent
    return "general"


def _llm_classify(question: str) -> str | None:
    """LLM 호출. API 키 없거나 실패하면 None."""
    from ...config import get_secrets
    if not get_secrets().openai_api_key:
        return None

    try:
        from ...models.serving.client import APIClient
        client = APIClient()
        raw = client.run(Task.CLASSIFY, question, max_tokens=16)
        return raw.strip().lower()
    except Exception as e:
        log.warning("classify LLM 호출 실패 — 키워드 폴백: %s", type(e).__name__)
        return None


def classify_intent(state: GraphState) -> GraphState:
    """질문을 의도·위험으로 분류한다.

    `general` 은 **"우리가 다루지 않는 질문"** 이다 — 이름 짓기·훈련·보험·브랜드 추천.
    이 경우 라우팅이 **검색하지 않고** `refused / 범위밖` 으로 보낸다 (D-46).
    검색해 봐야 관련 없는 청크가 0.5대로 딸려 오기 때문이다.

    Returns:
        `{"intent": ..., "risk": ...}` 만. 목록 밖이면 `intent="unknown"`.
    """
    question = state.get("question", "")

    # ① LLM 우선 (05 §4 — 자연어 의도 파악은 LLM 담당)
    intent = _llm_classify(question)

    # ② 폴백 — 키워드 매칭
    if intent is None:
        intent = _keyword_classify(question)

    # ③ 허용목록 검증 — 코드가 강제한다 (05 §4).
    #    LLM 이 지어낸 라벨은 여기서 걸러진다.
    if intent not in ALLOWED_INTENTS:
        log.warning("intent 허용목록 밖: %r → 'unknown'", intent)
        intent = "unknown"

    return {"intent": intent, "risk": intent}  # type: ignore[typeddict-item]