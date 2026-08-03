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

from ...models.tasks import SPECS, Task
from ..fallbacks import note_fallback
from ..state import GraphState

log = logging.getLogger(__name__)

#: 허용 라벨. LLM 출력이 여기 없으면 폴백한다 (05 §4).
#:
#: ⚠️ **프롬프트와 같은 것을 본다** (D-73 · D-22). 손으로 적어 두면
#: *"코드는 아는데 모델은 모르는 목록"* 이 생기고, 실제로 그랬다 —
#: 모델이 `'위험성우려'` 를 내고 코드가 전부 `unknown` 으로 걸러 거절이 됐다.
#: 여기 `" "` 가 들어가 있던 사고(2026-08-02)도 두 곳에 적혀 있어서 안 드러났다.
ALLOWED_INTENTS = SPECS[Task.CLASSIFY].labels


def _mentions_substance(question: str) -> bool:
    """문장에 **코퍼스 물질이나 별칭이 낱말로** 있나.

    규칙은 `vocabulary.mention_in` 한 곳에 있다 — ②슬롯이 쓰는 것과 **같은 함수**다.
    예전에는 둘이 각자 규칙을 들고 있었고, 슬롯이 잡은 질의(*"우유를 줘도 되나요"*)가
    분류에서 죽어 **범위밖 거절**이 됐다 (P2).

    `assumptions=False` — 종을 모르는 이 단계에서 추정 별칭(`프라이팬`·`냄비`)을
    끌어오면 종과 무관하게 중독이 된다. 추정은 ②가 종을 안 뒤에 본다.
    """
    from ...compute.vocabulary import mention_in

    return mention_in(question, assumptions=False) is not None


#: 키워드 폴백 — LLM 을 못 부를 때만 사용.
#: 순서가 중요하다: 중독 키워드가 증상 키워드보다 먼저 매칭돼야 한다
#: ("초콜릿 먹고 구토" 는 intoxication 이 우선).
_KEYWORDS: tuple[tuple[str, tuple[str, ...]], ...] = (
    (
        "intoxication",
        (
            "먹었",
            "먹은",
            "먹어",
            "먹고",
            "섭취",
            "삼켰",
            "중독",
            "독",
            "핥",
            "쪼",
            "물었",
            "씹",
            "태웠",
            "마셨",
        ),
    ),
    (
        "symptom",
        (
            "구토",
            "토하",
            "설사",
            "기침",
            "발작",
            "증상",
            "떨",
            "축 처",
            "처져",
            "무기력",
            "침을 흘",
            "침 흘",
            "기운이 없",
            "기운 없",
            "쓰러",
            "부풀",
            "아파",
            "헛구역질",
            "빵빵",
        ),
    ),
    ("nutrition", ("사료", "먹이", "급여", "먹여", "열량", "영양", "간식", "단백질", "칼슘")),
)


def _has(question: str, intent: str) -> bool:
    """그 의도의 키워드가 하나라도 있나."""
    return any(kw in question for _intent, kws in _KEYWORDS if _intent == intent for kw in kws)


def _keyword_classify(question: str) -> str:
    """키워드 기반 폴백 분류. 매칭이 없으면 'general' — D-46 상 도메인 밖 신호.

    ⚠️ **물질명을 여기 나열하지 않는다.** 흡수 전에는 `초콜릿·포도·양파·자일리톨·
    백합·알로에` 6개가 박혀 있었고, 그래서

        "앵무새 앞에서 프라이팬을 태웠어요"  → general → **범위밖 거절**
        "강아지가 부동액을 핥았어요"        → general → **범위밖 거절**

    ②를 아무리 고쳐도 **①에서 죽었다.** 어휘의 단일 출처는 `compute.vocabulary`
    (533종) + `compute.aliases`(51행)이고, 여기서도 그것을 본다 (D-22 · P2).
    """
    # **먹지 "않는" 것은 섭취가 아니라 증상이다.** `먹어`·`먹은` 이 넓어서
    # *"밥을 안 먹어요"* 가 중독으로 잡혔다 — 부정 표현을 먼저 가른다.
    if any(x in question for x in ("안 먹", "안먹", "못 먹", "못먹")):
        return "symptom"

    # ② 급여·영양 질문이 먼저다 — `단백질`·`칼슘` 은 코퍼스 물질이라
    #    물질 언급으로 보면 전부 중독이 된다 (*"하루 단백질 얼마나"* → 중독).
    if _has(question, "nutrition"):
        return "nutrition"

    # ③ **물질을 말했으면 중독이 이긴다.** *"초콜릿 먹고 구토해요"* 는 증상이 아니라
    #    중독 질의다 — 원래 주석이 정한 우선순위이고, 어휘를 보게 되면서 지켜진다.
    if _mentions_substance(question) or _has(question, "intoxication"):
        return "intoxication"

    for intent, keywords in _KEYWORDS:
        if any(kw in question for kw in keywords):
            return intent
    # 행동 키워드가 없어도 **물질을 말했으면** 중독 질의로 본다.
    # 어휘 밖이면 안 걸리므로 도메인 밖 질의는 그대로 `general` 이다 (D-46).
    if _mentions_substance(question):
        return "intoxication"
    return "general"


def _llm_classify(question: str) -> str | None:
    """LLM 호출. **모델이 없거나** 실패하면 None.

    ⚠️ 예전에는 `APIClient()` 를 여기서 직접 만들었다. 그래서 `configs` 의
    `model.*` 절이 서빙에 안 닿았고 04 비교군 C·D 를 돌릴 방법이 없었다 (D-65).
    """
    from ...models.serving.factory import get_client

    client = get_client()
    if client is None:
        note_fallback(Task.CLASSIFY)
        return None

    try:
        raw = client.run(Task.CLASSIFY, question, max_tokens=16)
        return raw.strip().lower()
    except Exception as e:
        log.warning("classify LLM 호출 실패 — 키워드 폴백: %s", type(e).__name__)
        note_fallback(Task.CLASSIFY)
        return None


def classify_intent(state: GraphState) -> GraphState:
    """질문을 의도·위험으로 분류한다.

    `general` 은 **"우리가 다루지 않는 질문"** 이다 — 이름 짓기·훈련·보험·브랜드 추천.
    이 경우 라우팅이 **검색하지 않고** `refused / 범위밖` 으로 보낸다 (D-46).
    검색해 봐야 관련 없는 청크가 0.5대로 딸려 오기 때문이다.

    Returns:
        `{"intent": ...}` 만. 목록 밖이면 `intent="unknown"`.

    ⚠️ 예전에는 `{"intent": intent, "risk": intent}` 로 **같은 값을 두 키에** 넣었다.
        `risk` 를 읽는 곳은 어디에도 없었고, 값이 `intent` 의 사본이라 읽을 것도 없었다.
        상태에 남은 안 읽히는 키는 *"누군가 쓰고 있겠지"* 로 보여 지우기 어려워진다.
    """
    question = state.get("question", "")

    # ① LLM 우선 (05 §4 — 자연어 의도 파악은 LLM 담당)
    intent = _llm_classify(question)

    # ② 허용목록 검증이 **폴백보다 먼저다** — 코드가 강제한다 (05 §4).
    #
    #    🔴 2026-08-03 까지 순서가 반대였다. 모델이 `'intoxication.'`(마침표)이나
    #       `'위험성우려'` 같은 목록 밖 문자열을 내면 `intent is None` 이 아니므로
    #       **키워드 폴백을 건너뛰고** 그대로 `unknown` 이 됐고, `_after_classify` 가
    #       `unknown` 을 `refuse_scope`(범위밖 거절)로 보냈다.
    #       *"강아지가 부동액을 핥았어요"* 가 라벨 한 글자 때문에 거절된다.
    #
    #       05 §6.1 이 한 절을 통째로 써서 금지한 동작이다 —
    #       *"LLM 출력이 한 번 흔들린 것이 곧 사용자 거절이 되면 안 된다."*
    #       이 모듈 머리말도 폴백 순서를 *"목록 밖 → 키워드 매칭"* 으로 적어 두었는데
    #       코드가 그 순서를 안 지키고 있었다. **지어낸 라벨은 없는 것으로 친다.**
    if intent is not None and intent not in ALLOWED_INTENTS:
        log.warning("intent 허용목록 밖: %r → 키워드 폴백으로 내린다", intent)
        intent = None

    # ③ 폴백 — 키워드 매칭
    if intent is None:
        intent = _keyword_classify(question)

    if intent not in ALLOWED_INTENTS:
        log.warning("키워드 폴백도 목록 밖: %r → 'unknown'", intent)
        intent = "unknown"

    return {"intent": intent}  # type: ignore[typeddict-item]
