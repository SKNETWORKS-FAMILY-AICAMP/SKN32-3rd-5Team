"""④ 근거 검증 노드 — **이 프로젝트의 핵심이다.**

설계 근거: 02 §2 · §9 · D-05 · 05 §4 (④)

    문장별로 `근거있음` / `근거없음` / `모순` 을 판정하고,
    **판정에 따른 조치는 코드가 한다** — 문장 제거 · 재검색 1회 · 거절.

    애매하면 `근거없음` 쪽으로 판정한다. 놓친 환각이 나가는 것보다 낫다.

    ⚠️ **2026-08-03까지 이 노드가 LLM을 한 번도 부르지 않았다.** `Task.VERIFY`
    프롬프트·태스크 정의는 있었지만 호출부가 없어서, ④를 파인튜닝해도 그
    결과를 쓸 자리가 없었다(05 §4가 ④를 "핵심"이라 부른 것과 정반대 상태).
    `_llm_judge_sentence` 가 그 호출부다 — LLM 우선, 실패·미설정이면
    2-gram 폴백(`_judge_sentence`)으로 내려간다 (05 §6과 같은 패턴).
"""

from __future__ import annotations

import logging
import re

from ..state import GraphState

log = logging.getLogger(__name__)

VERDICTS = ("근거있음", "근거없음", "모순")


def _llm_judge_sentence(sentence: str, context: str) -> str | None:
    """LLM으로 판정한다. 모델이 없거나 실패·목록 밖 응답이면 `None`.

    `None`이면 부르는 쪽이 `_judge_sentence`(2-gram) 폴백으로 내려간다.
    """
    from ...models.serving.factory import get_client
    from ...models.tasks import Task

    client = get_client()
    if client is None:
        return None
    try:
        user_input = f"문장: {sentence}\n\n근거 문서:\n{context}"
        raw = client.run(Task.VERIFY, user_input, max_tokens=16).strip()
    except Exception as e:  # noqa: BLE001
        log.warning("verify LLM 호출 실패 — 2-gram 폴백: %s", type(e).__name__)
        return None
    for v in VERDICTS:
        if v in raw:
            return v
    log.info("VERIFY 응답이 허용목록 밖이다: %r — 2-gram 폴백", raw[:40])
    return None


#: 재검색은 1회까지 (02 §2). 무한 루프를 막는다.
MAX_RETRY = 1

#: 근거 있음으로 판정하는 최소 2-gram 일치율. 낮게 설정 — 애매하면 근거없음으로 떨어뜨린다.
_GROUND_THRESHOLD = 0.3


def _split_sentences(text: str) -> list[str]:
    """문장 단위로 자른다. 마침표·물음표·느낌표 뒤에서 분리."""
    parts = re.split(r"[.!?。]\s*", text)
    return [p.strip() for p in parts if p.strip()]


def _char_ngrams(text: str, n: int = 2) -> set[str]:
    """의미 있는 문자 n-gram 만 뽑는다 (공백·구두점 제외)."""
    grams: set[str] = set()
    for i in range(len(text) - n + 1):
        ng = text[i : i + n]
        if not any(c in ng for c in " .,!?~—\n"):
            grams.add(ng)
    return grams


def _judge_sentence(sentence: str, context: str) -> str:
    """한 문장이 context 로 뒷받침되는지 판정. 애매하면 근거없음."""
    if not context or not sentence:
        return "근거없음"

    ngrams = _char_ngrams(sentence)
    if not ngrams:
        return "근거없음"

    matches = sum(1 for ng in ngrams if ng in context)
    ratio = matches / len(ngrams)

    if ratio < _GROUND_THRESHOLD:
        return "근거없음"
    return "근거있음"


def verify_grounding(state: GraphState) -> GraphState:
    """초안의 각 문장이 검색된 근거로 뒷받침되는지 검사한다.

    ⚠️ **문장 제거를 여기서 한다** (2026-08-03 추가). 예전에는 판정만 하고
    끝나서, 일부 문장만 근거없음/모순이어도 그 문장이 그대로 최종 답변에
    나갔다 — 전부 근거없음일 때의 거절 경로만 있고, "부분적으로 근거
    없는 문장을 뺀다"는 원래 설계(모듈 머리말 "문장 제거")가 미구현이었다.
    `근거있음`이 아닌 문장은 여기서 `draft`에서 걸러낸다. `simplify` 는
    이 필터를 거친 `draft` 만 본다 — 따로 손댈 필요가 없다.

    Returns:
        `{"verdicts": [...], "draft": (근거있음만 남긴 텍스트), "retry_count": n}`.
        전부 근거없음/모순이고 재검색도 끝났으면
        `{"status": "refused", "refusal_reason": "검증실패"}`.
    """
    draft = state.get("draft", "")
    context = state.get("context", "")
    retry_count = state.get("retry_count", 0)

    sentences = _split_sentences(draft) or ([draft.strip()] if draft.strip() else [])

    verdicts: list[dict[str, str]] = []
    for sentence in sentences:
        verdict = _llm_judge_sentence(sentence, context) or _judge_sentence(sentence, context)
        verdicts.append({"sentence": sentence, "verdict": verdict})

    # 전부 근거없음·모순(=근거있음이 하나도 없음) + 재검색 상한 도달 → 거절.
    # ⚠️ 예전에는 "근거없음"만 셌다 — 전부 "모순"인 답도 거절 없이 통과했다.
    all_ungrounded = bool(verdicts) and all(v["verdict"] != "근거있음" for v in verdicts)
    if all_ungrounded and retry_count >= MAX_RETRY:
        log.warning("verify_grounding: 전 문장 근거없음/모순 + 재검색 상한 도달 → 거절")
        return {  # type: ignore[typeddict-item]
            "status": "refused",
            "refusal_reason": "검증실패",
            "verdicts": verdicts,
        }

    # 근거있음만 남긴다 — 근거없음·모순 문장은 최종 답변에서 뺀다.
    # all_ungrounded 인데 재검색 여지가 남은 경우(→ retry)는 draft를 건드리지
    # 않는다. 어차피 simplify 로 안 가고 retrieve 로 되돌아간다.
    if not all_ungrounded:
        grounded = [v["sentence"] for v in verdicts if v["verdict"] == "근거있음"]
        draft = ". ".join(grounded)
        if draft and not draft.endswith("."):
            draft += "."

    return {  # type: ignore[typeddict-item]
        "verdicts": verdicts,
        "draft": draft,
        "retry_count": retry_count,
    }
