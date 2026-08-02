"""④ 근거 검증 노드 — **이 프로젝트의 핵심이다.**

설계 근거: 02 §2 · §9 · D-05 · 05 §4 (④)

    문장별로 `근거있음` / `근거없음` / `모순` 을 판정하고,
    **판정에 따른 조치는 코드가 한다** — 문장 제거 · 재검색 1회 · 거절.

    애매하면 `근거없음` 쪽으로 판정한다. 놓친 환각이 나가는 것보다 낫다.
"""

from __future__ import annotations

import logging
import re

from ..state import GraphState

log = logging.getLogger(__name__)

VERDICTS = ("근거있음", "근거없음", "모순")

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

    Returns:
        `{"verdicts": [{"sentence": ..., "verdict": ...}], "retry_count": n}`.
        전부 `근거없음` 이고 재검색도 끝났으면
        `{"status": "refused", "refusal_reason": "검증실패"}`.
    """
    draft = state.get("draft", "")
    context = state.get("context", "")
    retry_count = state.get("retry_count", 0)

    sentences = _split_sentences(draft) or ([draft.strip()] if draft.strip() else [])

    verdicts: list[dict[str, str]] = []
    for sentence in sentences:
        verdict = _judge_sentence(sentence, context)
        verdicts.append({"sentence": sentence, "verdict": verdict})

    # 전부 근거없음 + 재검색 상한 도달 → 거절 (검증실패).
    all_ungrounded = bool(verdicts) and all(v["verdict"] == "근거없음" for v in verdicts)
    if all_ungrounded and retry_count >= MAX_RETRY:
        log.warning("verify_grounding: 전 문장 근거없음 + 재검색 상한 도달 → 거절")
        return {  # type: ignore[typeddict-item]
            "status": "refused",
            "refusal_reason": "검증실패",
            "verdicts": verdicts,
        }

    return {  # type: ignore[typeddict-item]
        "verdicts": verdicts,
        "retry_count": retry_count,
    }