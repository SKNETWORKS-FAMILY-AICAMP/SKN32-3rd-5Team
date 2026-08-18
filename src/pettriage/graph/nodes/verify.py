"""④ 근거 검증 노드 — **이 프로젝트의 핵심이다.**

설계 근거: 02 §2 · §9 · D-05 · 05 §4 (④)

    문장별로 `근거있음` / `근거없음` / `모순` 을 판정하고,
    **판정에 따른 조치는 코드가 한다** — 재검색 1회 · 거절.

    🔴 **조치 「문장 제거」는 폐기됐다 (04b §3.4 · 02 §9).**

        2026-08-03  이 조치가 구현됐다 — 그때는 머리말이 명시한 설계였다
        2026-08-05  04b 측정에서 폐기 판정이 났다
        2026-08-18  머지로 들어온 것을 발견해 되돌렸다

    **폐기가 문서에만 적히고 이 머리말은 안 고쳐졌다.** 그래서 이미 있던 구현이
    아무 저항 없이 들어왔다. 사유를 문서가 아니라 여기 적는 이유가 그것이다.

        근거없음으로 떨어지는 문장의 다수가 **등급이 시킨 행동 권고**다.
        *"지금 바로 동물병원으로 가세요"* 는 코퍼스에서 온 문장이 아니라
        **코드가 등급에서 만든 문장**이라 2-gram 일치율이 낮다(실측 0.304).
        제거를 켜면 **등급 4인데 병원 가라는 문장이 사라진 답변**이 나간다.
        `none` 판에서 못 채운 문구가 즉시×7 · 전화×5 · 병원×4 였다.

    제대로 하려면 **등급 파생 문장을 코드가 식별해 제거 대상에서 빼야 한다**
    (04b §3.4 의 범주 셋 · §7-5). 그 전까지 이 조치는 켜지 않는다.

    애매하면 `근거없음` 쪽으로 판정한다. 놓친 환각이 나가는 것보다 낫다.

    ⚠️ **2026-08-03까지 이 노드가 LLM을 한 번도 부르지 않았다.** `Task.VERIFY`
    프롬프트·태스크 정의는 있었지만 호출부가 없어서, ④를 파인튜닝해도 그
    결과를 쓸 자리가 없었다(05 §4가 ④를 "핵심"이라 부른 것과 정반대 상태).
    `_llm_judge_sentence` 가 그 호출부다. 실패·미설정이면 2-gram 폴백
    (`_judge_sentence`)으로 내려간다 (05 §6과 같은 패턴). 둘 다 있으면
    `_combined_verdict` 가 합친다 — **2-gram이 바닥, LLM은 조이기만
    한다**(2026-08-03 합의). LLM의 `근거있음`이 2-gram의 `근거없음`을
    못 뒤집는다 — LLM의 관대한 오판 하나가 환각을 그대로 통과시키는
    사고를 막는다.
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


def _combined_verdict(sentence: str, context: str) -> str:
    """LLM과 2-gram을 합친다 — **2-gram이 바닥, LLM은 조이기만 한다.**

    합의(한빈·이서은, 2026-08-03): LLM이 `근거있음`이라고 해도 2-gram이
    `근거없음`이면 2-gram이 이긴다 — LLM의 관대한 판정이 2-gram의 신중함을
    못 뒤집는다. 반대로 LLM이 `모순`·`근거없음`(이미 엄격한 방향)이면
    그대로 믿는다 — 2-gram은 `모순` 개념이 없어 그 방향에서는 비교할
    상대가 못 된다. LLM이 아예 없거나 실패하면 2-gram 단독으로 돈다
    (기존 폴백과 동일).
    """
    llm_verdict = _llm_judge_sentence(sentence, context)
    if llm_verdict is None:
        return _judge_sentence(sentence, context)
    if llm_verdict == "근거있음" and _judge_sentence(sentence, context) == "근거없음":
        return "근거없음"
    return llm_verdict


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

    🔴 **`draft` 를 건드리지 않는다.** 일부 문장이 근거없음/모순이어도 그대로
    통과시킨다 — 문장 제거는 폐기된 조치다(모듈 머리말 참조). 부분 근거없음의
    대응은 **제거가 아니라 표시**이며, 그 배선(`[모델 서술]` 태그 · 04b §2.2)은
    아직 안 됐다. **지금은 판정만 남기고 조치는 재검색·거절 둘뿐이다.**

    Returns:
        `{"verdicts": [{"sentence": ..., "verdict": ...}], "retry_count": n}`.
        전부 근거없음/모순이고 재검색도 끝났으면
        `{"status": "refused", "refusal_reason": "검증실패"}`.
    """
    draft = state.get("draft", "")
    context = state.get("context", "")
    retry_count = state.get("retry_count", 0)

    sentences = _split_sentences(draft) or ([draft.strip()] if draft.strip() else [])

    verdicts: list[dict[str, str]] = []
    for sentence in sentences:
        verdict = _combined_verdict(sentence, context)
        verdicts.append({"sentence": sentence, "verdict": verdict})

    # 전부 근거없음/모순(=근거있음이 하나도 없음) → **실패를 세운다. 상한 판단은 하지 않는다.**
    #
    # ⚠️ 예전에는 "근거없음"만 셌다 — 전부 "모순"인 답도 거절 없이 통과했다 (2026-08-03).
    #
    # 🔴 그리고 그날까지 이 조건에 `and retry_count >= MAX_RETRY` 가 붙어 있었고,
    #    그래서 **재검색이 한 번도 발동할 수 없었다** —
    #
    #      · `verify` 가 `refused` 를 세우려면 `retry_count >= 1` 이어야 하고
    #      · `retry_count` 를 1로 만드는 `_retry` 노드에 가려면 `refused` 여야 한다
    #
    #    두 조건이 상호배타라 `retry_count` 는 영원히 0이었다. 따라서
    #    **④의 조치(당시 셋 — 문장 제거·재검색·거절) 중 어느 것도 일어난 적이 없고**,
    #    05 §5 가 랭그래프를 고른 유일한 근거인 `retry → retrieve` 엣지가
    #    죽은 코드였다. 전 문장이 근거없음이어도 그대로 답변으로 나갔다.
    #
    #    `build.py` 머리말이 이미 옳은 설계를 적어 두었다 —
    #    *"상한 판단은 라우터 한 곳이고 `verify_grounding` 은 판정만 한다"*.
    #    여기서 상한을 다시 보던 것이 그 결정을 어기고 있었다.
    all_ungrounded = bool(verdicts) and all(v["verdict"] != "근거있음" for v in verdicts)
    if all_ungrounded:
        log.warning(
            "verify_grounding: 전 문장 근거없음/모순 (retry_count=%d) → 재검색 또는 거절",
            retry_count,
        )
        return {  # type: ignore[typeddict-item]
            "status": "refused",
            "refusal_reason": "검증실패",
            "verdicts": verdicts,
        }

    # 🔴 여기서 근거있음만 남겨 `draft` 를 다시 쓰지 않는다 (2026-08-18 되돌림).
    #    폐기 사유는 모듈 머리말에 적었다 — 지우면 등급이 시킨 행동 권고가
    #    함께 사라진다. 다시 넣고 싶어지면 **등급 파생 문장 식별이 먼저다.**
    return {  # type: ignore[typeddict-item]
        "verdicts": verdicts,
        "retry_count": retry_count,
    }
