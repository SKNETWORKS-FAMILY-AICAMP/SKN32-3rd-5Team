"""② 슬롯 추출 · 되묻기 노드.

설계 근거: 02 §2 · §9 · D-10 · 05 §4 (②)

    **종이 없으면 검색으로 넘어가지 않는다** (D-10).
    포유류 기준을 조류에 적용하는 것이 이 도메인에서 가장 치명적인 오류다.

    LLM 이 슬롯을 뽑고, **코드가 JSON 스키마로 검증**한다.
    발화에 없는 값을 채우면 그게 곧 환각이므로 `null` 로 둔다.
"""

from __future__ import annotations

import logging
import re

from ...compute.vocabulary import SPECIES_WORDS, is_word_hit, mention_in
from ...models.tasks import SPECS, Task
from ..fallbacks import note_fallback
from ..state import GraphState, set_substance

log = logging.getLogger(__name__)

#: 의도별 필수 슬롯. 조류는 정량 임계치가 0건이라 체중·섭취량을 요구하지 않는다 (D-09).
REQUIRED_SLOTS = {
    "intoxication": ("species", "substance"),
    "symptom": ("species",),
    "nutrition": ("species",),
    "general": ("species",),
}

#: 🔴 **물질을 말한 것이 아니다.** LLM 이 슬롯을 비우지 않고 이런 말을 채워 넣는다.
#:
#: 이 목록이 없으면 *"뭔가 이상한 걸 먹었어요"* 가 **물질을 말한 것**으로 읽혀
#: `근거없음` 거절로 간다. 되물어야 할 것을 거절하는 것이고, 방향이 반대다 —
#: `없음`(사용자가 말 안 함)과 `모름`(말했는데 우리가 모름)을 가르는 것이 아래
#: `_unknown_substance` 인데, 그 판단의 입력이 오염되면 판단 전체가 뒤집힌다.
#:
#: 부분 일치로 본다 — `"뭔가 하얀 가루"` 처럼 꾸밈말이 붙어 온다.
_VAGUE_SURFACES: tuple[str, ...] = (
    "뭔가", "무언가", "뭘까", "뭔지", "미상", "모름", "모르겠", "알 수 없",
    "이상한", "정체불명", "unknown",
)


def _is_vague(surface: str | None) -> bool:
    """표면형이 **물질 이름이 아니라 모른다는 말**인가."""
    if not surface:
        return True
    return any(v in surface for v in _VAGUE_SURFACES)


#: 종 키워드. **이름·품종은 여기 넣지 않는다** — 이름에서 종을 추측하면 환각이다.
#: 종 표기의 단일 출처는 `compute.vocabulary` 다 (P2 · D-22).
#:
#: ⚠️ 예전에는 이 표가 여기 있었다. 그래서 **물질 어휘 쪽에서 종을 걸러낼 방법이
#: 없었고**, 코퍼스 이름을 쪼갤 때 떨어져 나온 `고양이` 가 물질로 잡혔다 (D-67).
_SPECIES_KEYWORDS = SPECIES_WORDS


#: 체중 추출 정규식 — "5kg", "5킬로그램", "3.5 kg" 등.
_WEIGHT_RE = re.compile(r"(\d+(?:\.\d+)?)\s*(?:kg|킬로|킬로그램)", re.IGNORECASE)

#: 섭취량 추출 정규식 — "30g", "30 그램" 등. **kg 은 제외**.
_AMOUNT_RE = re.compile(r"(\d+(?:\.\d+)?)\s*(?:g(?![a-z])|그램)", re.IGNORECASE)


def _extract_species(question: str) -> str | None:
    """종 키워드가 **낱말로** 있을 때만 반환. **이름·품종에서는 추측하지 않는다.**

    ⚠️ 부분 문자열로 보면 `"개"` 가 터진다. 흡수 전 실측 —

        고양이가 사과 3개 먹었어요      → dog   ← "3개"
        앵무새가 개봉한 초콜릿을 쪼았어요  → dog   ← "개봉"
        고양이 3개월인데 …             → dog   ← "3개월"

    **포유류 기준이 조류에 적용되는 것이 이 도메인에서 가장 치명적인 오류다** (D-10).
    그래서 별칭 표가 쓰는 것과 **같은 낱말 경계 규칙**을 쓴다 (D-60) — 규칙을
    새로 만들지 않는다. `"3개"` 는 앞이 숫자라, `"개봉"` 은 뒤가 조사가 아니라 탈락한다.
    """
    for species, keywords in _SPECIES_KEYWORDS.items():
        if any(is_word_hit(question, kw) for kw in keywords):
            return species
    return None


def _normalize_species(value: object) -> str | None:
    """LLM 이 낸 종 값을 코드가 쓰는 `dog·cat·bird` 로 올린다 (D-86).

    🔴 2026-08-03 실측 — 모델이 여섯 건 전부 `'개'`·`'고양이'` 를 냈고 코드는
       `("dog","cat","bird")` 만 받아 **전부 버렸다.** 그러고는 `llm.get("species")`
       가 truthy 라서 **키워드 폴백도 건너뛰었다** — 오늘 아침 `classify.py` 에서
       고친 것과 같은 구조가 여기 남아 있었다.

       프롬프트에 스키마를 실었으니 대부분 `dog` 로 올 것이다. 그래도 이 함수를
       두는 이유는 **모델 출력에 기대지 않기 위해서**다 — 종을 잘못 읽으면
       포유류 기준이 조류에 적용된다 (D-10, 이 도메인에서 가장 치명적인 오류).

    ⚠️ 어휘는 `vocabulary.SPECIES_WORDS` 한 곳에서 온다. 여기 다시 적지 않는다 (D-67).
    """
    if not isinstance(value, str) or not value.strip():
        return None
    v = value.strip().lower()
    if v in ("dog", "cat", "bird"):
        return v
    for code, words in _SPECIES_KEYWORDS.items():
        if any(w in value for w in words):
            return code
    return None


def _off_schema_keys(raw: dict | None) -> list[str]:
    """모델이 **스키마 밖 이름으로 낸 키.** 비어 있으면 스키마를 지킨 것이다.

    이것을 세는 이유 — 오늘의 사고는 *"모델이 답했는데 코드가 못 읽었다"* 인데
    **폴백 집계가 그것을 못 잡았다.** JSON 파싱은 성공했으니 `note_fallback` 이
    안 불렸고, 리포트는 *"3태스크 전부 모델 100%"* 를 찍었다.

    D-82 가 *"안 부른 태스크를 성공으로 세지 않는다"* 를 고쳤다면, 여기는
    **"부르고 버린 태스크를 성공으로 세지 않는다"** 이다. 같은 종류의 거짓 신호다.
    """
    if not isinstance(raw, dict):
        return []
    allowed = set(SPECS[Task.SLOT].output_keys)
    return sorted(k for k in raw if k not in allowed)


def _extract_substance_fallback(question: str, species: str | None) -> str | None:
    """**LLM 이 없을 때의 폴백.** 코퍼스 어휘와 별칭 표에서 표면형을 찾는다.

    ⚠️ 이것은 ②가 아니라 ②의 대역이다. 05 §4 는 슬롯 추출을 **LLM 에 맡긴 5개 지점**
    중 하나로 정했고, 04 §3 이 그 태스크의 지표를 잰다. 폴백을 ②로 굳히면
    **잴 태스크가 없어진다.**

    규칙은 `vocabulary.mention_in` **한 곳**에 있다 — ①분류도 같은 것을 본다 (P2).

    ⚠️ **폴백이 못 잡는 것**: 코퍼스 이름·별칭이 문장에 글자 그대로(또는 코퍼스가
    괄호 안에 열거한 이름으로) 있어야 한다. 그 밖은 명사구 추출이 필요하고
    그것이 ②를 LLM 에 맡긴 이유다. **못 잡으면 되묻는다** — 거절이 아니다 (D-49).
    """
    return mention_in(question, species)


def _resolvable(surface: str, species: str | None) -> bool:
    """이 표면형이 **폐쇄 목록 위로 올라갈 수 있는가.**

    이름이 서거나(직접·별칭·부분일치) 후보가 여럿이거나(모호 · D-62),
    종만 안 맞는 경우(종밖 · D-68) 전부 **안다**로 본다. 셋 다 다음 행선지가
    있고, 그 행선지들이 이미 설계되어 있다.

    ⚠️ `mention_in` 을 여기서 부르지 않는다. 그 함수는 **문장**을 훑는 것이라
       표면형에 쓰면 부분 매칭이 터진다 — 2026-08-03 D-87 사고가 그것이었다.
    """
    from ...compute.vocabulary import resolve_substance

    res = resolve_substance(surface, species)
    if res.name or res.candidates:
        return True
    return bool(species and resolve_substance(surface, None).name)



def _llm_slots(question: str) -> dict | None:
    """② SLOT 태스크. **LLM 이 표면형을 뽑고 코드가 정규화한다** (D-38).

    키가 없거나 실패하면 `None` — 부르는 쪽이 폴백으로 내려간다 (05 §6).
    JSON 파싱 실패도 실패로 본다 (04 §3 의 *JSON 파싱 성공률* 이 그것을 잰다).
    """
    from ...models.serving.factory import get_client
    from ...models.tasks import Task

    client = get_client()
    if client is None:
        note_fallback(Task.SLOT)
        return None
    try:
        import json

        raw = client.run(Task.SLOT, question, max_tokens=200).strip()
        start, end = raw.find("{"), raw.rfind("}")
        if start < 0 or end <= start:
            log.warning("SLOT: JSON 을 찾지 못했다")
            note_fallback(Task.SLOT)
            return None
        got = json.loads(raw[start : end + 1])
        if isinstance(got, dict):
            return got
        note_fallback(Task.SLOT)
        return None
    except Exception as e:  # noqa: BLE001 — 어떤 실패든 폴백으로 내려간다
        log.warning("SLOT LLM 실패: %s", type(e).__name__)
        note_fallback(Task.SLOT)
        return None


def _extract_weight(question: str) -> float | None:
    m = _WEIGHT_RE.search(question)
    return float(m.group(1)) if m else None


def _extract_amount(question: str) -> float | None:
    m = _AMOUNT_RE.search(question)
    return float(m.group(1)) if m else None


def extract_slots(state: GraphState) -> GraphState:
    """자유서술에서 슬롯을 뽑는다.

    Returns:
        `{"slots": ..., "missing_slots": [...]}`.
        발화에 없는 값은 **채우지 않는다**.
    """
    question = state.get("question", "")
    intent = state.get("intent", "general")
    existing = dict(state.get("slots") or {})

    # 새 발화에서 추출한 슬롯. **값이 있을 때만 키를 넣는다** (D-10 · 05 §4 ②).
    new_slots: dict = {}

    # ── ② LLM 이 먼저, 없으면 폴백 (05 §6) ────────────────────
    # **입력 state 를 변형하지 않는다** — 노드는 바뀐 키만 돌려준다 (state.py 머리말).
    extras: dict = {}
    # **`slot_llm_used` 를 여기 두지 않는다.** 상태에만 남고 아무도 안 읽었다.
    # 같은 사실을 `graph/fallbacks.py` 가 다섯 태스크에 대해 한 방식으로 남기고,
    # 그것은 응답과 평가 리포트까지 나간다 (D-22 — 두 곳에 적지 않는다).
    llm = _llm_slots(question) or {}

    # **스키마를 지켰는지 본다** (D-86). 지키지 않았으면 아래 코드가 값을 못 읽는다.
    off = _off_schema_keys(llm)
    if off:
        log.warning("SLOT: 스키마 밖 키 %s — 그 값은 버려진다 (D-86)", off)
        # 🔴 **아는 키가 하나도 없으면 모델을 안 쓴 것과 같다.** 폴백으로 센다 —
        #    JSON 파싱이 성공했다는 이유로 성공에 넣으면 지표가 거짓말을 한다.
        if not any(k in llm for k in SPECS[Task.SLOT].output_keys):
            note_fallback(Task.SLOT)

    # **정규화가 폴백보다 먼저다** (D-86). 예전에는 `llm.get("species") or 폴백` 이라
    # 모델이 `'개'` 를 내면 truthy 라서 **폴백을 건너뛰고 그대로 버려졌다.**
    species = _normalize_species(llm.get("species")) or _extract_species(question)
    if species:
        new_slots["species"] = species
    elif llm.get("species"):
        log.info("SLOT: 종을 못 올렸다 %r — 폴백도 못 잡았다 (05 §6 ①)", llm.get("species"))

    weight = _as_float(llm.get("weight_kg")) or _extract_weight(question)
    if weight is not None:
        new_slots["weight_kg"] = weight

    amount = _as_float(llm.get("amount_g")) or _extract_amount(question)
    if amount is not None:
        new_slots["amount_g"] = amount

    # 기존 슬롯과 병합 — 새 값이 우선하지만 없는 값은 지우지 않는다.
    merged = {**existing, **new_slots}
    sp = merged.get("species")

    # ── 물질: **표면형을 뽑고 코드가 정규화한다** (D-38 · D-62) ──
    #
    # `set_substance` 가 유일한 문이다. 폐쇄 목록 밖이면 키가 안 생기고,
    # 추정 별칭을 탔으면 `substance_is_assumed` 가 함께 선다 (D-59 ⑤).
    # 🔴 **성공 판정을 "LLM 이 문자열을 냈다" 에서 "폐쇄 목록에 오른다" 로 옮긴다** (D-88).
    #
    #   D-86 으로 ②슬롯이 진짜 일하기 시작하자 **그 성공이 폴백을 껐다.**
    #   전에는 모델이 `concern`·`item` 같은 키를 내서 `llm_surface` 가 `None` 이었고,
    #   그래서 문장 스캔이 돌아 `세제 거품` 안의 `세제` 를 잡았다. 이제는 모델이
    #   구(句)를 정확히 뽑아 오고, 그 값이 폴백을 가린 채 폐쇄 목록에서 떨어진다 —
    #   D-85 가 **아는 물질을 모른다고** 거절한다. 60건 실측 4건:
    #
    #       세제 거품(G-019) · 니코틴 껌 조각(G-107) · 양파국(G-048) · 감기약(G-022)
    #
    #   `mention_in` 은 **문장**용이다. 표면형에 쓰면 부분 매칭이 터진다(D-87).
    #   그러니 표면형이 아니라 **질문 문장**으로 내려간다 — 함수를 원래 용도로 쓴다.
    #
    #   ⚠️ 모호어(`뭔가`)는 내려가지 않는다. 사용자가 *모른다*고 말한 것이고,
    #      문장을 뒤져 물질을 세우면 그게 추정이다 (D-49).
    llm_surface = llm.get("substance")
    surface = llm_surface or _extract_substance_fallback(question, sp)
    if llm_surface and not _is_vague(llm_surface) and not _resolvable(llm_surface, sp):
        inner = _extract_substance_fallback(question, sp)
        if inner and inner != llm_surface:
            log.info(
                "SLOT: 표면형 %r 이 폐쇄 목록에 못 오른다 — 문장에서 %r 로 내려간다 (D-88)",
                llm_surface,
                inner,
            )
            surface = inner

    # ⚠️ **표면형을 못 찾았을 때, 종 때문인지 확인한다** (D-68).
    #
    #   `mention_in` 은 종으로 어휘를 좁힌다 — 조류 자료를 고양이에 적용하지 않기
    #   위해서다 (D-10). 그래서 `향초`(covers={'bird'})는 고양이 질의에서 **표면형
    #   단계부터 안 잡힌다.** 그 결과가 *"물질을 모른다"* 로 뭉뚱그려져 되묻기로 갔다.
    #
    #   실제 상황은 *"물질은 알겠는데 이 종에 자료가 없다"* 이고, 되물어도 답이
    #   안 나온다. **응급 상황에서 못 쓸 질문은 거절보다 나쁘다.** 근거없음으로 보낸다.
    #
    #   슬롯에는 **넣지 않는다** — 넣으면 조류 근거로 고양이에게 답하게 된다 (D-10).
    if not surface and sp:
        from ...compute.vocabulary import mention_in

        off = mention_in(question, None, assumptions=False)
        if off:
            extras["off_species_substance"] = off
            log.info("물질 %r 은 코퍼스에 있으나 종 %r 자료가 없다 — 근거없음 (D-68).", off, sp)

    if surface or "substance" not in merged:
        merged = set_substance(merged, surface, sp)

    # 물질을 못 올린 이유를 **셋으로 가른다.** 셋의 다음 행선지가 서로 다르다 (D-68).
    #
    #   모호      후보가 여럿이다        → 진행한다. 후보를 전부 검색어로 넘긴다 (D-62)
    #   종밖      물질은 아는데 이 종에 자료가 없다 → **근거없음 거절** (D-10)
    #   모름      말했는데 코퍼스에 없다   → **근거없음 거절** (D-85)
    #   없음      아무것도 못 찾았다      → 되묻는다 (D-49)
    #
    # 예전에는 셋을 전부 `결측 → 되묻기` 로 뭉쳤다. 그래서
    #   · `세제`(후보 2건)가 되묻기로 빠졌다 — D-62 가 *"모호는 실패가 아니다"* 라고
    #     정해 뒀는데 **후보를 실어 놓고도 결측으로 셌다.** 결정의 절반이 미구현이었다
    #   · `고양이가 향초를…` 이 되묻기로 빠졌다 — 향초 자료는 `covers={'bird'}` 뿐이라
    #     고양이에는 **근거가 없는 것**이지 물질을 모르는 것이 아니다.
    #     되물어도 답이 안 나온다. 응급 상황에서 못 쓸 질문은 거절보다 나쁘다
    if surface and "substance" not in merged:
        from ...compute.vocabulary import resolve_substance

        res = resolve_substance(surface, sp)
        if res.candidates:
            extras["substance_candidates"] = list(res.candidates)
        elif sp and resolve_substance(surface, None).name:
            # 표면형은 잡혔는데 종 필터에서 떨어진 경우도 같다.
            extras["off_species_substance"] = surface
        elif not _is_vague(surface):
            # 🔴 **네 번째 갈래 — 말했는데 우리가 모른다** (D-85).
            #
            #   `목캔디 · 달팽이약 · 실리카겔 · 매니큐어 · 모기향 · 계피가루`
            #   여섯 건이 여기 걸린다. 코퍼스에 없는 물질인데 시스템은
            #   *"무엇을 먹었나요?"* 를 되물었다. **사용자는 이미 말했다.**
            #
            #   되물어도 같은 답이 돌아온다 — D-68 이 종밖 물질에서 내린 결론과
            #   똑같은 구조다. *"응급 상황에서 못 쓸 질문은 거절보다 나쁘다."*
            #
            #   ⚠️ 슬롯에는 넣지 않는다. 폐쇄 목록 밖 이름이 슬롯에 들어가면
            #      계약(`SubstanceName`)이 막는 환각의 문이 열린다 (D-62).
            extras["unknown_substance"] = surface
            log.info("물질 %r 은 코퍼스에 없다 — 되묻지 않고 근거없음으로 보낸다 (D-85).", surface)

    # 결측 슬롯 판정
    required = REQUIRED_SLOTS.get(intent, ("species",))
    missing = [s for s in required if not merged.get(s)]
    # **모호는 결측이 아니다** (D-62). 후보를 들고 검색으로 간다.
    if "substance" in missing and extras.get("substance_candidates"):
        missing.remove("substance")
    # **종밖도 결측이 아니다** — 되묻기가 아니라 거절로 가야 한다 (D-68).
    if "substance" in missing and extras.get("off_species_substance"):
        missing.remove("substance")
    # **코퍼스에 없는 물질도 결측이 아니다** — 사용자는 말했다 (D-85).
    if "substance" in missing and extras.get("unknown_substance"):
        missing.remove("substance")

    return {"slots": merged, "missing_slots": missing, **extras}  # type: ignore[return-value]


def _as_float(v: object) -> float | None:
    """LLM 이 문자열로 준 수치를 받는다. 못 바꾸면 `None` — **추측하지 않는다.**"""
    try:
        return float(v) if v is not None and str(v).strip() != "" else None
    except (TypeError, ValueError):
        return None


#: 슬롯별 되묻기 문구. 여러 개 결측 시 앞에서부터 우선순위대로 묻는다.
_CLARIFY_QUESTIONS: dict[str, str] = {
    "species": "어떤 동물인가요? (개 · 고양이 · 앵무새)",
    "substance": "무엇을 먹었나요?",
    "weight_kg": "체중이 어떻게 되나요? (kg)",
    "amount_g": "얼마나 먹었나요? (g)",
}


def _compose_clarify(missing: list[str]) -> str:
    """결측 슬롯 목록으로 되묻기 문구를 만든다."""
    if not missing:
        return "추가로 알려주실 수 있나요?"

    parts = [_CLARIFY_QUESTIONS[s] for s in missing if s in _CLARIFY_QUESTIONS]
    if not parts:
        return "추가 정보를 알려주세요."
    return " ".join(parts)


def ask_clarify(state: GraphState) -> GraphState:
    """되묻기 문구를 만든다. 상한은 설정값(`triage.max_clarify_turns`)이다.

    Returns:
        `{"clarify_question": ..., "clarify_turns": n}`.
        상한 초과면 `{"status": "refused", "refusal_reason": "되묻기상한"}`.
    """
    missing = state.get("missing_slots") or []
    turns = state.get("clarify_turns", 0)

    # 설정값 로드 — configs/*.yaml 의 triage.max_clarify_turns.
    try:
        from ...config import get_config

        max_turns = get_config().triage.max_clarify_turns
    except Exception:
        max_turns = 2  # 계약 기본값 (02 §9 · contracts.MAX_CLARIFY_TURNS)

    # 상한 도달 → 거절 (02 §9).
    if turns >= max_turns:
        return {  # type: ignore[typeddict-item]
            "status": "refused",
            "refusal_reason": "되묻기상한",
        }

    return {  # type: ignore[typeddict-item]
        "clarify_question": _compose_clarify(missing),
        "clarify_turns": turns + 1,
    }
