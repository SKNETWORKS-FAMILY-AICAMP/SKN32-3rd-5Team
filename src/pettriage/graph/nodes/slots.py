"""② 슬롯 추출 · 되묻기 노드.

설계 근거: 02 §2 · §9 · D-10 · 05 §4 (②)

    **종이 없으면 검색으로 넘어가지 않는다** (D-10).
    포유류 기준을 조류에 적용하는 것이 이 도메인에서 가장 치명적인 오류다.

    LLM 이 슬롯을 뽑고, **코드가 JSON 스키마로 검증**한다.
    발화에 없는 값을 채우면 그게 곧 환각이므로 `null` 로 둔다.
"""

from __future__ import annotations

import re

from ..state import GraphState

#: 의도별 필수 슬롯. 조류는 정량 임계치가 0건이라 체중·섭취량을 요구하지 않는다 (D-09).
REQUIRED_SLOTS = {
    "intoxication": ("species", "substance"),
    "symptom": ("species",),
    "nutrition": ("species",),
    "general": ("species",),
}

#: 종 키워드. **이름·품종은 여기 넣지 않는다** — 이름에서 종을 추측하면 환각이다.
_SPECIES_KEYWORDS: dict[str, tuple[str, ...]] = {
    "dog":  ("강아지", "개", "멍멍이", "댕댕이"),
    "cat":  ("고양이", "냥이", "야옹이"),
    "bird": ("앵무새", "잉꼬"),
}

#: 물질 키워드. 코퍼스 사실표 · 골든셋에서 추출.
_SUBSTANCE_KEYWORDS: tuple[str, ...] = (
    "초콜릿", "포도", "건포도", "양파", "마늘", "자일리톨",
    "아보카도", "백합", "알로에", "마카다미아", "주목", "홉",
    "고구마", "커피", "카페인",
)

#: 체중 추출 정규식 — "5kg", "5킬로그램", "3.5 kg" 등.
_WEIGHT_RE = re.compile(r"(\d+(?:\.\d+)?)\s*(?:kg|킬로|킬로그램)", re.IGNORECASE)

#: 섭취량 추출 정규식 — "30g", "30 그램" 등. **kg 은 제외**.
_AMOUNT_RE = re.compile(r"(\d+(?:\.\d+)?)\s*(?:g(?![a-z])|그램)", re.IGNORECASE)


def _extract_species(question: str) -> str | None:
    """종 키워드가 명시적으로 있을 때만 반환. **이름·품종에서는 추측하지 않는다.**"""
    for species, keywords in _SPECIES_KEYWORDS.items():
        if any(kw in question for kw in keywords):
            return species
    return None


def _extract_substance(question: str) -> str | None:
    for kw in _SUBSTANCE_KEYWORDS:
        if kw in question:
            return kw
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

    species = _extract_species(question)
    if species is not None:
        new_slots["species"] = species

    substance = _extract_substance(question)
    if substance is not None:
        new_slots["substance"] = substance

    weight = _extract_weight(question)
    if weight is not None:
        new_slots["weight_kg"] = weight

    amount = _extract_amount(question)
    if amount is not None:
        new_slots["amount_g"] = amount

    # 기존 슬롯과 병합 — 새 값이 우선하지만 없는 값은 지우지 않는다.
    merged = {**existing, **new_slots}

    # 결측 슬롯 판정
    required = REQUIRED_SLOTS.get(intent, ("species",))
    missing = [s for s in required if not merged.get(s)]

    return {"slots": merged, "missing_slots": missing}  # type: ignore[typeddict-item]


#: 슬롯별 되묻기 문구. 여러 개 결측 시 앞에서부터 우선순위대로 묻는다.
_CLARIFY_QUESTIONS: dict[str, str] = {
    "species":   "어떤 동물인가요? (개 · 고양이 · 앵무새)",
    "substance": "무엇을 먹었나요?",
    "weight_kg": "체중이 어떻게 되나요? (kg)",
    "amount_g":  "얼마나 먹었나요? (g)",
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