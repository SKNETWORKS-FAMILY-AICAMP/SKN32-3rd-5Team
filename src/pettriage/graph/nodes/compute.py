"""계산 노드 (비-RAG).

설계 근거: 02 §7 · D-09 · D-16 · D-46 · D-50 · 05 §4

    **수치는 벡터 검색으로 찾지 않는다** (D-16). 관계형 테이블을 조회하고
    계산은 코드가 한다. LLM 은 여기 관여하지 않는다.

    종별 계산 로직이 다르다:
      · 개·고양이 → dose_per_kg + **바닥 등급** (rule_level, D-50)
      · 앵무새 → BER (Basal Energy Requirement, 일일 권장 열량)

    앵무새는 체중당 독성 임계치가 코퍼스에 0건이라 (D-09 개정) dose_per_kg 를 만들지 않는다 —
    대신 BER 공식(K × Wkg^0.75)으로 일일 권장 열량을 계산한다.
    K 값: Psittacine(앵무새) = 175.
"""

from __future__ import annotations

import logging

from ...compute.rules import rule_level_for, to_mg_per_kg
from ...config import get_config
from ..state import GraphState

log = logging.getLogger(__name__)

#: BER 공식 K 값 (Basal Energy Requirement).
#: Psittacine(앵무새) 계열.
_BIRD_K_PSITTACINE = 175


def _ber_kcal(weight_kg: float, k: int = _BIRD_K_PSITTACINE) -> float:
    """BER = K × (Wkg)^0.75 kcal/day.

    앵무새(Psittacine) 계열에 적용 (K=175).
    """
    return k * (weight_kg**0.75)


def compute_metrics(state: GraphState) -> GraphState:
    """종별로 다른 정량 계산 + 바닥 등급 판정.

    - 앵무새: 일일 권장 열량 (BER = K × Wkg^0.75, K=175)
    - 개·고양이: 체중당 섭취량 + 규칙 테이블 바닥 등급 (D-50)

    Returns:
        `{"computed": {...}, "rule_level": ..., "escalation_conditions": [...]}`.
        필요한 슬롯이 없으면 **빈 dict** — 지어낸 수치를 넣지 않는다.
    """
    slots = state.get("slots") or {}
    species = slots.get("species")
    weight_kg = slots.get("weight_kg")
    amount_g = slots.get("amount_g")
    substance = slots.get("substance")

    # 앵무새 — BER 열량 계산 (독성 정량 판정은 하지 않음, D-09).
    if species == "bird":
        if weight_kg is None or weight_kg <= 0:
            return {"computed": {}}  # type: ignore[typeddict-item]
        return {  # type: ignore[typeddict-item]
            "computed": {
                "daily_energy_kcal": _ber_kcal(float(weight_kg)),
                "formula": "BER",
                "k_value": _BIRD_K_PSITTACINE,
                "unit": "kcal/day",
            }
        }

    # 개·고양이 — 체중당 섭취량 (독성 임계치 판정용).
    try:
        quantitative = get_config().triage.quantitative_species
    except Exception:
        quantitative = ["dog", "cat"]

    if species not in quantitative:
        return {"computed": {}}  # type: ignore[typeddict-item]

    if weight_kg is None or amount_g is None or weight_kg <= 0:
        return {"computed": {}}  # type: ignore[typeddict-item]

    dose_per_kg = float(amount_g) / float(weight_kg)  # g/kg

    result: dict = {
        "computed": {
            "dose_per_kg": dose_per_kg,
            "unit": "g/kg",
        }
    }

    # 규칙 테이블 조회 → 바닥 등급 (D-50).
    # substance 가 있어야 조회 가능. 없으면 정성 답변으로 내려간다.
    if substance:
        # g/kg → mg/kg 변환
        amount_mg_per_kg = to_mg_per_kg(dose_per_kg, "g/kg")
        if amount_mg_per_kg is not None:
            verdict = rule_level_for(substance, species, amount_mg_per_kg)
            if verdict.level is not None:
                result["rule_level"] = int(verdict.level)
                if verdict.escalation_conditions:
                    result["escalation_conditions"] = list(verdict.escalation_conditions)
            elif verdict.reason:
                log.info("compute: rule_level None — %s", verdict.reason)

    return result  # type: ignore[typeddict-item]
