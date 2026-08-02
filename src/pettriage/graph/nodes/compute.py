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

from ...compute.content import content_for
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

    computed: dict = {"dose_per_kg": dose_per_kg, "unit": "g/kg"}
    result: dict = {"computed": computed}

    # 규칙 테이블 조회 → 바닥 등급 (D-50).
    # substance 가 있어야 조회 가능. 없으면 정성 답변으로 내려간다.
    if not substance:
        return result  # type: ignore[typeddict-item]

    # ── **먹은 무게를 유효성분 무게로 바꾼다** (D-78) ────────────
    #
    # 역치는 유효성분 기준이다 — `F-034-020` 은 *테오브로민·카페인* 20 mg/kg 이라고
    # 말한다. 초콜릿 무게를 그대로 대면 **200배 크게 잰다** (밀크 초콜릿 2 mg/g).
    # 그 오차의 방향은 과대이고, 골든셋이 `MONITOR` 라고 적어 둔 질의에
    # *"지금 병원"* 을 내게 된다.
    #
    # 계수가 **없는 물질은 예전 그대로** 돈다 — 마늘 5 g/kg 처럼 역치 자체가
    # 물질 무게 기준인 경우가 그렇다. 있는 것만 바꾼다.
    content = content_for(substance, species)
    lookup_name = substance

    if content is not None:
        lookup_name = content.active_substance
        computed["active_substance"] = content.active_substance
        computed["content_mg_per_g"] = content.mg_per_g
        computed["content_basis"] = "|".join(content.basis)

        if not content.quantifiable:
            # 🔴 **0 으로 채우지 않는다** (D-16). 원문이 수치를 주지 않았다.
            #    정량 판정을 건너뛰면 `apply_rule_table` 이 역치 미만(MONITOR)으로
            #    바닥을 깔고 상승 조건을 함께 낸다 — 말할 수 있는 것만 말한 답이다.
            log.info(
                "함량이 수치로 없다 (%s · %s) — 정량 판정을 건너뛴다 (%s)",
                substance,
                content.bound,
                "|".join(content.basis),
            )
            return result  # type: ignore[typeddict-item]

        active_mg = content.active_mg(float(amount_g))
        assert active_mg is not None  # quantifiable 이 보증한다
        amount_mg_per_kg: float | None = active_mg / float(weight_kg)
        computed["active_mg"] = active_mg
        computed["active_mg_per_kg"] = amount_mg_per_kg
        computed["content_is_lower_bound"] = content.is_lower_bound
    else:
        amount_mg_per_kg = to_mg_per_kg(dose_per_kg, "g/kg")

    if amount_mg_per_kg is None:
        return result  # type: ignore[typeddict-item]

    verdict = rule_level_for(lookup_name, species, amount_mg_per_kg)
    if verdict.level is not None:
        result["rule_level"] = int(verdict.level)
        # **잰 자리라고 표시한다** (D-80). 이 등급은 출처 달린 역치와 계산된 용량에서
        # 나왔으므로 LLM 이 그 위로 올릴 수 없다. 정성 표·양 미상 바닥은 세우지 않는다 —
        # 거기서는 LLM 이 맞는 경우가 있다 (G-011·G-017).
        result["rule_is_quantitative"] = True
        if verdict.escalation_conditions:
            result["escalation_conditions"] = list(verdict.escalation_conditions)
    elif verdict.reason:
        log.info("compute: rule_level None — %s", verdict.reason)

    return result  # type: ignore[typeddict-item]


def numeric_evidence(state: GraphState) -> str:
    """⑨ LLM 트리아지 판정에 **코드가 계산한 값과 자료의 역치를 함께 준다** (D-79).

    ⚠️ **규칙의 결론(등급)은 주지 않는다.** 주면 LLM 이 그대로 따라 읽고,
    `llm_level` 이 독립적이지 않게 된다 — `overridden` 이 의미를 잃고 산출물 ④에서
    *"하향 금지 게이트가 실제로 작동했다"* 를 보일 증거가 사라진다 (D-09).
    주는 것은 **수치와 역치**뿐이다. 판단은 LLM 이 그대로 한다.

    왜 필요한가 — 2026-08-02 프로브 실측:

        G-028  밀크초콜릿 20 g / 5 kg  →  코드: 테오브로민 8 mg/kg (역치 20 미만)
                                          LLM: EMERGENCY        →  최종 4
        골든셋 기대: MONITOR

        판정 노드에 넘어가던 것은 **검색 문맥뿐**이었다. 8 mg/kg 인지 4,000 mg/kg 인지
        모른 채 `개 + 초콜릿` 만 보고 답했고, 그 상황에서는 그것이 합리적인 답이다.
        게이트가 `max(rule, llm)` 이므로 **코드가 맞게 계산한 등급이 매번 덮였다** —
        프로브 8건에서 과대평가 50%(4건)가 전부 이 경로였다.

    **눈을 가리고 시킨 판단을 판단이라고 부르지 않는다.**

    Returns:
        프롬프트에 덧붙일 문자열. 줄 것이 없으면 빈 문자열.
    """
    from ...compute.rules import lookup

    slots = state.get("slots") or {}
    computed = state.get("computed") or {}
    substance = slots.get("substance")
    species = slots.get("species")
    if not substance or not species:
        return ""

    from ...compute.content import content_for, threshold_substance
    from ...compute.rules import parse_low, to_mg_per_kg

    lines: list[str] = []
    unknown: list[str] = []
    content = content_for(substance, species)

    # ── 코드가 계산한 값 ──
    #
    # 🔴 **단위를 맞춰서 낸다.** 2026-08-02 실측 — 화이트초콜릿(G-041)에서
    #    `화이트 초콜릿 20 g/kg` 과 `임상징후 발현 20 mg/kg` 을 나란히 냈다.
    #    앞은 초콜릿 무게, 뒤는 테오브로민 기준이라 **1000배 넘긴 것처럼 보인다.**
    #    그 정보만 보면 EMERGENCY 가 합리적인 답이고, 실제로 그렇게 나왔다.
    #    **오도한 것은 모델이 아니라 우리다.**
    if computed.get("active_mg_per_kg") is not None:
        lines.append("[코드가 계산한 값]")
        lines.append(
            f"  {computed.get('active_substance', '유효성분')} "
            f"{computed['active_mg_per_kg']:.4g} mg/kg"
            f"  ({substance} {computed.get('content_mg_per_g')} mg/g × "
            f"{slots.get('amount_g')} g ÷ {slots.get('weight_kg')} kg)"
        )
        if computed.get("content_is_lower_bound"):
            lines.append("  ⚠ 함량이 원문에서 하한(이상)이라 실제 값은 더 높을 수 있다")
    elif content is not None and not content.quantifiable:
        # 함량을 수치로 모른다 → **아무 수치도 내지 않는다.** 0 도, 물질 무게도 아니다.
        #
        # ⚠️ 이것을 `[확인 안 된 것]` 에 넣지 않는다. **모르는 것이 아니라 자료가 말한
        #    것**이다 — 원문은 *"테오브로민 급원으로서는 무의미한 수준"* 이라고 적는다.
        #    없음(정보 부재)과 무의미함(확인된 사실)을 같은 칸에 두면, 뒤에 붙는
        #    *"모르는 것을 안전으로 읽지 않는다"* 가 **자료가 말한 것까지 밀어 올린다.**
        lines.append("[자료가 말하는 것]")
        lines.append(
            f"  {substance} — {content.active_substance} 함량이 **무의미한 수준**이다 "
            f"({'·'.join(content.basis)})"
        )
        lines.append("  수치가 없어 정량 판정은 하지 않는다. 자료가 확인한 사실이다.")
    elif computed.get("dose_per_kg") is not None:
        mg = to_mg_per_kg(computed["dose_per_kg"], "g/kg")
        lines.append("[코드가 계산한 값]")
        lines.append(
            f"  {substance} {mg:.4g} mg/kg"
            f"  ({slots.get('amount_g')} g ÷ {slots.get('weight_kg')} kg)"
        )
    elif computed.get("daily_energy_kcal") is not None:
        lines.append("[코드가 계산한 값]")
        lines.append(f"  일일 권장 열량 {computed['daily_energy_kcal']:.4g} kcal/day (BER)")

    # ── 자료의 역치 ── **여기 숫자를 손으로 적지 않는다** (D-22). 표에서 온다.
    #    원문 표기는 그대로 두되(`40-50 mg/kg`), 위 계산값과 **같은 단위로 환산한 값**을
    #    함께 적는다 — 읽는 쪽이 단위를 맞추게 두지 않는다.
    rows = lookup(threshold_substance(substance, species) or substance, species)
    if rows:
        lines.append(f"[자료의 역치] ({species})")
        for r in rows:
            low = parse_low(r.dose)
            norm = to_mg_per_kg(low, r.unit) if low is not None else None
            extra = f" = {norm:.4g} mg/kg" if norm is not None and r.unit.strip() != "mg/kg" else ""
            lines.append(f"  {r.threshold_type} {r.dose} {r.unit}{extra}  ({r.fact_id})")

    # ── 확인 안 된 것 ── **모른다는 사실도 근거다** (D-13).
    unknown += [n for k, n in (("weight_kg", "체중"), ("amount_g", "섭취량")) if not slots.get(k)]
    if rows and unknown:
        lines.append("[확인 안 된 것]")
        for u in unknown:
            lines.append(f"  {u}")
        lines.append("  → **정량 판정을 할 수 없다.** 적게 먹었다는 뜻이 아니다.")
        lines.append("  모르는 것을 안전으로 읽지 않는다.")

    return "\n".join(lines)
