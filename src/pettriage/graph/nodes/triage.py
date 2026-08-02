"""트리아지 판정 노드.

설계 근거: 02 §7.2 · 05 §5 · D-09 · D-39

    🔒 **`max()` 를 직접 쓰지 않는다.** 반드시 `triage.gate.apply_gate` 를 부른다 —
    MONITOR 상승 조건 검사가 거기 붙어 있다.

    LLM 은 등급을 낮출 수 없다 — apply_gate 가 `max(rule, llm)` 로 강제한다.
"""

from __future__ import annotations

import logging

from ...triage.gate import MonitorWithoutConditions, apply_gate
from ..state import GraphState

log = logging.getLogger(__name__)


def apply_rule_table(state: GraphState) -> GraphState:
    """규칙 표에서 (물질 × 종) **바닥 등급**을 얻는다 (D-50).

    ⚠️ **여기에 표를 두지 않는다.** 2026-08-02 흡수 전에는 `engine.py` 안에
    `_RULE_TABLE` 12줄이 손으로 적혀 있었다 —

        ("초콜릿", "dog"): (CALL_NOW, [...])
        ("알로에", "cat"): (VISIT_SOON, [...])

    그 12줄이 `rule_level` 을 만들고, `rule_level` 은 `apply_gate` 의 **바닥**이다.
    바닥이 손으로 적은 값이면 하향 금지 게이트가 **근거 없는 등급을 지켜준다.**
    진짜 표는 사실 표 888행에서 `make rules` 로 파생된 CSV 두 장이다 (D-16 · D-22 · D-38).

    ⚠️ **이 함수가 엔진 안에 있었다** (`GraphEngine._apply_rule_table`).
        판정 로직이 배달 계층에 섞여 있었다는 뜻이고, 그래서 노드 테스트가 닿지 않았다.
        노드는 노드 폴더에 둔다 — 그래프가 부를 수 있는 것은 노드뿐이다 (D-40).

    정량 계산이 가능하면 `compute_metrics` 가 이미 `rule_level` 을 세운다.
    여기서는 **양을 모를 때의 정성 바닥**만 채운다 — 표에 (물질 × 종) 행이
    있다는 것 자체가 *"이 종에 이 물질은 위험하다"* 는 근거다.

    Returns:
        `{"rule_level": ..., "escalation_conditions": [...]}` 또는 `{}`.
        **빈 dict 은 실패가 아니다** — 근거가 없다는 뜻이고 `decide_triage` 가 판정불가로 보낸다.
    """
    if state.get("rule_level") is not None:
        return {}  # compute_metrics 가 정량으로 이미 정했다 — 덮지 않는다

    slots = state.get("slots") or {}
    substance = slots.get("substance")
    species = slots.get("species")
    if not substance or not species:
        return {}

    from ...compute.rules import qualitative_level_for

    out: GraphState = {}

    # ① 코퍼스가 등급을 말한 경우 — **정성 등급 표**가 바닥이다 (D-50).
    #    조류는 정량 역치가 0행이라 **이 경로가 유일하다** (D-09 개정).
    v = qualitative_level_for(substance, species)
    if v.level is not None:
        out["rule_level"] = int(v.level)
        if v.conditions and not state.get("escalation_conditions"):
            out["escalation_conditions"] = list(v.conditions)
        log.info("rule_level(정성) %s — %s", int(v.level), v.reason)
        return out

    # ② 등급은 없지만 **역치 행이 있는** 경우 — 위험하다는 사실 자체는 근거가 있다.
    #    양을 모르므로 역치 미만과 같이 다룬다 (`BELOW_THRESHOLD_LEVEL`).
    #    LLM 이 올릴 수 있게 바닥만 두는 것이 D-09 의 설계다.
    from ...compute.rules import BELOW_THRESHOLD_LEVEL, _signs_of, lookup

    rows = lookup(substance, species)
    if not rows:
        return {}  # 근거 없음 — decide_triage 가 판정불가로 보낸다
    out["rule_level"] = int(BELOW_THRESHOLD_LEVEL)
    if not state.get("escalation_conditions"):
        conditions = _signs_of(rows)
        if conditions:
            out["escalation_conditions"] = list(conditions)
    return out


def decide_triage(state: GraphState) -> GraphState:
    """규칙 1차 → 미적용 시 LLM → 하향 금지 게이트.

    규칙 테이블 입력이 **종별로 다르다** (D-09 개정).
    개·고양이는 `물질 × 체중당 섭취량` 정량, 조류는 `물질 × 증상` 정성이다.
    조류에 수치를 요구하면 근거에 없는 값을 모델이 지어낸다.

    Returns:
        `{"rule_level": ..., "llm_level": ..., "triage_level": ...,
          "escalation_conditions": [...]}`
        MONITOR인데 상승 조건이 없으면 `refused / 판정불가`.
    """
    rule_level = state.get("rule_level")
    llm_level = state.get("llm_level")
    escalation = state.get("escalation_conditions") or []

    try:
        decision = apply_gate(
            rule_level=rule_level,
            llm_level=llm_level,
            escalation_conditions=tuple(escalation),
        )
    except MonitorWithoutConditions:
        # 조건 없는 '관찰'은 과소평가로 채점된다 (D-39 · 04 §4.1.0).
        log.warning("MONITOR without escalation conditions — 거절로 전환")
        return {  # type: ignore[typeddict-item]
            "status": "refused",
            "refusal_reason": "판정불가",
        }
    except ValueError:
        # rule·llm 둘 다 None (판정 근거 없음).
        return {  # type: ignore[typeddict-item]
            "status": "refused",
            "refusal_reason": "판정불가",
        }

    return {  # type: ignore[typeddict-item]
        "triage_level": int(decision.level),
        "rule_level": int(decision.rule_level) if decision.rule_level is not None else None,
        "llm_level": int(decision.llm_level) if decision.llm_level is not None else None,
        "escalation_conditions": list(decision.escalation_conditions),
        "overridden": decision.overridden,
    }
