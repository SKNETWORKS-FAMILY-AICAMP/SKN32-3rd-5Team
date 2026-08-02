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
    }