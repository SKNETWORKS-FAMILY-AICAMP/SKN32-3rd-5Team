"""하향 금지 게이트 — 이 프로젝트에서 가장 안전에 민감한 테스트.

D-09가 "지표가 아니라 구조로 막는다"고 선언했으므로,
그 구조가 실제로 작동함을 테스트가 증명해야 한다.
"""

from __future__ import annotations

import pytest

from pettriage.triage.gate import (
    MonitorWithoutConditions,
    TriageDecision,
    apply_gate,
)
from pettriage.triage.levels import TriageLevel as T


class TestNoDowngrade:
    """LLM은 등급을 낮출 수 없다."""

    @pytest.mark.parametrize(
        ("rule", "llm", "expected"),
        [
            (T.EMERGENCY, T.MONITOR, T.EMERGENCY),  # 최악의 시도 — 4 → 1
            (T.EMERGENCY, T.VISIT_SOON, T.EMERGENCY),
            (T.CALL_NOW, T.MONITOR, T.CALL_NOW),
            (T.VISIT_SOON, T.MONITOR, T.VISIT_SOON),
        ],
    )
    def test_llm_downgrade_is_ignored(self, rule, llm, expected):
        d = apply_gate(rule, llm, escalation_conditions=("증상 지속",))
        assert d.level is expected
        assert d.overridden is True

    @pytest.mark.parametrize(
        ("rule", "llm", "expected"),
        [
            (T.MONITOR, T.EMERGENCY, T.EMERGENCY),  # 상향은 수용
            (T.VISIT_SOON, T.CALL_NOW, T.CALL_NOW),
        ],
    )
    def test_llm_upgrade_is_accepted(self, rule, llm, expected):
        d = apply_gate(rule, llm, escalation_conditions=("증상 지속",))
        assert d.level is expected
        assert d.overridden is False


class TestSingleSided:
    def test_rule_only(self):
        assert apply_gate(T.EMERGENCY, None).level is T.EMERGENCY

    def test_llm_only_when_rule_misses(self):
        assert apply_gate(None, T.CALL_NOW).level is T.CALL_NOW

    def test_both_missing_raises(self):
        """판정 근거가 없으면 추측하지 않고 거절 경로로 (D-11)."""
        with pytest.raises(ValueError):
            apply_gate(None, None)


class TestMonitorNeedsConditions:
    """D-39: 조건 없는 '관찰'은 그 자체가 과소평가다."""

    def test_monitor_without_conditions_raises(self):
        with pytest.raises(MonitorWithoutConditions):
            apply_gate(T.MONITOR, T.MONITOR)

    def test_monitor_with_conditions_ok(self):
        d = apply_gate(T.MONITOR, None, escalation_conditions=("구토가 계속되면",))
        assert d.level is T.MONITOR
        assert d.escalation_conditions == ("구토가 계속되면",)

    def test_higher_levels_do_not_need_conditions(self):
        assert apply_gate(T.EMERGENCY, None).level is T.EMERGENCY


class TestOrdering:
    """정수 방향이 뒤집히면 max()가 min()이 되어 게이트가 무력화된다."""

    def test_numbers_increase_with_risk(self):
        assert T.MONITOR < T.VISIT_SOON < T.CALL_NOW < T.EMERGENCY
        assert int(T.EMERGENCY) == 4 and int(T.MONITOR) == 1

    def test_max_picks_the_dangerous_one(self):
        assert max(T.MONITOR, T.EMERGENCY) is T.EMERGENCY


class TestGoldenSetConflict:
    """코퍼스에서 실제로 나온 상충 사례 (D-39 · 04 §2.3).

    발작 대응 — AAHA S-037은 CALL_NOW, FOUR PAWS S-030은 EMERGENCY.
    게이트가 EMERGENCY를 채택해야 정답이다.
    """

    def test_seizure_conflict_resolves_upward(self):
        d: TriageDecision = apply_gate(rule_level=T.EMERGENCY, llm_level=T.CALL_NOW)
        assert d.level is T.EMERGENCY
        assert d.badge == "응급"
        assert d.message == "지금 바로 동물병원으로 가세요"
