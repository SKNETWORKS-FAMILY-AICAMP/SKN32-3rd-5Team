"""트리아지 하향 금지 게이트.

설계 근거: docs/06_설계결정기록.md · D-09 (2026-07-30 종별 분기 개정)

    LLM 판정이 과소평가(under-triage)를 낼 수 있고, 이 도메인에서
    과소평가는 유일하게 생명과 직결되는 오류다.
    지표로 관리하지 않고 **구조로 막는다.**

    최종 등급 = max(규칙, LLM)  — LLM은 등급을 낮출 수 없다.

이 모듈은 프로젝트에서 가장 안전에 민감한 코드다.
수정 시 tests/test_triage_gate.py 가 반드시 통과해야 한다.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from .levels import TriageLevel


class MonitorWithoutConditions(ValueError):
    """MONITOR인데 상승 조건이 비어 있다.

    D-39: 이 등급의 본질은 "if not resolving"이라는 **조건부 상승**이다.
    조건 없는 "관찰"은 그 자체가 과소평가이므로 출력을 막는다 (04 §4.1.0).
    """


@dataclass(frozen=True)
class TriageDecision:
    """게이트 통과 후의 최종 판정. 근거를 함께 들고 다닌다."""

    level: TriageLevel
    rule_level: TriageLevel | None
    llm_level: TriageLevel | None
    escalation_conditions: tuple[str, ...] = field(default_factory=tuple)
    overridden: bool = False  # LLM이 낮추려 한 것을 게이트가 막았는가

    @property
    def badge(self) -> str:
        return self.level.badge

    @property
    def message(self) -> str:
        return self.level.message


def _coerce(level: TriageLevel | int | None, field: str) -> TriageLevel | None:
    """등급 값을 `TriageLevel` 로 강제한다. 범위 밖이면 즉시 실패한다."""
    if level is None:
        return None
    try:
        return TriageLevel(int(level))
    except (ValueError, TypeError) as e:
        raise ValueError(f"{field}: 알 수 없는 트리아지 등급 {level!r} (1~4)") from e


def apply_gate(
    rule_level: TriageLevel | int | None,
    llm_level: TriageLevel | int | None,
    *,
    escalation_conditions: tuple[str, ...] = (),
) -> TriageDecision:
    """규칙 판정과 LLM 판정을 병합한다. **상향만 수용한다.**

    Args:
        rule_level: 규칙 테이블 1차 판정. 미적용이면 None.
        llm_level:  LLM 판정. 규칙이 적중해 LLM을 부르지 않았으면 None.
        escalation_conditions: MONITOR일 때 함께 출력할 상승 조건.

    Raises:
        ValueError: 양쪽 모두 None (판정 근거가 아예 없음 — 거절 경로로 가야 한다).
        MonitorWithoutConditions: MONITOR인데 조건이 비었을 때.
    """
    if rule_level is None and llm_level is None:
        raise ValueError(
            "규칙·LLM 판정이 모두 없다. 등급을 추측하지 않고 거절 경로로 보낸다 (D-11)."
        )

    # LLM 판정을 JSON에서 int로 파싱해 넘기는 구현이 흔하다.
    # 순수 int가 들어오면 `is` 비교가 전부 거짓이 되어 MONITOR 가드가 뚫린다.
    # 그래서 경계에서 한 번 강제 변환한다 — 범위 밖이면 ValueError로 크게 실패한다.
    rule_level = _coerce(rule_level, "rule_level")
    llm_level = _coerce(llm_level, "llm_level")

    candidates = [lv for lv in (rule_level, llm_level) if lv is not None]
    final = max(candidates)

    overridden = rule_level is not None and llm_level is not None and llm_level < rule_level

    if final == TriageLevel.MONITOR and not escalation_conditions:
        raise MonitorWithoutConditions(
            "MONITOR는 상승 조건 없이 출력할 수 없다 (D-39). "
            "조건 없는 '관찰'은 과소평가로 채점된다."
        )

    return TriageDecision(
        level=final,
        rule_level=rule_level,
        llm_level=llm_level,
        escalation_conditions=escalation_conditions,
        overridden=overridden,
    )
