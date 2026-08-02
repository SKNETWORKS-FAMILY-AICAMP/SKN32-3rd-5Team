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
    overridden: bool = False  # LLM이 **낮추려** 한 것을 게이트가 막았는가 (D-09)
    #: LLM 이 **올리려** 한 것을 게이트가 막았는가 (D-80).
    #: 규칙 등급이 **정량 계산**에서 나왔을 때만 참이 될 수 있다.
    llm_capped: bool = False

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
    rule_is_quantitative: bool = False,
) -> TriageDecision:
    """규칙 판정과 LLM 판정을 병합한다. **하향은 언제나 막고, 상향은 조건부로 받는다.**

    ## D-80 — 잰 자리에서는 규칙이, 못 잰 자리에서는 LLM 이 맞다

    2026-08-03 골든셋 60건 실측 (`--arm A` · gpt-4o-mini) —

        LLM 이 올렸다  10건   그중 **과대로 끝난 것 7건** (70%)
        🔒 하향 차단     5건

        틀린 상승 7건 중 6건이 `rule=3 llm=4` — CALL_NOW 를 EMERGENCY 로 한 칸씩
        올렸다. 유형별로 `dose` 가 가장 나빴다 (과대 42.9%).

        반대로 옳았던 상승도 있다 — 기준선에서 과소였던 G-011(백합·cat)·G-017
        (아보카도·bird)이 LLM 상승으로 고쳐졌다. **둘 다 양이 없어 못 잰 자리**다.

    경계가 선명하다 —

        못 잰 자리(정성 표 · 양 미상 바닥)  → LLM 이 맞다. 그대로 올린다
        잰 자리(섭취량 ÷ 체중 → 역치 비교)  → **규칙이 맞다. 올리지 않는다**

    출처 달린 역치와 계산된 용량이 있는데 그 위로 올리는 것은 **근거 없는 상승**이고,
    근거 없는 주장은 방향이 위든 아래든 자료 밖으로 나가는 것이다 (D-10 · D-16).
    D-09 를 뒤집는 것이 아니라 **적용 범위를 정하는 개정**이다 — 하향 금지는 그대로다.

    ⚠️ **막았다는 사실은 지우지 않는다.** `llm_level` 은 그대로 싣고 `llm_capped` 로
    표시한다. 조용히 무시하면 *"LLM 이 규칙과 늘 같다"* 로 보이고, 그것은 거짓이다.

    Args:
        rule_level: 규칙 테이블 1차 판정. 미적용이면 None.
        llm_level:  LLM 판정. 규칙이 적중해 LLM을 부르지 않았으면 None.
        escalation_conditions: MONITOR일 때 함께 출력할 상승 조건.
        rule_is_quantitative: `rule_level` 이 **실제 용량 계산**에서 나왔는가.
            참이면 LLM 은 그 위로 올릴 수 없다. `compute_metrics` 만 이것을 세운다 —
            정성 표·양 미상 바닥은 세우지 않는다.

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

    # **잰 자리에서는 올리지 않는다** (D-80). 하향 금지는 위에서 이미 `max` 가 지켰다.
    llm_capped = (
        rule_is_quantitative
        and rule_level is not None
        and llm_level is not None
        and llm_level > rule_level
    )
    if llm_capped:
        final = rule_level

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
        llm_capped=llm_capped,
    )
