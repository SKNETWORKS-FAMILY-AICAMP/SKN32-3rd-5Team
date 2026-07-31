"""트리아지 등급 — 코퍼스의 행동 지시어에서 도출한 4단계.

설계 근거: docs/06_설계결정기록.md · D-39

    등급 이름을 임의로 정하지 않았다. 살아있는 코퍼스 42건에서
    "보호자에게 무엇을 시키는가"(action language)를 모아 4단계로 갈랐다.
    각 등급의 verbatim 근거는 D-39 표를 참조.

    ⚠️ 숫자가 클수록 위험하다.
       D-09의 하향 금지 게이트가 max()로 성립하려면 이 방향이어야 한다.
       (초안의 L1=최고위험 표기는 min()을 요구해 폐기되었다)
"""

from __future__ import annotations

from enum import IntEnum


class TriageLevel(IntEnum):
    """사고 대응 긴급도. int 상속이므로 max()·비교가 그대로 동작한다."""

    MONITOR = 1
    VISIT_SOON = 2
    CALL_NOW = 3
    EMERGENCY = 4

    @property
    def badge(self) -> str:
        """UI 배지 (짧게)."""
        return _BADGE[self]

    @property
    def message(self) -> str:
        """사용자에게 보이는 행동 문장.

        의학적 중증도 어휘(`위험`·`중증`)를 쓰지 않는다 — D-11 진단 금지.
        코퍼스의 소스들도 등급 이름 없이 행동만 지시하고 있었다.
        """
        return _MESSAGE[self]


_BADGE: dict[TriageLevel, str] = {
    TriageLevel.EMERGENCY: "응급",
    TriageLevel.CALL_NOW: "전화",
    TriageLevel.VISIT_SOON: "내원",
    TriageLevel.MONITOR: "관찰",
}

_MESSAGE: dict[TriageLevel, str] = {
    TriageLevel.EMERGENCY: "지금 바로 동물병원으로 가세요",
    TriageLevel.CALL_NOW: "지금 수의사에게 전화해 상태를 알리세요",
    TriageLevel.VISIT_SOON: "오늘 중 진료를 받으세요",
    TriageLevel.MONITOR: "집에서 지켜보고, 아래 증상이 나타나면 연락하세요",
}

# 각 등급을 도출한 코퍼스 근거 (D-39). 발표·검증 시 추적용.
EVIDENCE: dict[TriageLevel, tuple[str, str]] = {
    TriageLevel.EMERGENCY: (
        "S-038",
        "Don't call. Take your pet straight to the nearest ER.",
    ),
    TriageLevel.CALL_NOW: (
        "S-086",
        "call your veterinarian to discuss whether your pet needs to be seen",
    ),
    TriageLevel.VISIT_SOON: (
        "S-080",
        "Contact your veterinarian if repetitive behavior occurs daily or worsens",
    ),
    TriageLevel.MONITOR: (
        "S-092",
        "Monitor condition. If condition is not resolving seek veterinary care.",
    ),
}


class FeedingLevel(IntEnum):
    """축 B — 급여 가부. 긴급도(축 A)와 **섞지 않는다**.

    "포도는 개에게 금지"(축 B)와 "포도를 먹었으니 EMERGENCY"(축 A)는 다른 판정이다.
    D-38 템플릿의 ``{risk_level}`` 은 이 축이다.

    조류는 ``SAFE`` 를 쓰지 않는다 — S-005와 S-091이 동일 항목(Peanuts·Grit·
    Mushrooms)을 서로 다른 티어에 배정해, 3단계를 노출하면 출처에 없는
    정밀도를 주장하게 된다 (D-39).
    """

    SAFE = 1
    CAUTION = 2
    NEVER = 3

    @property
    def label(self) -> str:
        return {
            FeedingLevel.NEVER: "급여 금지",
            FeedingLevel.CAUTION: "조건부",
            FeedingLevel.SAFE: "안전",
        }[self]


BIRD_FEEDING_LEVELS: frozenset[FeedingLevel] = frozenset({FeedingLevel.NEVER, FeedingLevel.CAUTION})
