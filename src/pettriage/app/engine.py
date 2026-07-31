"""질의 처리 엔진 — 인터페이스와 스텁 구현.

설계 근거: docs/02_시스템-아키텍처.md §6 · §9 · docs/05 §5

    라우터는 `QAEngine` 프로토콜에만 의존한다. WS2가 LangGraph 그래프를
    완성하면 `GraphEngine` 을 같은 프로토콜로 끼워 넣고 `deps.py` 의
    한 줄만 바꾼다. 프론트·계약·테스트는 손대지 않는다.

`StubEngine` 은 "빈 껍데기"가 아니다. 02 §9 정책 분기와
**하향 금지 게이트를 실제로 통과시킨다** — 검색과 LLM만 고정값이다.
따라서 지금 시연해도 되묻기·거절·게이트 작동이 진짜로 보인다.
"""

from __future__ import annotations

import logging
from typing import Protocol

from ..config import AppConfig, get_config
from ..triage.gate import MonitorWithoutConditions, TriageDecision, apply_gate
from ..triage.levels import TriageLevel
from .contracts import (
    AskRequest,
    AskResponse,
    Citation,
    ClarifyPrompt,
    Refusal,
    TriageResult,
)
from .session import Session

log = logging.getLogger(__name__)


class QAEngine(Protocol):
    """WS2가 구현할 계약. 라우터가 아는 것은 이것뿐이다."""

    name: str

    def ask(self, req: AskRequest, session: Session) -> AskResponse: ...


# ─────────────────────────────────────────────────────────────
# 응답 조립 — 코드가 한다 (05 §4 "응답 조립: 고지·출처·배지·거절 문구")
# ─────────────────────────────────────────────────────────────
def to_triage_result(decision: TriageDecision) -> TriageResult:
    return TriageResult(
        level=int(decision.level),
        name=decision.level.name,
        badge=decision.badge,
        message=decision.message,
        escalation_conditions=list(decision.escalation_conditions),
        # `if x` 가 아니라 `is not None` — 0 등급이 생기면 진리값 분기가 조용히 틀린다
        rule_level=int(decision.rule_level) if decision.rule_level is not None else None,
        llm_level=int(decision.llm_level) if decision.llm_level is not None else None,
        overridden=decision.overridden,
    )


def refuse(session: Session, reason: str, message: str) -> AskResponse:
    return AskResponse(
        status="refused",
        session_id=session.session_id,
        refusal=Refusal(reason=reason, message=message),  # type: ignore[arg-type]
    )


def clarify(session: Session, missing: list[str], question: str, *, max_turns: int) -> AskResponse:
    """되묻기. 상한을 넘기면 거절로 전환한다 (02 §9).

    `max_turns=0` 이면 되묻기를 아예 하지 않고 곧장 거절한다 —
    평가 프로파일이 이 경로를 쓴다. 되묻기가 섞이면 과소평가율 분모가 흔들린다.
    """
    session.clarify_turns += 1
    if session.clarify_turns > max_turns:
        return refuse(
            session,
            "되묻기상한",
            f"필요한 정보({', '.join(missing)})를 확인하지 못해 답변을 드릴 수 없습니다.",
        )
    return AskResponse(
        status="clarify",
        session_id=session.session_id,
        clarify=ClarifyPrompt(
            missing=missing, question=question, turn=session.clarify_turns, max_turns=max_turns
        ),
    )


# ─────────────────────────────────────────────────────────────
# 스텁 지식 — WS1의 규칙 테이블이 들어오기 전까지의 자리
# ─────────────────────────────────────────────────────────────
#: 실제 수치는 src/pettriage/compute/tables/ 의 사실표에서 온다.
#: 여기 값은 **경로가 살아 있는지 보여주기 위한 최소 표본**이며,
#: D-09 규칙 테이블이 확정되면 통째로 교체된다.
#:
#: `citation` 은 인스턴스가 아니라 **생성 인자**로 둔다 —
#: 모듈 전역 인스턴스를 응답에 그대로 실으면 한 요청의 변조가 전역으로 번진다.
_STUB_RULES: dict[str, dict] = {
    "초콜릿": {
        "species": {"dog", "cat"},
        "needs_dose": True,
        "level": TriageLevel.CALL_NOW,
        "escalation": ["구토", "심박 증가", "발작"],
        "citation": {
            "source_id": "S-034",
            "publisher": "Veterinary Sciences",
            "title": "Common toxicologic emergencies in companion animals",
            "locator": "Table 1",
            "route": "사실추출",
        },
    },
    "포도": {
        "species": {"dog"},
        "needs_dose": False,
        "level": TriageLevel.EMERGENCY,
        "escalation": [],
        "citation": {
            "source_id": "S-034",
            "publisher": "Veterinary Sciences",
            "title": "Common toxicologic emergencies in companion animals",
            "locator": "Table 1",
            "route": "사실추출",
        },
    },
    "아보카도": {
        "species": {"bird"},
        "needs_dose": False,
        "level": TriageLevel.EMERGENCY,
        "escalation": [],
        "citation": {
            "source_id": "S-005",
            "publisher": "Lafeber",
            "title": "Foods to avoid feeding pet birds",
            "locator": "Never 등급",
            "route": "사실추출",
        },
    },
}


class StubEngine:
    """고정 지식으로 02 §9 분기를 전부 태우는 엔진.

    벡터DB·LLM 없이도 다음이 실제로 동작한다.

      · 종 미확인 → 강제 되묻기
      · 슬롯 결측 → 되묻기 (상한 초과 시 거절). **진전이 있으면 카운터를 되돌린다**
      · 매칭 없음 → 거절 `근거없음`
      · 매칭 → 규칙 판정 → **하향 금지 게이트** → 답변 + 근거

    문장 생성은 템플릿이 한다. 여기서도 LLM을 부르지 않는다 (D-38).
    """

    name = "stub"

    def __init__(self, config: AppConfig | None = None) -> None:
        self._config = config

    @property
    def cfg(self) -> AppConfig:
        return self._config or get_config()

    def ask(self, req: AskRequest, session: Session) -> AskResponse:
        triage_cfg = self.cfg.triage
        max_turns = triage_cfg.max_clarify_turns

        # 되묻기에 진전이 있었으면 예산을 되돌린다.
        # 그러지 않으면 슬롯을 하나씩 채우는 협조적인 사용자가 상한에 걸려 거절된다.
        if session.merge(req):
            session.clarify_turns = 0

        # ① 종 미확인 — 되묻기 강제 (02 §9 · D-10)
        if session.species is None:
            return clarify(
                session,
                ["species"],
                "어떤 동물인가요? (개 / 고양이 / 앵무새) — 종에 따라 판단이 완전히 달라집니다.",
                max_turns=max_turns,
            )

        # ② 검색 (스텁: 키워드 매칭)
        hit_key = next((k for k in _STUB_RULES if k in req.question), None)
        if hit_key is None:
            return refuse(session, "근거없음", "제공된 자료에서 근거를 찾을 수 없습니다.")

        rule = _STUB_RULES[hit_key]
        if session.species not in rule["species"]:
            return refuse(
                session,
                "근거없음",
                f"보유 자료에 {hit_key}과(와) 해당 종에 대한 근거가 없습니다.",
            )

        # ③ 슬롯 결측 — 체중당 섭취량이 필요한 규칙만.
        #    조류는 정량 임계치가 0건이라 수치를 요구하지 않는다 (D-09 개정).
        #    요구하면 근거에 없는 값을 LLM이 지어낸다.
        quantitative = session.species in triage_cfg.quantitative_species
        if rule["needs_dose"] and quantitative:
            missing = [
                f
                for f, v in (("weight_kg", session.weight_kg), ("amount_g", session.amount_g))
                if v is None
            ]
            if missing:
                return clarify(
                    session,
                    missing,
                    "체중(kg)과 먹은 양(g)을 알려주세요 — 체중당 섭취량으로 판단합니다.",
                    max_turns=max_turns,
                )

        # ④ 규칙 판정 → 하향 금지 게이트 (05 §5)
        #    스텁이므로 llm_level 은 None. 실제 엔진은 규칙 미적중 시 LLM을 부른다.
        try:
            decision = apply_gate(
                rule_level=rule["level"],
                llm_level=None,
                escalation_conditions=tuple(rule["escalation"]),
            )
        except MonitorWithoutConditions:
            # 조건 없는 '관찰'은 과소평가다 (D-39 · 04 §4.1.0).
            # 추측해서 내보내지 않고 안전한 쪽으로 실패한다.
            log.warning("MONITOR without escalation conditions — 거절로 전환 (%s)", hit_key)
            return refuse(session, "판정불가", "상태를 판단할 근거가 부족합니다.")
        except ValueError:
            return refuse(session, "판정불가", "상태를 판단할 근거가 부족합니다.")

        session.clarify_turns = 0
        return AskResponse(
            status="answered",
            session_id=session.session_id,
            answer=self._compose(hit_key, session.species, decision),
            triage=to_triage_result(decision),
            citations=[Citation(**rule["citation"])],
        )

    @staticmethod
    def _compose(substance: str, species: str, decision: TriageDecision) -> str:
        """응답 문장 조립 — 코드가 한다 (05 §4).

        의학적 중증도 어휘를 쓰지 않는다. 행동만 지시한다 (D-11 · D-39).

        상승 조건은 여기 넣지 않는다 — `escalation_conditions` 필드로 나간다.
        화면이 없는 클라이언트는 `full_text` 를 쓰면 조건까지 붙은 문장을 받는다.
        """
        ko = {"dog": "개", "cat": "고양이", "bird": "앵무새"}[species]
        return f"{ko}가 {substance}을(를) 섭취한 상황입니다. {decision.message}."
