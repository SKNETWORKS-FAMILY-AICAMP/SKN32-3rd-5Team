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

from ..triage.gate import MonitorWithoutConditions, TriageDecision, apply_gate
from ..triage.levels import TriageLevel
from .contracts import (
    MAX_CLARIFY_TURNS,
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
        rule_level=int(decision.rule_level) if decision.rule_level else None,
        llm_level=int(decision.llm_level) if decision.llm_level else None,
        overridden=decision.overridden,
    )


def refuse(session: Session, reason: str, message: str) -> AskResponse:
    return AskResponse(
        status="refused",
        session_id=session.session_id,
        refusal=Refusal(reason=reason, message=message),  # type: ignore[arg-type]
    )


def clarify(session: Session, missing: list[str], question: str) -> AskResponse:
    """되묻기. 상한을 넘기면 거절로 전환한다 (02 §9)."""
    session.clarify_turns += 1
    if session.clarify_turns > MAX_CLARIFY_TURNS:
        return refuse(
            session,
            "되묻기상한",
            f"필요한 정보({', '.join(missing)})를 확인하지 못해 답변을 드릴 수 없습니다.",
        )
    return AskResponse(
        status="clarify",
        session_id=session.session_id,
        clarify=ClarifyPrompt(missing=missing, question=question, turn=session.clarify_turns),
    )


# ─────────────────────────────────────────────────────────────
# 스텁 지식 — WS1의 규칙 테이블이 들어오기 전까지의 자리
# ─────────────────────────────────────────────────────────────
#: 실제 수치는 src/pettriage/compute/tables/ 의 사실표에서 온다.
#: 여기 값은 **경로가 살아 있는지 보여주기 위한 최소 표본**이며,
#: D-09 규칙 테이블이 확정되면 통째로 교체된다.
_STUB_RULES: dict[str, dict] = {
    "초콜릿": {
        "species": {"dog", "cat"},
        "needs_weight": True,
        "level": TriageLevel.CALL_NOW,
        "escalation": ["구토", "심박 증가", "발작"],
        "citation": Citation(
            source_id="S-034",
            publisher="Veterinary Sciences",
            title="Common toxicologic emergencies in companion animals",
            locator="Table 1",
            route="사실추출",
        ),
    },
    "포도": {
        "species": {"dog"},
        "needs_weight": False,
        "level": TriageLevel.EMERGENCY,
        "escalation": [],
        "citation": Citation(
            source_id="S-034",
            publisher="Veterinary Sciences",
            title="Common toxicologic emergencies in companion animals",
            locator="Table 1",
            route="사실추출",
        ),
    },
    "아보카도": {
        "species": {"bird"},
        "needs_weight": False,
        "level": TriageLevel.EMERGENCY,
        "escalation": [],
        "citation": Citation(
            source_id="S-005",
            publisher="Lafeber",
            title="Foods to avoid feeding pet birds",
            locator="Never 등급",
            route="사실추출",
        ),
    },
}


class StubEngine:
    """고정 지식으로 02 §9 분기를 전부 태우는 엔진.

    벡터DB·LLM 없이도 다음이 실제로 동작한다.

      · 종 미확인 → 강제 되묻기
      · 슬롯 결측 → 되묻기 (상한 2회 초과 시 거절)
      · 매칭 없음 → 거절 `근거없음`
      · 매칭 → 규칙 판정 → **하향 금지 게이트** → 답변 + 근거

    문장 생성은 템플릿이 한다. 여기서도 LLM을 부르지 않는다 (D-38).
    """

    name = "stub"

    def ask(self, req: AskRequest, session: Session) -> AskResponse:
        session.merge(req)

        # ① 종 미확인 — 되묻기 강제 (02 §9)
        if session.species is None:
            return clarify(
                session,
                ["species"],
                "어떤 동물인가요? (개 / 고양이 / 앵무새) — 종에 따라 판단이 완전히 달라집니다.",
            )

        # ② 검색 (스텁: 키워드 매칭)
        hit_key = next((k for k in _STUB_RULES if k in req.question), None)
        if hit_key is None:
            return refuse(
                session,
                "근거없음",
                "제공된 자료에서 근거를 찾을 수 없습니다.",
            )

        rule = _STUB_RULES[hit_key]
        if session.species not in rule["species"]:
            return refuse(
                session,
                "근거없음",
                f"보유 자료에 {hit_key}과(와) 해당 종에 대한 근거가 없습니다.",
            )

        # ③ 슬롯 결측 — 체중당 섭취량이 필요한 규칙만
        missing = [
            f
            for f, v in (("weight_kg", session.weight_kg), ("amount_g", session.amount_g))
            if rule["needs_weight"] and v is None
        ]
        if missing:
            return clarify(
                session,
                missing,
                "체중(kg)과 먹은 양(g)을 알려주세요 — 체중당 섭취량으로 판단합니다.",
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
            citations=[rule["citation"]],
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
