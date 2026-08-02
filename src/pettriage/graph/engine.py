"""`GraphEngine` — 그래프를 배달 계층에 물린다.

설계 근거: docs/06 D-40 · docs/02 §6·§12.1

    `deps.get_engine()` 이 `configs/*.yaml` 의 `serve.engine` 을 보고 고른다.
    `graph` 로 두면 이 클래스가 물리고, 계약·프론트·테스트는 그대로다.

    노드 7개를 순서대로 실행하고, 세 상태(answered/clarify/refused)를
    `AskResponse` 로 조립한다.
"""

from __future__ import annotations

import logging

from ..app.contracts import (
    AskRequest,
    AskResponse,
    Citation,
    ClarifyPrompt,
    Refusal,
    TriageResult,
)
from ..app.session import Session
from ..triage.levels import TriageLevel
from .nodes import (
    MAX_RETRY,
    ask_clarify,
    build_filter,
    classify_intent,
    compress_context,
    compute_metrics,
    decide_triage,
    extract_slots,
    generate_draft,
    retrieve,
    simplify,
    verify_grounding,
)
from .state import GraphState, initial_state

log = logging.getLogger(__name__)


class EngineNotReady(RuntimeError):
    """그래프 노드가 아직 구현되지 않았다.

    `pytest -m todo` 로 남은 일을 확인한다.
    """


# ─────────────────────────────────────────────────────────────
# 규칙 테이블 (1차 판정) — 데모용 최소 세트.
# 실제 규칙 테이블(WS1 사실표 기반)이 준비되면 교체된다.
# ─────────────────────────────────────────────────────────────
_RULE_TABLE: dict[tuple[str, str], tuple[TriageLevel, list[str]]] = {
    ("초콜릿", "dog"): (TriageLevel.CALL_NOW, ["구토", "심박 증가", "발작"]),
    ("초콜릿", "cat"): (TriageLevel.CALL_NOW, ["구토", "심박 증가"]),
    ("포도", "dog"): (TriageLevel.EMERGENCY, []),
    ("건포도", "dog"): (TriageLevel.EMERGENCY, []),
    ("양파", "dog"): (TriageLevel.CALL_NOW, ["빈혈 증상", "무기력"]),
    ("양파", "cat"): (TriageLevel.CALL_NOW, ["빈혈 증상", "무기력"]),
    ("마늘", "dog"): (TriageLevel.CALL_NOW, ["빈혈 증상"]),
    ("자일리톨", "dog"): (TriageLevel.EMERGENCY, []),
    ("아보카도", "bird"): (TriageLevel.EMERGENCY, []),
    ("백합", "cat"): (TriageLevel.EMERGENCY, []),
    ("알로에", "cat"): (TriageLevel.VISIT_SOON, ["구토", "설사"]),
    ("초콜릿", "bird"): (TriageLevel.EMERGENCY, []),
}


_REFUSAL_MESSAGES: dict[str, str] = {
    "근거없음":   "제공된 자료에서 근거를 찾을 수 없습니다.",
    "검증실패":   "답변의 근거를 확인하지 못했습니다.",
    "되묻기상한": "필요한 정보를 확인하지 못해 답변을 드릴 수 없습니다.",
    "판정불가":   "상태를 판단할 근거가 부족합니다.",
    "범위밖":     "이 시스템은 반려동물 응급·건강 상담에 특화되어 있어 답변할 수 없습니다.",
}


class GraphEngine:
    """LangGraph 기반 질의 엔진.

    완료 기준:
      1. `pytest -m todo` 가 전부 통과한다 ✅
      2. `PETTRIAGE__SERVE__ENGINE=graph` 로 띄워 `/api/ask` 가 세 상태를 모두 낸다
      3. `tests/test_api.py` 68건이 그대로 통과한다 — 계약은 바뀌지 않는다
    """

    name = "graph"

    def __init__(self) -> None:
        from .nodes import NODES_IMPLEMENTED

        if not NODES_IMPLEMENTED:
            raise EngineNotReady(
                "그래프 노드가 비어 있다. src/pettriage/graph/nodes/ 를 구현하고 "
                "nodes/__init__.py 의 NODES_IMPLEMENTED 를 True 로 바꿀 것. "
                "남은 일: pytest -m todo"
            )

    def ask(self, req: AskRequest, session: Session) -> AskResponse:
        """질의 파이프라인 1회 실행."""
        # 세션 슬롯에 새 발화 정보 병합 (진전 있으면 되묻기 카운터 리셋).
        progressed = session.merge(req)
        if progressed:
            session.clarify_turns = 0

        state = self._build_state(req, session)

        try:
            state = self._run_pipeline(state)
        except Exception as e:
            log.error(
                "graph pipeline failure — type=%s session=%s",
                type(e).__name__, session.session_id,
            )
            return self._refused(session, "판정불가", _REFUSAL_MESSAGES["판정불가"])

        return self._build_response(state, session)

    # ── 파이프라인 ───────────────────────────────────────────

    def _build_state(self, req: AskRequest, session: Session) -> GraphState:
        """AskRequest + Session → 초기 GraphState."""
        slots: dict = {}
        if session.species:
            slots["species"] = session.species
        if session.weight_kg is not None:
            slots["weight_kg"] = session.weight_kg
        if session.amount_g is not None:
            slots["amount_g"] = session.amount_g

        return initial_state(
            question=req.question,
            session_id=session.session_id,
            pet_id=req.pet_id or "",
            slots=slots,
            clarify_turns=session.clarify_turns,
        )

    def _run_pipeline(self, state: GraphState) -> GraphState:
        """노드를 순서대로 실행. 분기 조건 만나면 조기 반환."""
        # ① 분류
        state.update(classify_intent(state))

        # D-46: general/unknown 은 도메인 밖 → 검색 없이 거절
        if state.get("intent") in ("general", "unknown"):
            state["status"] = "refused"
            state["refusal_reason"] = "범위밖"
            return state

        # ② 슬롯 추출
        state.update(extract_slots(state))

        # ③ 결측이면 되묻기
        if state.get("missing_slots"):
            state.update(ask_clarify(state))
            if state.get("status") != "refused":
                state["status"] = "clarify"
            return state

        # ④·⑤ 필터 + 검색
        state.update(build_filter(state))
        state.update(retrieve(state))

        if not state.get("hits"):
            state["status"] = "refused"
            state["refusal_reason"] = "근거없음"
            return state

        # ⑥ 계산
        state.update(compute_metrics(state))

        # 규칙 테이블 1차 판정 (rule_level 세팅)
        self._apply_rule_table(state)

        # ⑦·⑧ 압축 + 초안
        state.update(compress_context(state))
        state.update(generate_draft(state))

        # ⑨ 트리아지 (하향 금지 게이트)
        state.update(decide_triage(state))
        if state.get("status") == "refused":
            return state

        # ⑩ 근거 검증 (실패 시 1회 재검색)
        state.update(verify_grounding(state))
        if state.get("status") == "refused" and state.get("retry_count", 0) < MAX_RETRY:
            state["retry_count"] = state.get("retry_count", 0) + 1
            # 재검색 → 재압축 → 재초안 → 재검증
            state.pop("status", None)
            state.pop("refusal_reason", None)
            state.update(retrieve(state))
            if state.get("hits"):
                state.update(compress_context(state))
                state.update(generate_draft(state))
                state.update(verify_grounding(state))
            else:
                state["status"] = "refused"
                state["refusal_reason"] = "검증실패"
                return state

        if state.get("status") == "refused":
            return state

        # ⑪ 평이화
        state.update(simplify(state))

        state["status"] = "answered"
        return state

    def _apply_rule_table(self, state: GraphState) -> None:
        """규칙 테이블에서 (물질 × 종) 매칭 → rule_level 세팅.

        매칭 없으면 rule_level 은 None. decide_triage 가 판정불가로 처리한다.
        """
        slots = state.get("slots") or {}
        substance = slots.get("substance")
        species = slots.get("species")
        if not substance or not species:
            return
        key = (substance, species)
        if key in _RULE_TABLE:
            level, escalation = _RULE_TABLE[key]
            state["rule_level"] = int(level)
            if escalation:
                state["escalation_conditions"] = escalation

    # ── 응답 조립 ────────────────────────────────────────────

    def _build_response(self, state: GraphState, session: Session) -> AskResponse:
        """GraphState → AskResponse."""
        status = state.get("status", "refused")

        if status == "clarify":
            session.clarify_turns = state.get("clarify_turns", 1)
            return AskResponse(
                status="clarify",
                session_id=session.session_id,
                clarify=ClarifyPrompt(
                    missing=list(state.get("missing_slots") or []),
                    question=state.get("clarify_question", "추가 정보를 알려주세요."),
                    turn=state.get("clarify_turns", 1),
                ),
            )

        if status == "refused":
            return self._refused(
                session,
                state.get("refusal_reason", "판정불가"),
                _REFUSAL_MESSAGES.get(
                    state.get("refusal_reason", "판정불가"),
                    _REFUSAL_MESSAGES["판정불가"],
                ),
            )

        # answered — 성공한 경우 세션 되묻기 카운터 리셋
        session.clarify_turns = 0
        return AskResponse(
            status="answered",
            session_id=session.session_id,
            answer=state.get("answer") or state.get("draft", ""),
            triage=self._triage_result(state),
            citations=self._citations_from_hits(state.get("hits") or []),
        )

    def _refused(self, session: Session, reason: str, message: str) -> AskResponse:
        return AskResponse(
            status="refused",
            session_id=session.session_id,
            refusal=Refusal(reason=reason, message=message),  # type: ignore[arg-type]
        )

    def _triage_result(self, state: GraphState) -> TriageResult:
        level = int(state.get("triage_level") or TriageLevel.VISIT_SOON)
        lv = TriageLevel(level)
        return TriageResult(
            level=level,
            name=lv.name,
            badge=lv.badge,
            message=lv.message,
            escalation_conditions=list(state.get("escalation_conditions") or []),
            rule_level=state.get("rule_level"),
            llm_level=state.get("llm_level"),
        )

    def _citations_from_hits(self, hits: list) -> list[Citation]:
        """Hit → Citation 변환.

        publisher 는 청크 메타데이터에 없으므로 source_id 를 대체값으로 쓴다.
        실제 매니페스트 연동은 후속 작업.
        """
        cites: list[Citation] = []
        for h in hits:
            chunk = getattr(h, "chunk", None)
            if chunk is None:
                continue
            route = getattr(chunk, "route", "사실추출")
            cites.append(
                Citation(
                    source_id=getattr(chunk, "source_id", ""),
                    publisher=f"[출처 {getattr(chunk, 'source_id', '?')}]",
                    title=None,
                    route=route,
                    # 경로 ② 는 원문 인용 실을 수 없음 (D-37)
                    quote=None if route == "사실추출" else getattr(chunk, "text", None),
                )
            )
        return cites