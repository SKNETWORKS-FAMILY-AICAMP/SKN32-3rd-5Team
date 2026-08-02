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
from .state import GraphState, initial_state

log = logging.getLogger(__name__)


class EngineNotReady(RuntimeError):
    """그래프 노드가 아직 구현되지 않았다.

    ⚠️ `nodes.NODES_IMPLEMENTED` 가 `True` 인 지금은 **나지 않는다.**
    노드를 다시 비우는 일이 생기면 `deps._build_engine` 이 이것을 잡아
    `EngineUnavailable` 로 올린다 — 그 경로를 살려 두려고 남긴다.
    """


_REFUSAL_MESSAGES: dict[str, str] = {
    "근거없음": "제공된 자료에서 근거를 찾을 수 없습니다.",
    "검증실패": "답변의 근거를 확인하지 못했습니다.",
    "되묻기상한": "필요한 정보를 확인하지 못해 답변을 드릴 수 없습니다.",
    "판정불가": "상태를 판단할 근거가 부족합니다.",
    "범위밖": "이 시스템은 반려동물 응급·건강 상담에 특화되어 있어 답변할 수 없습니다.",
}


def _assumption_notice(slots: dict) -> str:
    """**밝히지 않은 추정은 환각이다** — 그 가정을 문장 맨 앞에 세운다 (D-59 ⑤ · D-62).

    `프라이팬 → PTFE` 는 도약이다. 무쇠·스테인리스 팬은 PTFE 를 내지 않는다.
    답을 못 하는 것보다는 낫지만(D-13), **말없이 확정처럼 내보내는 것보다는 낫지 않다.**

    ⚠️ 이 문장을 `AskResponse.full_text` 안에서 만들지 않는다.
        계약(`_assumption_must_be_stated`)이 *"가정이 문장에 실렸나"* 를 보는데,
        계약이 스스로 그 문장을 붙이면 **자기가 붙인 것을 자기가 확인하는 꼴**이라
        검사가 항상 통과한다. 만드는 층과 검사하는 층을 분리한다 (D-57 · D-58).

    조사를 쓰지 않는다 — 물질명이 `PTFE`·`자일리톨`처럼 받침 유무가 갈려
    `으로/로`, `을/를` 이 문장마다 틀린다. 표기 사고는 신뢰를 깎는다.
    """
    surface = slots.get("substance_surface") or "말씀하신 것"
    return (
        f"[확인되지 않은 가정] '{surface}' = '{slots['substance']}'. "
        f"확인된 것이 아니니 다르면 알려주세요."
    )


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

        # ⚠️ **여기서 그래프를 컴파일한다.** 첫 질의로 미루지 않는다.
        #
        # 2026-08-02 실측 — `langgraph` 가 없는 환경에서 서버가 **정상 기동**하고,
        # 모든 질의가 `ImportError` 를 맞아 `판정불가` 거절로 나갔다. HTTP 200 이었다.
        # 팀원이 `git pull` 만 하고 재설치를 안 하면 정확히 이 상태가 된다 —
        # *"시스템이 다 거절해요"* 만 보이고 원인은 안 보인다.
        #
        # **평가를 돌리면 전부 거절로 집계된다.** `deps.EngineUnavailable` 이
        # *"조용히 스텁으로 내려가면 지표가 오염된다"* 며 막으려던 그 사고이고,
        # 게으른 컴파일이 그 방어를 우회하고 있었다. **크게 실패하게 둔다** (04 §8).
        try:
            from .build import get_graph

            get_graph()
        except ImportError as e:
            raise EngineNotReady(
                f"질의 그래프를 만들 수 없다 — {e}. `langgraph` 는 **핵심 의존성**이다 "
                "(2026-08-02 D-64 로 [rag] extra 에서 올라왔다). 저장소를 갱신했다면 "
                "재설치가 필요하다:\n"
                "  pip install -e '.[api,rag,ingest,dev]' -c constraints.txt"
            ) from e

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
                type(e).__name__,
                session.session_id,
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
        """컴파일된 `StateGraph` 를 1회 돌린다.

        ⚠️ **여기 순서를 다시 적지 않는다.** 2026-08-02 까지 이 메서드는 79줄짜리
            손으로 펼친 선형 실행기였고, 05 §5 가 *"선형 체인으로 표현 불가"* 라고
            적어 둔 재검색 순환이 **복붙 4줄**로 들어가 있었다. 순서는 `build.py`
            한 곳에만 있다 (D-40 · P2).

        `reset_llm_fallbacks()` 는 그래프 **밖**에서 부른다 — 전역 카운터를 비우는
        것은 요청 하나의 경계에서 일어나는 일이고, 그 경계를 아는 것은 엔진이다.
        """
        from .build import RECURSION_LIMIT, get_graph
        from .nodes.generate import reset_llm_fallbacks

        reset_llm_fallbacks()
        out = get_graph().invoke(state, config={"recursion_limit": RECURSION_LIMIT})
        return dict(out)  # type: ignore[return-value]

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

        slots = state.get("slots") or {}
        substance = slots.get("substance")
        assumed = bool(substance) and bool(slots.get("substance_is_assumed"))

        answer = state.get("answer") or state.get("draft", "")
        if assumed:
            answer = f"{_assumption_notice(slots)} {answer}".strip()

        return AskResponse(
            status="answered",
            session_id=session.session_id,
            answer=answer,
            triage=self._triage_result(state),
            citations=self._citations_from_hits(state.get("hits") or []),
            # **추정과 확정을 한 필드에 담지 않는다.** 담으면 읽는 쪽이 구분하지 않고
            # 쓰게 되고, 그 순간 도약이 확정이 된다. 둘 중 **하나만** 찬다 (D-59 ⑤).
            assumed_substance=substance if assumed else None,
            identified_substance=None if assumed else substance,
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
