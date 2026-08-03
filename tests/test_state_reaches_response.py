"""상태에 담긴 값이 **응답까지 나오는가** (D-22 · D-16 · D-47 · 05 §6).

2026-08-02 데이터 흐름 점검에서 `GraphState` 키 다섯이 **채우는 사람은 있는데
읽는 사람이 없는** 상태였다 —

    llm_fallbacks     성공 종료 노드가 세우고 엔진이 안 읽었다
    computed          compute 노드가 채우고 아무도 안 읽었다 (D-16 의 결과물이다)
    removed_contacts  finalize 노드가 채우고 아무도 안 읽었다 (D-47)
    slot_llm_used     ② 가 채우고 아무도 안 읽었다
    risk              ① 이 intent 사본을 넣고 아무도 안 읽었다

끊긴 자리는 **테스트가 없어서** 안 보였다. 노드 단위 테스트는 "노드가 키를 돌려주는가"
까지만 보고, 계약 테스트는 "응답이 유효한가" 부터 본다. **그 사이가 비어 있었다.**
이 파일이 그 사이를 본다.
"""

from __future__ import annotations

from dataclasses import dataclass

import pytest

from pettriage.app.contracts import AskResponse
from pettriage.app.session import Session
from pettriage.graph.engine import GraphEngine
from pettriage.graph.state import GraphState


@pytest.fixture
def engine() -> GraphEngine:
    """`__init__` 을 건너뛴다 — 응답 조립만 보므로 그래프를 컴파일할 필요가 없다."""
    return GraphEngine.__new__(GraphEngine)


@pytest.fixture
def session() -> Session:
    return Session(session_id="s-1")


@dataclass
class _Chunk:
    source_id: str = "S-034"
    route: str = "사실추출"
    text: str = "본문"


@dataclass
class _Hit:
    chunk: _Chunk


def _answered_state(**kw) -> GraphState:
    st: GraphState = {
        "status": "answered",
        "answer": "수의사에게 연락하세요.",
        "hits": [_Hit(_Chunk())],
        "triage_level": 2,
        "rule_level": 2,
        "escalation_conditions": ["구토가 계속되면"],
    }
    st.update(kw)  # type: ignore[typeddict-item]
    return st


# ── 죽은 키는 스키마에서도 사라졌다 ──────────────────────────────
class Test지운키:
    @pytest.mark.parametrize("key", ["risk", "slot_llm_used"])
    def test_상태에_없다(self, key):
        """읽는 곳이 없는 키를 스키마에 남겨 두지 않는다.

        남겨 두면 *"누군가 쓰고 있겠지"* 로 보여 다음 사람이 못 지운다.
        `risk` 는 값까지 `intent` 의 사본이라 읽을 것도 없었다.
        """
        assert key not in GraphState.__annotations__


# ── 폴백 기록 ───────────────────────────────────────────────────
class Test폴백기록:
    def test_다섯_태스크가_같은_문을_쓴다(self):
        """①분류·②슬롯이 폴백을 **기록하지 않던** 것이 이 검사의 이유다.

        기록하는 자리가 `generate.py` 안에 있어서, 그 파일의 노드만 남겼다.
        """
        from pettriage.graph import fallbacks
        from pettriage.graph.nodes import classify, generate, slots

        for mod in (classify, slots, generate):
            assert mod.note_fallback is fallbacks.note_fallback, mod.__name__

    def test_모델이_없으면_분류가_기록한다(self, monkeypatch):
        from pettriage.graph.fallbacks import LLM_FALLBACKS, reset_llm_fallbacks
        from pettriage.graph.nodes import classify

        reset_llm_fallbacks()
        monkeypatch.setattr(
            "pettriage.models.serving.factory.get_client", lambda: None, raising=False
        )
        classify.classify_intent({"question": "강아지가 초콜릿을 먹었어요"})
        assert "classify" in LLM_FALLBACKS
        reset_llm_fallbacks()

    def test_모델이_없으면_슬롯이_기록한다(self, monkeypatch):
        from pettriage.graph.fallbacks import LLM_FALLBACKS, reset_llm_fallbacks
        from pettriage.graph.nodes import slots

        reset_llm_fallbacks()
        monkeypatch.setattr(
            "pettriage.models.serving.factory.get_client", lambda: None, raising=False
        )
        assert slots._llm_slots("강아지가 초콜릿을 먹었어요") is None
        assert "slot" in LLM_FALLBACKS
        reset_llm_fallbacks()

    def test_되묻기와_거절에도_실린다(self, engine, session):
        """🔴 **성공 경로에만 붙이면 안 된다.**

        예전에는 성공 종료 노드가 세워서, 되묻기·거절로 끝난 건은 기록을 잃었다.
        04 §3 이 확인해야 하는 것 중에는 *"거절된 건이 모델을 타긴 했나"* 가 있다.
        """
        for st in (
            {"status": "clarify", "missing_slots": ["species"], "clarify_question": "어떤 동물"},
            {"status": "refused", "refusal_reason": "범위밖"},
            _answered_state(),
        ):
            st["llm_fallbacks"] = ["classify", "slot"]  # type: ignore[typeddict-item]
            resp = engine._build_response(st, session)  # type: ignore[arg-type]
            assert resp.llm_fallbacks == ["classify", "slot"], resp.status


# ── 계산 수치 (D-16) ────────────────────────────────────────────
class Test계산수치:
    def test_응답에_실린다(self, engine, session):
        st = _answered_state(computed={"dose_per_kg": 4.0, "unit": "g/kg"})
        resp = engine._build_response(st, session)
        assert resp.computed is not None
        assert resp.computed.dose_per_kg == 4.0
        assert resp.computed.unit == "g/kg"

    def test_계산할_것이_없으면_None(self, engine, session):
        """**빈 값을 0으로 채우지 않는다** (D-10).

        빈 모델을 실으면 *"계산했는데 0이다"* 로 읽힌다 — 체중을 몰라 못 한 것과 다르다.
        """
        assert engine._build_response(_answered_state(computed={}), session).computed is None
        assert engine._build_response(_answered_state(), session).computed is None

    def test_조류는_열량_칸을_쓴다(self, engine, session):
        """종마다 채워지는 칸이 다르다 (D-09)."""
        st = _answered_state(
            computed={"daily_energy_kcal": 55.0, "formula": "BER", "unit": "kcal/day"}
        )
        c = engine._build_response(st, session).computed
        assert c is not None and c.daily_energy_kcal == 55.0
        assert c.dose_per_kg is None


# ── 연락처 차단 (D-47) ──────────────────────────────────────────
class Test연락처:
    def test_개수만_나가고_문장은_안_나간다(self, engine, session):
        """🔴 **뺀 문장을 응답에 실으면 방금 지운 번호가 되살아난다.**

        `removed` 안에는 차단 대상 번호가 그대로 들어 있다 (`contacts.ScrubResult`
        docstring — *"`removed` 는 로그용이다"*). 필드 이름만 바꿔 되돌려주는 것은
        차단이 아니다.
        """
        leaked = "구토가 계속되면 855-764-7661 로 연락하세요."
        st = _answered_state(removed_contacts=[leaked])
        resp = engine._build_response(st, session)

        assert resp.removed_contact_count == 1
        assert "855" not in resp.model_dump_json()

    def test_안_뺐으면_0(self, engine, session):
        assert engine._build_response(_answered_state(), session).removed_contact_count == 0

    def test_래퍼가_더한다(self):
        """그래프가 뺀 것과 래퍼가 뺀 것은 **합쳐야** 이 응답에서 빠진 수가 된다.

        `finalize` 가 뺀 문장은 래퍼에 도착하기 전에 이미 사라져 래퍼의 `removed` 에
        안 잡힌다. 덮어쓰면 앞의 것이 없던 일이 된다.
        """
        from pettriage.app.contracts import Refusal
        from pettriage.app.safety_engine import scrub_response

        # `model_construct` 로 만든다 — 계약(`_no_foreign_contacts`)이 이런 응답을
        # 애초에 못 만들게 막기 때문이다. 래퍼는 **그 계약에 닿기 전**에 도는 자리이고,
        # 여기서 재현하려는 것이 정확히 그 순간이다 (`test_api.py` 의 `Leaky` 와 같은 수법).
        resp = AskResponse.model_construct(
            status="refused",
            session_id="s-1",
            refusal=Refusal(reason="근거없음", message="855-764-7661 로 연락하세요."),
            removed_contact_count=2,  # 그래프의 finalize 가 이미 2건 뺐다
        )
        out = scrub_response(resp)
        assert out.removed_contact_count > 2
        assert "855" not in out.model_dump_json()


# ── 재검색 때 등급을 다시 매기지 않는다 ─────────────────────────
class Test재검색:
    def test_judge_는_재검색에서_돌지_않는다(self, monkeypatch):
        """`build._after_retrieve` 주석이 막으려던 것을 **실제로 막는다.**

        그 갈림길은 `compute`·`rules` 만 건너뛰었고 `judge` 는 그대로 다시 돌았다.
        같은 질의가 회차마다 다른 등급을 내면 04 §8 의 재현성이 깨진다.
        """
        from pettriage.graph.nodes import generate

        calls: list[str] = []
        monkeypatch.setattr(generate, "_call_raw", lambda *a, **k: calls.append("x") or "EMERGENCY")

        first = generate.judge_triage({"context": "근거 문장", "retry_count": 0})
        assert first == {"llm_level": 4}
        assert len(calls) == 1

        again = generate.judge_triage({"context": "근거 문장", "retry_count": 1})
        assert again == {}
        assert len(calls) == 1, "재검색인데 LLM 을 다시 불렀다"


# ── 하네스가 집계한다 ───────────────────────────────────────────
class Test하네스집계:
    def test_전건_폴백이면_모델을_안_탄_것으로_센다(self):
        import sys
        from pathlib import Path

        sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "eval" / "harness"))
        from metrics import score_case, summarize

        rows = [
            score_case(
                {"case_id": f"G-{i}", "expected_status": "refused"},
                status="refused",
                level=None,
                answer_text="",
                citations=[],
                llm_fallbacks=("classify", "slot"),
            )
            for i in range(3)
        ]
        s = summarize(rows)
        assert s.fully_llm == 0
        assert s.fully_llm_rate == 0.0
        assert s.fallback_counts["classify"] == 3
        assert s.fallback_counts["slot"] == 3
