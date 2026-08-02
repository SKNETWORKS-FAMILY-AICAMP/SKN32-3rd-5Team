"""그래프 조립 검사 — **랭그래프가 실제로 돌고 있는가.**

설계 근거: 05 §5 · D-40 · D-48 교훈 #6

이 파일이 있는 이유는 하나다. 2026-08-02 까지 이 저장소는

  - 폴더 이름이 `graph/` 였고
  - `GraphEngine` docstring 에 *"LangGraph 기반 질의 엔진"* 이라고 적혀 있었고
  - `pyproject.toml` 에 `langgraph>=0.2.62` 가 선언돼 있었는데

**`import langgraph` 가 소스 어디에도 없었다.** 409건이 전부 초록이었다.
테스트가 *"파이프라인이 옳은 답을 내는가"* 만 보고 *"무엇이 그 파이프라인인가"* 를
안 봤기 때문이다. 문서로 적어 두는 것으로는 다시 안 돌아간다는 보장이 없다 —
**못 어기게 만든다** (D-40).
"""

from __future__ import annotations

import pytest

from pettriage.graph import initial_state
from pettriage.graph.build import RECURSION_LIMIT, get_graph
from pettriage.graph.state import GraphState


def test_engine_runs_a_real_langgraph():
    """컴파일 결과가 랭그래프 객체다. 손수 만든 실행기로 되돌아가면 여기서 걸린다."""
    from langgraph.graph.state import CompiledStateGraph

    assert isinstance(get_graph(), CompiledStateGraph)


def test_langgraph_is_a_core_dependency():
    """`.[api]` 만 깔아도 그래프가 돈다 (D-48 교훈 #6).

    `[rag]` 에 두면 CI test 잡에서 **그래프가 한 줄도 실행되지 않는다.**
    424줄이 CI에서 한 번도 안 돌았던 그 사고와 같은 모양이다.
    """
    import tomllib
    from pathlib import Path

    root = Path(__file__).resolve().parents[1]
    pyproject = root / "pyproject.toml"
    if not pyproject.exists():  # 설치본에서 돌 때
        pytest.skip("pyproject.toml 없음 — 소스 트리에서만 검사한다")
    deps = " ".join(tomllib.loads(pyproject.read_text(encoding="utf-8"))["project"]["dependencies"])
    assert "langgraph" in deps, "langgraph 가 extra 로 내려가면 CI 에서 그래프가 안 돈다"


class TestTopology:
    """**그래프의 모양 자체가 설계 결정이다.** 엣지가 사라지면 결정이 사라진다."""

    def _edges(self) -> set[tuple[str, str]]:
        g = get_graph().get_graph()
        return {(e.source, e.target) for e in g.edges}

    def test_the_retry_loop_exists(self):
        """05 §5 가 랭그래프를 고른 **유일한 이유**가 이 순환이다.

        *"되묻기 순환 + 근거 검증 실패 시 재검색이 있어 선형 체인으로 표현 불가"*.
        이 엣지가 없으면 선형 체인이고, 그러면 랭그래프를 쓸 이유가 없다.
        """
        edges = self._edges()
        assert ("verify", "retry") in edges
        assert ("retry", "retrieve") in edges, "재검색 순환이 끊겼다 — 05 §5 의 근거가 사라진다"

    def test_out_of_scope_never_reaches_retrieval(self):
        """D-46 — 도메인 밖은 **검색조차 하지 않는다.**

        임계값이 막아 줄 것으로 설계돼 있었으나 실측에서 성립하지 않았다
        (*"고양이 이름 지어주세요"* 0.550 > 근거 있는 질의 최저점 0.547).
        `classify` 에서 끊는 것이 유일한 방어다.
        """
        edges = self._edges()
        assert ("classify", "refuse_scope") in edges
        assert ("classify", "retrieve") not in edges
        assert ("classify", "filter") not in edges

    def test_judgement_comes_before_the_gate(self):
        """D-09 — `judge` 다음에 `decide` 다.

        뒤집으면 `apply_gate` 가 `llm_level=None` 으로 돌아 **상승이 통째로 사라진다.**
        하향 금지 게이트는 남지만 올릴 것이 없어진다 (D-50 — 규칙 등급은 바닥이다).
        """
        edges = self._edges()
        assert ("generate", "judge") in edges
        assert ("judge", "decide") in edges
        assert ("generate", "decide") not in edges

    def test_rules_run_before_generation(self):
        """`rule_level` 은 `apply_gate` 의 **바닥**이다. 판정 전에 서야 한다 (D-50)."""
        edges = self._edges()
        assert ("compute", "rules") in edges
        assert ("rules", "compress") in edges

    def test_clarify_is_terminal(self):
        """되묻기는 **응답이다.** 계속 진행하면 결측 슬롯으로 검색이 돈다 (D-10)."""
        edges = self._edges()
        assert ("extract", "clarify") in edges
        assert not [t for s, t in edges if s == "clarify" and not t.startswith("__end__")]


class TestRouters:
    """라우터는 순수 함수다. 그래프를 안 돌리고 갈림길만 직접 본다."""

    def test_retry_keeps_its_own_refusal_reason(self):
        """재검색 뒤에도 못 찾으면 `근거없음` 이 아니라 **`검증실패`** 다.

        둘을 뭉치면 *"애초에 자료가 없었다"* 와 *"자료는 있는데 근거를 못 붙였다"* 가
        같은 이유로 기록돼 평가에서 원인을 못 가른다 (04 §4).
        """
        from pettriage.graph.build import _after_retrieve

        st: GraphState = initial_state("q", "s1", hits=[], retry_count=0)
        assert _after_retrieve(st) == "refuse_nohit"
        st["retry_count"] = 1
        assert _after_retrieve(st) == "refuse_verify"

    def test_retry_does_not_re_judge(self):
        """재검색은 **근거 문장을 다시 붙이려는 것**이지 등급을 다시 매기려는 것이 아니다.

        `compute → rules → judge → decide` 를 또 돌면 같은 입력에 LLM 판정이
        한 번 더 끼어들어 등급이 흔들린다 — 재현성이 깨지고 평가가 못 믿을 게 된다.
        """
        from pettriage.graph.build import _after_retrieve

        st: GraphState = initial_state("q", "s1", hits=[object()], retry_count=0)
        assert _after_retrieve(st) == "compute"
        st["retry_count"] = 1
        assert _after_retrieve(st) == "compress", "재검색인데 판정을 다시 돌린다"

    def test_retry_is_capped_by_max_retry(self):
        from pettriage.graph.build import _after_verify
        from pettriage.graph.nodes import MAX_RETRY

        st: GraphState = initial_state("q", "s1", status="refused", retry_count=0)
        assert _after_verify(st) == "retry"
        st["retry_count"] = MAX_RETRY
        assert _after_verify(st) == "end", "재검색 상한이 안 먹으면 무한 루프다"

    def test_routers_do_not_mutate_state(self):
        """**라우터는 어디로 갈지만 정한다.** 상태를 고치면 그래프를 읽어서 알 수 없다."""
        import copy

        from pettriage.graph.build import (
            _after_classify,
            _after_retrieve,
            _after_slots,
            _after_triage,
            _after_verify,
        )

        st: GraphState = initial_state(
            "q", "s1", intent="intoxication", hits=["h"], status="refused", retry_count=0
        )
        before = copy.deepcopy(dict(st))
        for router in (
            _after_classify,
            _after_slots,
            _after_retrieve,
            _after_triage,
            _after_verify,
        ):
            router(st)
        assert dict(st) == before, "라우터가 상태를 고쳤다"


def test_recursion_limit_is_above_the_longest_path():
    """상한은 `MAX_RETRY` 이고 `RECURSION_LIMIT` 은 뒷받침이다.

    최장 경로(재검색 1회 포함)가 상한을 넘으면 **정상 질의가 재귀 초과로 죽는다.**
    """
    from pettriage.graph.nodes import MAX_RETRY

    longest = 14 + 4 * MAX_RETRY  # 정상 경로 + 재검색마다 retry·retrieve·compress·generate·verify
    assert longest < RECURSION_LIMIT


def test_missing_langgraph_fails_loudly_not_silently(monkeypatch: pytest.MonkeyPatch):
    """**의존성이 없으면 크게 실패한다** — 조용히 전부 거절하지 않는다 (04 §8).

    2026-08-02 실측: `langgraph` 가 없는 환경에서 서버가 **정상 기동**하고
    모든 질의가 `ImportError` 를 맞아 `판정불가` 거절로 나갔다. **HTTP 200 이었다.**
    팀원이 `git pull` 만 하고 재설치를 안 하면 정확히 이 상태가 된다 —
    *"시스템이 다 거절해요"* 만 보이고 원인은 안 보이며, **평가는 전부 거절로 집계된다.**

    원인은 게으른 컴파일이었다. `get_graph()` 를 첫 질의로 미루면 `deps` 의
    `EngineUnavailable` 방어를 **우회한다.** 생성 시점에 컴파일해서 그 문을 지나게 한다.
    """
    import builtins

    real_import = builtins.__import__

    def blocked(name, *a, **kw):
        if name == "langgraph" or name.startswith("langgraph."):
            raise ImportError(f"No module named {name!r}")
        return real_import(name, *a, **kw)

    from pettriage.graph.build import get_graph as _g

    _g.cache_clear()
    monkeypatch.setattr(builtins, "__import__", blocked)

    from pettriage.graph.engine import EngineNotReady, GraphEngine

    with pytest.raises(EngineNotReady) as ei:
        GraphEngine()
    # **재설치 명령이 메시지에 실려야 한다.** 원인만 알고 해법을 모르면 반쯤 실패다.
    assert "pip install" in str(ei.value)

    monkeypatch.undo()
    _g.cache_clear()
