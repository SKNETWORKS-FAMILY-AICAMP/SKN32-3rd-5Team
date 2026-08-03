"""LangGraph `StateGraph` 조립.

설계 근거: 05 §5 · 02 §6 · D-09 · D-46 · D-49 · D-63

    05 §5 가 랭그래프를 고른 이유는 하나다 —
    *"되묻기 순환 + 근거 검증 실패 시 재검색이 있어 **선형 체인으로 표현 불가**"*.

    그런데 2026-08-02 까지 `import langgraph` 는 소스 어디에도 없었다.
    폴더 이름이 `graph/` 였고, 엔진 docstring 이 *"LangGraph 기반"* 이라고 적혀 있었고,
    `pyproject.toml [rag]` 에 의존성이 선언돼 있었을 뿐이다. 실체는
    `GraphEngine._run_pipeline` 79줄짜리 **손으로 펼친 선형 실행기**였고,
    05 §5 가 *"표현 불가"* 라고 한 그 순환은 이렇게 복붙돼 있었다 —

        compress → generate → verify                      # 정상 경로
        if 실패: retrieve → compress → generate → verify   # ← 같은 4줄이 또

    **문서에 적힌 결정이 코드에서 강제되지 않고 있었다** (D-48).

## 이 파일이 바꾸는 것

    ① 위 4줄 중복이 **조건부 엣지 하나**가 된다 (`verify → retrieve`).
    ② `retry_count` 상한 관리가 `verify_grounding` 안과 엔진 두 곳에 있었다.
       이제 상한 판단은 라우터 한 곳이고 `verify_grounding` 은 판정만 한다 (P2).
    ③ **노드가 상태에 무엇을 쓸 수 있는지 `GraphState` 가 강제한다.**
       예전에는 `state["llm_fallbacks"] = ...` 처럼 스키마에 없는 키를
       `# type: ignore[typeddict-unknown-key]` 로 넘겼다 — 런타임은 아무 말도 안 했다.
       랭그래프는 스키마 밖 키를 쓰면 **거기서 터진다** (D-40).

## 리듀서를 두지 않는다

    `GraphState` 에 `Annotated[..., reducer]` 를 하나도 붙이지 않는다.
    붙이지 않으면 채널이 기본값(last-value)이라 노드 반환 dict 이 그 키만 덮는다 —
    **기존 `state.update(...)` 와 의미가 정확히 같다.**

    ⚠️ 잘못 붙이면 `rule_level` 이 조용히 덮인다. 그것은 `apply_gate` 의 **바닥**이고,
    바닥이 무너지면 **하향 금지 게이트(D-09)가 통째로 무력해진다.** 이식하면서
    의미를 바꾸지 않는 것이 이 파일의 유일한 안전 조건이다.

## 순환의 상한

    루프는 `verify → retrieve` 하나뿐이고 `MAX_RETRY` 가 실제 상한이다.
    `RECURSION_LIMIT` 은 그것이 깨졌을 때를 위한 **뒷받침**이지 설계상의 상한이 아니다.
"""

from __future__ import annotations

import logging
from functools import lru_cache
from typing import Any

from .nodes import (
    MAX_RETRY,
    apply_rule_table,
    ask_clarify,
    build_filter,
    classify_intent,
    compress_context,
    compute_metrics,
    decide_triage,
    extract_slots,
    generate_draft,
    judge_triage,
    retrieve,
    simplify,
    verify_grounding,
)
from .state import GraphState

log = logging.getLogger(__name__)

#: 재귀 상한. 실제 상한은 `MAX_RETRY` 이고 이것은 **그것이 깨졌을 때의 뒷받침**이다.
#: 최장 경로는 재검색 1회를 포함해 18스텝이라 기본값 25 와 여유가 크지 않다.
RECURSION_LIMIT = 50


# ── 노드: 종료 상태를 세우는 자리들 ──────────────────────────────
#
# 거절 이유를 라우터가 아니라 **노드가** 쓴다. 라우터는 어디로 갈지만 정한다 —
# 라우터가 상태를 고치면 그래프를 읽어서 무슨 일이 벌어지는지 알 수 없게 된다.


def _refuse_out_of_scope(state: GraphState) -> GraphState:
    """도메인 밖 — **검색조차 하지 않는다** (D-46).

    유사도 임계값이 이것을 막아 줄 것으로 설계돼 있었으나 실측에서 성립하지 않았다.
    *"고양이 이름 지어주세요"*(0.550)가 근거 있는 질의의 최저점(0.547)보다 높다.
    검색하면 무관한 청크가 0.5대로 딸려 오므로 **여기서 끊는 것이 유일한 방어다.**
    """
    return {"status": "refused", "refusal_reason": "범위밖"}


def _refuse_no_evidence(state: GraphState) -> GraphState:
    """검색 0건 — **빈 결과는 실패가 아니라 신호다** (02 §8.3)."""
    return {"status": "refused", "refusal_reason": "근거없음"}


def _refuse_verification(state: GraphState) -> GraphState:
    """재검색까지 하고도 근거를 못 붙였다. 애매하면 거절이 맞다 (04 §4.1.0)."""
    return {"status": "refused", "refusal_reason": "검증실패"}


def _clarify(state: GraphState) -> GraphState:
    """되묻기. 상한을 넘으면 `ask_clarify` 가 이미 거절을 세워 두므로 덮지 않는다."""
    out = dict(ask_clarify(state))
    if out.get("status") != "refused":
        out["status"] = "clarify"
    return out  # type: ignore[return-value]


def _retry(state: GraphState) -> GraphState:
    """재검색 진입 — **거절 표시를 지우고** 카운터를 올린다.

    지우지 않으면 `verify` 가 세운 `refused` 가 그대로 남아 재시도가 성공해도
    거절로 나간다. 지우는 자리를 노드로 둔 이유는 위와 같다 — 라우터는 상태를 안 고친다.
    """
    return {
        "retry_count": state.get("retry_count", 0) + 1,
        "status": None,  # type: ignore[typeddict-item]
        "refusal_reason": "",
    }


def _answered(state: GraphState) -> GraphState:
    """성공 종료. **폴백 기록을 응답 단위로 고정한다** (05 §6).

    전역 `LLM_FALLBACKS` 는 서버에서 요청 간에 누적되므로 그대로 읽으면
    *"이 응답이 폴백으로 만들어졌는가"* 에 답할 수 없다.
    """
    from .nodes.generate import LLM_FALLBACKS

    return {"status": "answered", "llm_fallbacks": sorted(LLM_FALLBACKS)}


# ── 라우터: 어디로 갈지만 정한다 ─────────────────────────────────


def _after_classify(state: GraphState) -> str:
    # D-46 — `general`·`unknown` 은 우리가 다루지 않는 질문이다.
    return "refuse_scope" if state.get("intent") in ("general", "unknown") else "extract"


def _after_slots(state: GraphState) -> str:
    """② 뒤 갈림길 **셋**. *"왜 물질을 못 올렸나"* 가 행선지를 가른다 (D-68).

    - **종밖** — 물질은 아는데 이 종 자료가 없다 → `근거없음`. 되물어도 답이 안 나온다
    - **결측** — 종이 없거나 물질을 모른다 → 되묻는다 (D-10 · D-49)
    - 그 외 → 검색. `모호` 는 후보를 들고 여기로 온다 (D-62 — 모호는 실패가 아니다)
    """
    if state.get("off_species_substance"):  # type: ignore[typeddict-item]
        return "refuse_nohit"
    return "clarify" if state.get("missing_slots") else "filter"


def _after_retrieve(state: GraphState) -> str:
    """검색 뒤 갈림길 **넷**. 재검색이었는지가 이유와 목적지를 모두 바꾼다.

    - 히트 있음 · 첫 검색  → 계산부터
    - 히트 있음 · 재검색   → **계산·판정을 다시 하지 않는다.** 등급은 이미 정해졌고
      재검색은 *근거 문장*을 다시 붙이려는 것이다. 다시 돌리면 같은 입력에
      LLM 판정이 한 번 더 끼어들어 등급이 흔들린다
    - 히트 없음 · 첫 검색  → `근거없음`
    - 히트 없음 · 재검색   → `검증실패` (원래 실패 이유를 유지한다)
    """
    retried = state.get("retry_count", 0) > 0
    if state.get("hits"):
        return "compress" if retried else "compute"
    return "refuse_verify" if retried else "refuse_nohit"


def _after_triage(state: GraphState) -> str:
    # 조건 없는 MONITOR·판정 근거 없음 → 거절 (D-39). 여기서 끝난다.
    return "end" if state.get("status") == "refused" else "verify"


def _after_verify(state: GraphState) -> str:
    """④ 검증 뒤. **이 갈림길이 이 파일이 존재하는 이유다** (05 §5).

    실패 → 재검색 → 재압축 → 재초안 → 재검증. 예전에는 이 네 단계가
    정상 경로 바로 아래에 **복붙**돼 있었다.
    """
    if state.get("status") != "refused":
        return "simplify"
    if state.get("retry_count", 0) < MAX_RETRY:
        return "retry"
    return "end"


# ── 조립 ────────────────────────────────────────────────────────


def build_graph() -> Any:
    """`StateGraph` 를 조립해 컴파일한다.

    노드는 전부 `state -> 부분 dict` 순수 함수라 랭그래프 노드 시그니처와 **이미 같다.**
    이식에서 함수 본문은 하나도 건드리지 않았다 — 바뀐 것은 *순서를 누가 정하는가* 뿐이다.
    """
    from langgraph.graph import END, START, StateGraph

    g = StateGraph(GraphState)

    # ⚠️ 노드 이름은 **상태 키와 겹칠 수 없다** (랭그래프 제약).
    #    `slots`·`draft` 가 겹쳐서 조립이 터졌다. 이름을 상태 키와 분리해 두면
    #    *"이 이름은 단계인가 값인가"* 를 읽는 사람도 헷갈리지 않는다.

    g.add_node("classify", classify_intent)
    g.add_node("extract", extract_slots)
    g.add_node("clarify", _clarify)
    g.add_node("filter", build_filter)
    g.add_node("retrieve", retrieve)
    g.add_node("compute", compute_metrics)
    g.add_node("rules", apply_rule_table)
    g.add_node("compress", compress_context)
    g.add_node("generate", generate_draft)
    g.add_node("judge", judge_triage)
    g.add_node("decide", decide_triage)
    g.add_node("verify", verify_grounding)
    g.add_node("retry", _retry)
    g.add_node("simplify", simplify)
    g.add_node("answered", _answered)
    g.add_node("refuse_scope", _refuse_out_of_scope)
    g.add_node("refuse_nohit", _refuse_no_evidence)
    g.add_node("refuse_verify", _refuse_verification)

    g.add_edge(START, "classify")
    g.add_conditional_edges(
        "classify", _after_classify, {"refuse_scope": "refuse_scope", "extract": "extract"}
    )
    g.add_conditional_edges(
        "extract",
        _after_slots,
        {"clarify": "clarify", "filter": "filter", "refuse_nohit": "refuse_nohit"},
    )
    g.add_edge("filter", "retrieve")
    g.add_conditional_edges(
        "retrieve",
        _after_retrieve,
        {
            "compute": "compute",
            "compress": "compress",
            "refuse_nohit": "refuse_nohit",
            "refuse_verify": "refuse_verify",
        },
    )

    # 계산 → 규칙 바닥 → 압축 → 초안. 순서를 바꾸면 `rule_level` 없이 판정이 돈다.
    g.add_edge("compute", "rules")
    g.add_edge("rules", "compress")
    g.add_edge("compress", "generate")

    # ⑨ LLM 판정 **다음에** 게이트. 순서를 뒤집으면 `max(rule, llm)` 이
    #    llm_level=None 으로 돌아 상승이 통째로 사라진다 (D-09 · D-50).
    g.add_edge("generate", "judge")
    g.add_edge("judge", "decide")
    g.add_conditional_edges("decide", _after_triage, {"end": END, "verify": "verify"})

    g.add_conditional_edges(
        "verify", _after_verify, {"simplify": "simplify", "retry": "retry", "end": END}
    )
    g.add_edge("retry", "retrieve")  # ← 05 §5 가 말한 그 순환

    g.add_edge("simplify", "answered")
    g.add_edge("answered", END)
    for n in ("clarify", "refuse_scope", "refuse_nohit", "refuse_verify"):
        g.add_edge(n, END)

    return g.compile()


@lru_cache(maxsize=1)
def get_graph() -> Any:
    """컴파일된 그래프 1개를 프로세스에 상주시킨다 (D-53 — 팩토리를 쓴다)."""
    return build_graph()
