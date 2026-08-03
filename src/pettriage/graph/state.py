"""LangGraph 상태 (02 §6.1).

설계 근거: docs/02_시스템-아키텍처.md §6 · docs/05 §3

    이 State 는 **되묻기 세션 상태**다 — 조각 3. 휘발성이고 한 질의 안에서만 산다.
    반려동물 일일 기록은 여기 들어오지 않는다. 그건 조각 4(RAG)의 검색 대상이다 (05 §3).

노드는 State 를 받아 **바뀐 키만** 돌려준다. LangGraph 가 병합한다.
"""

from __future__ import annotations

import logging
from typing import Any, Literal

# ⚠️ `typing.TypedDict` 가 아니라 **`typing_extensions`** 다.
#    파이썬 3.11 이하의 `typing.TypedDict` 는 필드별 `Required/NotRequired` 정보를
#    런타임에 안 남겨서 pydantic 이 스키마를 못 만든다. 랭그래프가 그 스키마로
#    그래프 구조를 그리므로, 이것을 안 바꾸면 **아키텍처 다이어그램을 생성할 수 없다**
#    (필수 산출물 ②). 동작은 완전히 같다.
from typing_extensions import TypedDict

log = logging.getLogger(__name__)

Intent = Literal["intoxication", "symptom", "nutrition", "general", "unknown"]


class Slots(TypedDict, total=False):
    """② 슬롯 추출 결과. **없는 값은 키를 두지 않는다** — 추정 금지 (D-10)."""

    species: str  # dog · cat · bird
    breed: str
    weight_kg: float
    age_month: int
    substance: str
    #: `substance` 가 **추정 별칭을 타고 올라왔나** (D-59 ⑤ · D-62).
    #:
    #: 참이면 ② 를 부른 쪽이 `AskResponse.assumed_substance` 로 **옮겨야 한다.**
    #: 옮기지 않으면 `프라이팬 → PTFE` 같은 도약이 **확정처럼 나간다** —
    #: 무쇠·스테인리스 팬은 PTFE 를 내지 않는다. **밝히지 않은 추정은 환각이다.**
    #:
    #: 이 키가 없어서 `resolve_substance` 가 계산한 `assumption` 이 여기서 버려지고 있었다
    #: (2026-08-02 존재의의 재검토). `_assumption_must_be_stated` 는 그것을 막으려고
    #: 만든 계약인데 **필드를 채워 줄 경로가 끊겨 있어 발동하지 않았다.**
    #:
    #: 거짓일 때는 **키를 두지 않는다** (D-10).
    substance_is_assumed: bool
    #: 추정을 탔을 때 **사용자가 실제로 쓴 말**. `substance_is_assumed` 와 짝이다.
    #:
    #: 가정을 밝히려면 *"무엇을 무엇으로 봤는지"* 둘 다 필요하다 — `PTFE 로 가정했습니다`
    #: 만으로는 사용자가 자기 말(`프라이팬`)과 연결하지 못해 **정정할 기회를 잃는다** (D-59 ④).
    #: 확정으로 올라온 경우에는 키를 두지 않는다 (D-10).
    substance_surface: str
    amount_g: float
    elapsed_hours: float
    signs: list[str]


class GraphState(TypedDict, total=False):
    """질의 파이프라인 전체가 공유하는 상태."""

    # ── 입력 ────────────────────────────────────────────────
    question: str
    session_id: str
    pet_id: str

    # ── ① 분류 ──────────────────────────────────────────────
    intent: Intent

    # ── ② 슬롯 ──────────────────────────────────────────────
    slots: Slots
    missing_slots: list[str]
    clarify_turns: int
    clarify_question: str
    #: 물질을 하나로 못 좁혔을 때의 후보 전체. **하나를 고르면 나머지를 배제한 것이고
    #: 그 배제가 곧 진단이다** — 고르지 않고 전부 검색어로 넘긴다 (D-49 · D-58).
    substance_candidates: list[str]
    #: **물질은 아는데 이 종 자료가 없다** (D-68). 되묻기가 아니라 `근거없음` 거절로 간다.
    #:
    #: `향초` 는 코퍼스에 있으나 `covers={'bird'}` 뿐이다. 고양이 보호자에게
    #: *"어떤 물질인가요"* 를 되물어도 답이 안 나온다 — **응급 상황에서 못 쓸 질문은
    #: 거절보다 나쁘다.** 조류 자료를 고양이에 적용하는 것은 더 나쁘다 (D-10).
    off_species_substance: str
    #: **사용자가 말했는데 코퍼스에 없는 물질** (D-85). 되묻기가 아니라 `근거없음` 이다.
    #:
    #: `없음`(말 안 함)과 `모름`(말했는데 우리가 모름)은 다르다. 되물으면 같은 답이
    #: 돌아오고, 그 사이 시간이 간다. 슬롯에는 넣지 않는다 — 폐쇄 목록 밖 이름이
    #: 슬롯에 들어가면 계약이 막는 환각의 문이 열린다 (D-62).
    unknown_substance: str

    # ── 검색 · 계산 ─────────────────────────────────────────
    where: dict[str, Any]  # 검색 필터 — 코드가 만든다 (05 §4)
    hits: list[Any]  # retrieval.Hit
    #: 계산 노드 결과 — 체중당 섭취량(g/kg) · 조류 일일 권장 열량(kcal/day).
    #:
    #: **수치는 코드가 계산한다** (D-16). 그런데 2026-08-02 까지 이 값을 읽는 곳이
    #: 어디에도 없었다 — `compute_metrics` 가 채우고 그대로 버려졌고, 같은 노드가
    #: 함께 내는 `rule_level` 만 살아남았다. **계산했다는 증거가 응답에 없었다.**
    #: 지금은 `engine._build_response` 가 `AskResponse.computed` 로 올린다.
    #:
    #: ⚠️ 답변 *문장*에 넣는 것은 아직 아니다. 초안 프롬프트에 수치를 주면
    #:    답이 달라지고, 그러면 기준선과 비교군을 같은 자로 잴 수 없다 (04 §8).
    computed: dict[str, Any]

    # ── ③ 압축 · 생성 ───────────────────────────────────────
    context: str
    draft: str

    # ── 트리아지 ────────────────────────────────────────────
    rule_level: int | None
    #: `rule_level` 이 **실제 용량 계산**에서 나왔나 (D-80).
    #:
    #: 참이면 LLM 은 그 위로 등급을 올릴 수 없다 — 출처 달린 역치와 계산된 용량이
    #: 있는데 그 위로 올리는 것은 **근거 없는 상승**이다 (D-16).
    #: **`compute_metrics` 만 세운다.** 정성 표(`qualitative_level_for`)와
    #: 양 미상 바닥(D-79)은 세우지 않는다 — 거기서는 LLM 이 맞는 경우가 있고,
    #: 실제로 G-011·G-017 이 그렇게 고쳐졌다 (2026-08-03 실측).
    rule_is_quantitative: bool
    #: **규칙 등급을 누가 어떻게 냈나** (D-81). `triage.basis.Basis` 값.
    #: 세우는 곳은 `compute_metrics`(정량계산)와 `apply_rule_table`(정성·양미상) 둘뿐이다.
    rule_basis: str
    llm_level: int | None
    triage_level: int | None
    #: LLM 이 **올리려** 한 것을 게이트가 막았나 (D-80). 감사 정보다.
    llm_capped: bool
    #: **조건 없는 MONITOR 라서 거절했다** (D-39 · 04 §4.1.0).
    #:
    #: 거절이라 응답에 `triage` 가 없고, 그러면 채점에서 등급 분모에서 빠진다.
    #: 그런데 04 §4.1.0 은 이것을 **과소평가로 채점하라**고 정했다. 하네스가 그
    #: 규칙을 지킬 수 있도록 표시를 남긴다 — 없으면 `판정불가`(판정 근거 자체가
    #: 없는 경우)와 구분할 방법이 없다.
    monitor_without_conditions: bool
    escalation_conditions: list[str]

    # ── ④ 근거 검증 ─────────────────────────────────────────
    verdicts: list[dict[str, str]]  # 문장별 근거있음/근거없음/모순
    retry_count: int

    # ── 출력 ────────────────────────────────────────────────
    status: Literal["answered", "clarify", "refused"]
    answer: str
    refusal_reason: str
    #: 연락처 차단으로 뺀 **문장 원문** (D-47). 비면 아무것도 안 뺐다.
    #:
    #: 🔴 **이 목록을 그대로 응답에 실으면 안 된다.** 뺀 문장 안에 그 번호가 그대로 있다 —
    #:    막 지운 연락처를 다른 필드로 되돌려주는 꼴이다. `contacts.ScrubResult` 도
    #:    *"`removed` 는 로그용이다"* 라고 적어 두었다.
    #:    응답에는 **몇 개를 뺐는지**만 나간다 (`AskResponse.removed_contact_count`).
    removed_contacts: list[str]
    #: 이 응답을 만드는 동안 LLM 대신 폴백을 탄 태스크 이름 (05 §6).
    #: 전역 `fallbacks.LLM_FALLBACKS` 는 서버에서 요청 간에 누적되므로 **상태에 남긴다** —
    #: *"이 응답이 폴백으로 만들어졌는가"* 는 응답 단위로만 말할 수 있는 사실이다.
    #: 채우는 곳은 `engine._run_pipeline` 하나다 — 되묻기·거절 경로에도 실려야 한다.
    llm_fallbacks: list[str]


def set_substance(slots: Slots, name: str | None, species: str | None = None) -> Slots:
    """② 가 뽑은 **표면형**을 폐쇄 목록 위로 올려 상태에 넣는다 (D-59 ① · D-40).

        slots = set_substance(slots, llm_output.get("substance"))

    올리지 못하면 **키를 두지 않는다** (D-10). 부르는 쪽은 `missing_slots` 에 넣고
    `ask_clarify` 로 보낸다 — 02 §6 그래프의 `결측·물질미상 → ask_clarify` 가 그것이다.

    ⚠️ **예외를 던지지 않는다.** 처음에는 목록 밖이면 터지게 만들었는데,
    05 §6 이 정해 둔 실패 방식과 달랐다 —
    *①분류는 폴백 + 로그 · ②슬롯은 검증 실패 시 되묻기.*
    그리고 실측하니 흔한 표면형 30개 중 12개가 목록 밖이었다 (`커피`·`우유`·**`대파`**).
    터지게 두면 **정상 질의가 죽는다.** 2026-08-02 재검토에서 방향을 바꿨다.

    ⚠️ **이 함수가 있다고 강제되는 것은 아니다.** `Slots` 는 TypedDict 라
    노드가 `slots["substance"] = ...` 로 직접 써도 파이썬이 막지 못한다.
    마지막 방어선은 응답 계약(`contracts.SubstanceName`)이고, 정규화를 거친 값만
    올라오므로 **정상 경로에서는 걸리지 않는다** — 걸렸다면 이 문을 안 거쳤다는 뜻이다
    (`_no_foreign_contacts` 와 같은 지위).

    `모호` 로 못 정한 경우 후보를 잃지 않으려면 `resolve_substance` 를 직접 부른다 —
    후보 여럿은 **검색어로 전부 넘기고 LLM 이 다 읽게** 한다 (D-58).

    ⚠️ **추정 별칭을 탔으면 `substance_is_assumed` 가 참으로 선다.**
    부르는 쪽은 그것을 `AskResponse.assumed_substance` 로 **반드시 옮겨야 한다** —
    옮기지 않으면 `프라이팬 → PTFE` 같은 도약이 확정처럼 나가고, 그것이 환각이다.
    `tests/todo/test_graph_nodes.py` 가 노드 구현이 옮기는지 본다.
    """
    from ..compute.vocabulary import resolve_substance

    out: Slots = dict(slots)  # type: ignore[assignment]
    r = resolve_substance(name or "", species or slots.get("species"))
    if r.name is None:
        out.pop("substance", None)
        if r.surface:
            log.info(
                "물질을 폐쇄 목록에 못 올렸다 — %r (%s · 후보 %d). 되묻기로 보낸다 (D-49).",
                r.surface,
                r.how,
                len(r.candidates),
            )
        return out
    if r.how != "직접":
        log.info(
            "물질을 정규화했다 — %r → %r (%s%s)",
            r.surface,
            r.name,
            r.how,
            " · 추정" if r.assumption else "",
        )
    out["substance"] = r.name
    # **추정이라는 사실을 여기서 잃지 않는다.** 잃으면 도약이 확정처럼 나간다 (D-59 ⑤).
    if r.assumption:
        out["substance_is_assumed"] = True
        out["substance_surface"] = r.surface
    else:
        out.pop("substance_is_assumed", None)
        out.pop("substance_surface", None)
    return out


def initial_state(question: str, session_id: str, **kw: Any) -> GraphState:
    """빈 상태. **카운터는 반드시 0으로 시작한다** — 없으면 루프 상한이 안 먹는다."""
    st: GraphState = {
        "question": question,
        "session_id": session_id,
        "slots": {},
        "missing_slots": [],
        "clarify_turns": 0,
        "retry_count": 0,
        "hits": [],
        "computed": {},
        "verdicts": [],
        "escalation_conditions": [],
        "rule_level": None,
        "llm_level": None,
        "triage_level": None,
    }
    st.update(kw)  # type: ignore[typeddict-item]
    return st
