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
    risk: str

    # ── ② 슬롯 ──────────────────────────────────────────────
    slots: Slots
    missing_slots: list[str]
    clarify_turns: int
    clarify_question: str
    #: ②가 LLM 으로 뽑았나, 규칙 폴백이었나 (05 §6 — 폴백은 기록한다).
    slot_llm_used: bool
    #: 물질을 하나로 못 좁혔을 때의 후보 전체. **하나를 고르면 나머지를 배제한 것이고
    #: 그 배제가 곧 진단이다** — 고르지 않고 전부 검색어로 넘긴다 (D-49 · D-58).
    substance_candidates: list[str]
    #: **물질은 아는데 이 종 자료가 없다** (D-68). 되묻기가 아니라 `근거없음` 거절로 간다.
    #:
    #: `향초` 는 코퍼스에 있으나 `covers={'bird'}` 뿐이다. 고양이 보호자에게
    #: *"어떤 물질인가요"* 를 되물어도 답이 안 나온다 — **응급 상황에서 못 쓸 질문은
    #: 거절보다 나쁘다.** 조류 자료를 고양이에 적용하는 것은 더 나쁘다 (D-10).
    off_species_substance: str

    # ── 검색 · 계산 ─────────────────────────────────────────
    where: dict[str, Any]  # 검색 필터 — 코드가 만든다 (05 §4)
    hits: list[Any]  # retrieval.Hit
    computed: dict[str, Any]  # 계산 노드 결과 (체중당 섭취량 등)

    # ── ③ 압축 · 생성 ───────────────────────────────────────
    context: str
    draft: str

    # ── 트리아지 ────────────────────────────────────────────
    rule_level: int | None
    llm_level: int | None
    triage_level: int | None
    escalation_conditions: list[str]
    #: 게이트가 LLM의 하향 판정을 막았는가 (`triage.gate.apply_gate` 정의).
    #: `contracts.TriageResult` 가 rule_level·llm_level 과 대조해 검증한다.
    overridden: bool

    # ── ④ 근거 검증 ─────────────────────────────────────────
    verdicts: list[dict[str, str]]  # 문장별 근거있음/근거없음/모순
    retry_count: int

    # ── 출력 ────────────────────────────────────────────────
    status: Literal["answered", "clarify", "refused"]
    answer: str
    refusal_reason: str
    removed_contacts: list[str]  # 연락처 차단으로 뺀 문장 (D-47). 비면 아무것도 안 뺐다
    #: 이 응답을 만드는 동안 LLM 대신 폴백을 탄 태스크 이름 (05 §6).
    #: 전역 `generate.LLM_FALLBACKS` 는 서버에서 요청 간에 누적되므로 **상태에 남긴다** —
    #: *"이 응답이 폴백으로 만들어졌는가"* 는 응답 단위로만 말할 수 있는 사실이다.
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
        "overridden": False,
    }
    st.update(kw)  # type: ignore[typeddict-item]
    return st
