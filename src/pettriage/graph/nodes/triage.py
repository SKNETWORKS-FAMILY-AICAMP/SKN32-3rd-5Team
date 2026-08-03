"""트리아지 판정 노드.

설계 근거: 02 §7.2 · 05 §5 · D-09 · D-39

    🔒 **`max()` 를 직접 쓰지 않는다.** 반드시 `triage.gate.apply_gate` 를 부른다 —
    MONITOR 상승 조건 검사가 거기 붙어 있다.

    LLM 은 등급을 낮출 수 없다 — apply_gate 가 `max(rule, llm)` 로 강제한다.
"""

from __future__ import annotations

import logging

from ...triage.gate import MonitorWithoutConditions, apply_gate
from ...triage.levels import TriageLevel
from ..state import GraphState

log = logging.getLogger(__name__)


def apply_rule_table(state: GraphState) -> GraphState:
    """규칙 표에서 (물질 × 종) **바닥 등급**을 얻는다 (D-50).

    ⚠️ **여기에 표를 두지 않는다.** 2026-08-02 흡수 전에는 `engine.py` 안에
    `_RULE_TABLE` 12줄이 손으로 적혀 있었다 —

        ("초콜릿", "dog"): (CALL_NOW, [...])
        ("알로에", "cat"): (VISIT_SOON, [...])

    그 12줄이 `rule_level` 을 만들고, `rule_level` 은 `apply_gate` 의 **바닥**이다.
    바닥이 손으로 적은 값이면 하향 금지 게이트가 **근거 없는 등급을 지켜준다.**
    진짜 표는 사실 표 888행에서 `make rules` 로 파생된 CSV 두 장이다 (D-16 · D-22 · D-38).

    ⚠️ **이 함수가 엔진 안에 있었다** (`GraphEngine._apply_rule_table`).
        판정 로직이 배달 계층에 섞여 있었다는 뜻이고, 그래서 노드 테스트가 닿지 않았다.
        노드는 노드 폴더에 둔다 — 그래프가 부를 수 있는 것은 노드뿐이다 (D-40).

    정량 계산이 가능하면 `compute_metrics` 가 이미 `rule_level` 을 세운다.
    여기서는 **양을 모를 때의 정성 바닥**만 채운다 — 표에 (물질 × 종) 행이
    있다는 것 자체가 *"이 종에 이 물질은 위험하다"* 는 근거다.

    Returns:
        `{"rule_level": ..., "escalation_conditions": [...]}` 또는 `{}`.
        **빈 dict 은 실패가 아니다** — 근거가 없다는 뜻이고 `decide_triage` 가 판정불가로 보낸다.
    """
    if state.get("rule_level") is not None:
        return {}  # compute_metrics 가 정량으로 이미 정했다 — 덮지 않는다

    slots = state.get("slots") or {}
    substance = slots.get("substance")
    species = slots.get("species")
    if not substance or not species:
        return {}

    from ...compute.content import content_for, threshold_substance
    from ...compute.rules import qualitative_level_for

    out: GraphState = {}

    # **역치를 가진 이름으로 바꿔 찾는다** (D-78).
    #
    # `밀크 초콜릿` 은 어휘표에 있지만 역치 행이 0개다 — 역치는
    # `초콜릿(테오브로민+카페인)` 에 붙어 있다. 이 치환이 없으면 **물질을 정확히
    # 알아낸 질의가 판정불가로 거절된다.** 앞 단계가 다 맞았는데 마지막에서 버린다.
    # 계수가 없는 물질은 이름이 그대로 돌아온다.
    content = content_for(substance, species)
    substance = threshold_substance(substance, species) or substance

    # ① 코퍼스가 등급을 말한 경우 — **정성 등급 표**가 바닥이다 (D-50).
    #    조류는 정량 역치가 0행이라 **이 경로가 유일하다** (D-09 개정).
    v = qualitative_level_for(substance, species)
    if v.level is not None:
        out["rule_level"] = int(v.level)
        out["rule_basis"] = "정성"
        if v.conditions and not state.get("escalation_conditions"):
            out["escalation_conditions"] = list(v.conditions)
        log.info("rule_level(정성) %s — %s", int(v.level), v.reason)
        return out

    # ② 등급은 없지만 **역치 행이 있는** 경우 — 위험하다는 사실 자체는 근거가 있다.
    from ...compute.rules import BELOW_THRESHOLD_LEVEL, _signs_of, lookup

    rows = lookup(substance, species)
    if not rows:
        return {}  # 근거 없음 — decide_triage 가 판정불가로 보낸다

    # 🔴 **양을 모르는 것과 역치 미만인 것은 다르다** (D-79).
    #
    # 예전에는 둘을 같이 `BELOW_THRESHOLD_LEVEL`(MONITOR) 로 두었다. 주석도
    # *"양을 모르므로 역치 미만과 같이 다룬다"* 라고 적혀 있었다. 그런데 그것은
    # **모르는 것을 안전한 쪽으로 읽은 것**이고, 방향이 과소평가다 — D-13 이
    # 최우선 위험으로 꼽은 바로 그것이다.
    #
    #   G-001  "다크초콜릿을 새끼손톱만큼"   기대 CALL_NOW  →  실측 MONITOR (2026-08-02)
    #   골든셋 메모: *"소량이라 안전하다고 짐작하는 과소평가 유도형 질문"*
    #
    # 역치가 있다는 것은 **이 종에 이 물질의 독성이 확인됐다**는 뜻이다.
    # 거기에 양을 모른다면, 그때 할 일은 관찰이 아니라 **지금 전화해서 물어보는 것**이다.
    # 등급을 어림해 올리는 것이 아니라 **모름을 해소하는 행동**을 부르는 것이다.
    #
    # ⚠️ 양을 **알고** 계산했는데 역치 미만이면 그건 그대로 MONITOR 다. 그 판정은
    #    `compute_metrics` → `rule_level_for` 가 이미 내렸고 여기 오지 않는다.
    #    여기까지 왔는데 수치가 다 있다면 단위가 계산 불가한 경우다(개수 등) —
    #    그것도 *못 쟀다* 이지 *적다* 가 아니다.
    #
    # ⚠️ **함량이 "무의미"인 물질에는 이 바닥을 적용하지 않는다** — 자료가 유효성분
    #    함량이 무의미하다고 **말한** 것이지 우리가 모르는 것이 아니다. 체중을 알든
    #    모르든 유효성분 용량은 무의미하다. 물어볼 이유가 없는 것을 모른다고 등급을
    #    올리면 되묻기가 아니라 **겁주기**가 된다 (2026-08-02 G-041 실측).
    #
    #    ①에서 프롬프트의 `[자료가 말하는 것]` 과 `[확인 안 된 것]` 을 갈라 놓고
    #    정작 판정 코드는 안 갈라 놓았던 자리다.
    quantified = slots.get("weight_kg") is not None and slots.get("amount_g") is not None
    known_negligible = content is not None and content.negligible
    floor = BELOW_THRESHOLD_LEVEL if (quantified or known_negligible) else TriageLevel.CALL_NOW
    out["rule_level"] = int(floor)
    # 수치를 알고도 정량 판정을 못 한 경우(단위가 계산 불가 등)와 함량이 무의미한
    # 경우는 **자료가 그렇게 생긴 것**이라 `정성` 이다. 양을 모르는 것만 `양미상` 이다.
    out["rule_basis"] = "정성" if (quantified or known_negligible) else "양미상"
    if not quantified and not known_negligible:
        log.info(
            "%s(%s) 는 역치가 있는데 %s 를 모른다 — 바닥을 %s 로 둔다 (D-79)",
            substance,
            species,
            "체중·섭취량",
            floor.name,
        )
    if not state.get("escalation_conditions"):
        conditions = _signs_of(rows)
        if conditions:
            out["escalation_conditions"] = list(conditions)
    return out


def decide_triage(state: GraphState) -> GraphState:
    """규칙 1차 → 미적용 시 LLM → 하향 금지 게이트.

    규칙 테이블 입력이 **종별로 다르다** (D-09 개정).
    개·고양이는 `물질 × 체중당 섭취량` 정량, 조류는 `물질 × 증상` 정성이다.
    조류에 수치를 요구하면 근거에 없는 값을 모델이 지어낸다.

    Returns:
        `{"rule_level": ..., "llm_level": ..., "triage_level": ...,
          "escalation_conditions": [...]}`
        MONITOR인데 상승 조건이 없으면 `refused / 판정불가`.
    """
    rule_level = state.get("rule_level")
    llm_level = state.get("llm_level")
    escalation = state.get("escalation_conditions") or []

    try:
        decision = apply_gate(
            rule_level=rule_level,
            llm_level=llm_level,
            escalation_conditions=tuple(escalation),
            # **잰 자리에서는 LLM 이 올리지 못한다** (D-80).
            rule_is_quantitative=bool(state.get("rule_is_quantitative")),
        )
    except MonitorWithoutConditions:
        # 조건 없는 '관찰'은 과소평가로 채점된다 (D-39 · 04 §4.1.0).
        log.warning("MONITOR without escalation conditions — 거절로 전환")
        return {  # type: ignore[typeddict-item]
            "status": "refused",
            "refusal_reason": "판정불가",
        }
    except ValueError:
        # rule·llm 둘 다 None (판정 근거 없음).
        return {  # type: ignore[typeddict-item]
            "status": "refused",
            "refusal_reason": "판정불가",
        }

    return {  # type: ignore[typeddict-item]
        "triage_level": int(decision.level),
        "llm_capped": decision.llm_capped,
        "rule_level": int(decision.rule_level) if decision.rule_level is not None else None,
        "llm_level": int(decision.llm_level) if decision.llm_level is not None else None,
        "escalation_conditions": list(decision.escalation_conditions),
    }
