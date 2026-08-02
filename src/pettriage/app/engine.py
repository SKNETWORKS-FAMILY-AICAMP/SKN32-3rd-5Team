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

from ..compute.rules import rule_level_for
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
#:
#: ⚠️ `toxin_content_mg_per_g` — **역치의 기준이 음식이 아니라 독소인 물질**에만 있다.
#:
#: 규칙 테이블 12행을 보면 기준이 두 종류다.
#:
#:     양파 15-30 g/kg · 마늘 5 g/kg · 자일리톨 0.03 g/kg · 알리움 0.5%
#:         → **먹은 음식의 질량**이 곧 역치의 단위다. 그대로 쓴다
#:     초콜릿(테오브로민+카페인) 20 mg/kg
#:         → **테오브로민의 질량**이다. 초콜릿 질량이 아니다
#:
#: 이 둘을 구분하지 않고 `amount_g` 를 그대로 mg/kg 으로 바꾸면
#: **초콜릿을 100% 테오브로민으로 계산하게 된다** — 약 1만 배 과대평가다.
#: 안전한 쪽 오류라 눈에 안 띄지만, D-51 이 막은 것과 정확히 같은 종류의 조작이다
#: (*"잎 무게는 독소 무게가 아니다"*). 2026-08-02 배선 중 발견.
#:
#: 함량 값은 **원문 그대로다** (S-034 · F-034-023~025). 우리가 만든 값이 아니다.
#: `>14` 는 `parse_low` 와 같은 규율로 낮은 쪽인 14 로 읽는다.
#: 화이트 초콜릿은 원문에 수치가 없어 **목록에 넣지 않는다** — 정성 답변으로 내려간다.
_STUB_RULES: dict[str, dict] = {
    "초콜릿": {
        "species": {"dog", "cat"},
        "needs_dose": True,
        "level": TriageLevel.CALL_NOW,
        "escalation": ["구토", "심박 증가", "발작"],
        "toxin_content_mg_per_g": {
            "베이킹": 14.0,  # F-034-023 >14 mg/g
            "코코아": 14.0,  # F-034-023
            "다크": 5.0,  # F-034-024 세미스위트 다크 5 mg/g
            "세미스위트": 5.0,  # F-034-024
            "밀크": 2.0,  # F-034-025 2 mg/g
        },
        "content_question": (
            "어떤 초콜릿인가요? (다크 / 밀크 / 베이킹·코코아) — "
            "종류에 따라 테오브로민 함량이 2~14배 다릅니다."
        ),
        "citation": {
            "source_id": "S-034",
            "publisher": "Veterinary Sciences",
            "title": "Common toxicologic emergencies in companion animals",
            "locator": "Table 1",
            "route": "사실추출",
        },
    },
    # 역치가 **음식 질량** 기준인 물질 — 함량 환산이 필요 없다.
    # 정량 경로의 두 갈래를 모두 시연·평가에서 밟게 하려고 둔다.
    "양파": {
        "species": {"dog", "cat"},
        "needs_dose": True,
        "level": TriageLevel.CALL_NOW,
        "escalation": ["구토", "설사", "무기력", "호흡곤란", "잇몸 창백"],
        "citation": {
            "source_id": "S-034",
            "publisher": "Veterinary Sciences",
            "title": "Common toxicologic emergencies in companion animals",
            "locator": "Table 1",
            "route": "사실추출",
        },
    },
    "자일리톨": {
        "species": {"dog"},
        "needs_dose": True,
        "level": TriageLevel.CALL_NOW,
        "escalation": ["구토", "무기력", "운동실조", "발작"],
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

            # ③-b 역치가 **독소 기준**인 물질은 종류를 알아야 환산된다.
            #     초콜릿 20 mg/kg 은 테오브로민 기준이고, 함량은 종류마다 2~14 mg/g 로
            #     **7배** 다르다. 종류를 모르면 어떤 값을 골라도 그것은 우리가 만든 숫자다.
            #     골든셋 G-047 이 기대하는 되묻기가 이것이다.
            if rule.get("toxin_content_mg_per_g") and self._content_of(req, session, rule) is None:
                return clarify(
                    session,
                    ["substance_detail"],
                    rule["content_question"],
                    max_turns=max_turns,
                )

        # ④ 규칙 판정 → 하향 금지 게이트 (05 §5)
        #    스텁이므로 llm_level 은 None. 실제 엔진은 규칙 미적중 시 LLM을 부른다.
        rule_level, escalation, verdict_note = self._quantify(hit_key, req, session, rule)
        try:
            decision = apply_gate(
                rule_level=rule_level,
                llm_level=None,
                escalation_conditions=escalation,
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
            answer=self._compose(hit_key, session.species, decision, verdict_note),
            triage=to_triage_result(decision),
            citations=[Citation(**rule["citation"])],
        )

    @staticmethod
    def _content_of(req: AskRequest, session: Session, rule: dict) -> tuple[str, float] | None:
        """질문에서 **독소 함량**을 고른다. 못 고르면 `None` — 지어내지 않는다.

        누적된 되묻기 답변까지 함께 본다. *"다크초콜릿이요"* 는 두 번째 턴에 온다.
        """
        table: dict[str, float] = rule["toxin_content_mg_per_g"]
        text = " ".join(filter(None, [req.question, *session.question_history]))
        for kw, mg in table.items():
            if kw in text:
                return kw, mg
        return None

    def _quantify(
        self, substance: str, req: AskRequest, session: Session, rule: dict
    ) -> tuple[TriageLevel | None, tuple[str, ...], str]:
        """되물어 받은 **체중과 섭취량으로 실제 규칙 테이블을 조회한다** (D-50).

        예전에는 이 함수가 없었고 `rule["level"]` 상수를 그대로 게이트에 넣었다.
        그래서 4kg 개가 초콜릿을 `0.1g` 먹든 `500g` 먹든 답이 `CALL_NOW` 로 같았다 —
        **되묻기로 받은 값을 쓰지 않으면서** *"체중당 섭취량으로 판단합니다"* 라고
        말하고 있었던 것이다 (2026-08-02 검토).

        같은 검토에서 드러난 더 큰 문제는 `compute.rules` 쪽이었다.
        `rule_level_for`·`computable_for`·`to_mg_per_kg` 의 **실행 가능한 호출자가
        저장소에 하나도 없었다** — 이 프로젝트가 가장 공들인 계산 모듈이
        한 번도 실전 경로를 밟아 본 적이 없었다. 여기서 부르면서 검증된다.

        규칙 테이블이 등급을 못 내면(조류·역치 없음·출처 상충) 상수로 되돌아간다.
        **정량이 안 된다고 답을 포기하지 않는다** — 정성 판정은 여전히 유효하다 (D-46).
        """
        fallback = (rule["level"], tuple(rule["escalation"]), "")
        if session.weight_kg is None or session.amount_g is None:
            return fallback

        # 역치가 독소 기준이면 함량으로 환산한다. 음식 기준이면 질량 그대로다.
        # (`_STUB_RULES` 의 `toxin_content_mg_per_g` 주석 참조)
        detail = ""
        if rule.get("toxin_content_mg_per_g"):
            picked = self._content_of(req, session, rule)
            if picked is None:
                return fallback  # ③-b 에서 이미 되물었다. 여기 오면 종류를 끝내 못 정한 것이다
            kind, mg_per_g = picked
            toxin_mg = session.amount_g * mg_per_g
            detail = f"{kind} {mg_per_g:g}mg/g 기준"
        else:
            toxin_mg = session.amount_g * 1000.0  # g → mg. 역치 단위가 곧 음식 질량이다

        mg_per_kg = toxin_mg / session.weight_kg
        verdict = rule_level_for(substance, session.species or "", mg_per_kg)
        if verdict.level is None:
            log.info(
                "정량 판정 불가 — %s·%s: %s (정성 등급으로 되돌아간다)",
                substance,
                session.species,
                verdict.reason,
            )
            return fallback

        note = (
            f"체중 {session.weight_kg:g}kg · 섭취 {session.amount_g:g}g"
            + (f" · {detail}" if detail else "")
            + f" → 약 {mg_per_kg:,.1f} mg/kg"
        )
        return verdict.level, verdict.escalation_conditions or tuple(rule["escalation"]), note

    @staticmethod
    def _compose(substance: str, species: str, decision: TriageDecision, note: str = "") -> str:
        """응답 문장 조립 — 코드가 한다 (05 §4).

        의학적 중증도 어휘를 쓰지 않는다. 행동만 지시한다 (D-11 · D-39).

        상승 조건은 여기 넣지 않는다 — `escalation_conditions` 필드로 나간다.
        화면이 없는 클라이언트는 `full_text` 를 쓰면 조건까지 붙은 문장을 받는다.

        `note` 는 **계산 근거**다. 판정에 쓴 수치를 보여주지 않으면
        보호자가 그 판정을 검산할 수 없다 (04 §8 재현성).
        """
        ko = {"dog": "개", "cat": "고양이", "bird": "앵무새"}[species]
        head = f"{ko}가 {substance}을(를) 섭취한 상황입니다."
        body = f" {note}." if note else ""
        return f"{head}{body} {decision.message}."
