"""API 계약 — 프론트(WS5)와 파이프라인(WS2)의 인터페이스.

설계 근거: docs/02_시스템-아키텍처.md §9 · §12 · docs/00 §9.3

    00 §9.3 "UI 스펙은 WS2와 사전 합의하여 인터페이스 재작업을 방지한다"의
    **합의문이 이 파일이다.** 화면이 아니라 계약을 먼저 고정한다.

이 파일의 핵심은 **02 §9 거절·되묻기 정책을 타입으로 강제**하는 것이다.

    감점 포인트 (00 §7): "검색 실패와 생성 실패를 구분하지 않은 오류 분석"
                         "고지 문구 없이 단정적으로 답하는 응답 화면"

    → `status` 를 3값으로 나누고, 각 상태가 요구하는 필드를 검증기로 묶었다.
      `answered` 인데 근거가 비어 있으면 **응답을 만들 수 없다** — 런타임에 터진다.
      환각 방지를 문구가 아니라 스키마로 표현한 것이다.
"""

from __future__ import annotations

from datetime import datetime
from typing import Annotated, Any, Literal, get_args

from pydantic import (
    AfterValidator,
    BaseModel,
    ConfigDict,
    EmailStr,
    Field,
    computed_field,
    model_validator,
)

from ..compute.vocabulary import SPECIES as _SPECIES
from ..compute.vocabulary import check_substance
from ..safety import has_contact
from ..triage.levels import TriageLevel

#: **폐쇄 목록 안의 물질명만** 담을 수 있는 문자열 (D-59 ① · D-40).
#:
#: D-59 ①은 *"물질 동정은 생성이 아니라 533종 폐쇄 목록에서의 선택"* 으로 정했는데,
#: 그때 만든 것은 **프롬프트가 그렇게 부탁하는 것**까지였다. 부탁은 어길 수 있다.
#: 모델이 `일산화탄소`(코퍼스에 없다) 라고 답해도 막는 것이 없었다.
#:
#: 이 타입을 쓰는 필드는 **목록 밖 이름으로는 값이 만들어지지 않는다.**
#: `'없음'` 은 통과한다 — 고를 것이 없다는 것도 정상 선택지다 (D-59 ④).
SubstanceName = Annotated[str, AfterValidator(check_substance)]

#: 02 §9 — 모든 응답에 노출한다. 응답 모델의 기본값이므로 누락이 불가능하다.
DISCLAIMER = "본 안내는 참고용이며 수의학적 진단이 아닙니다. 이상이 의심되면 수의사와 상담하세요."

#: 종. **정의는 `compute.vocabulary.SPECIES` 하나뿐이다** (P2 · D-22).
#:
#: 예전에는 이 파일과 `compute/aliases.py` 와 `tests/` 가 각자 사본을 들고 있었다.
#: `Literal` 은 상수를 못 받으므로 값은 여기 적되, **다르면 임포트 시점에 터진다** —
#: 두 곳이 조용히 어긋나는 것보다 낫다.
Species = Literal["dog", "cat", "bird"]
assert set(get_args(Species)) == set(_SPECIES), (
    f"종 정의가 어긋났다 — contracts={get_args(Species)} vocabulary={_SPECIES}. "
    "단일 출처는 compute.vocabulary.SPECIES 다."
)

#: 되묻기 상한 (02 §9 · 05 §4). 초과하면 거절로 넘어간다.
MAX_CLARIFY_TURNS = 2


# ─────────────────────────────────────────────────────────────
# 요청
# ─────────────────────────────────────────────────────────────
class AskRequest(BaseModel):
    """질의 1건.

    보호자 개인정보는 받지 않는다 (D-36). `pet_id` 는 가명 식별자다.
    """

    question: str = Field(min_length=1, max_length=2000)
    species: Species | None = None
    pet_id: str | None = Field(default=None, max_length=64)
    weight_kg: float | None = Field(default=None, gt=0, le=200)
    amount_g: float | None = Field(default=None, ge=0)
    #: 되묻기를 이어갈 때만 채운다. 없으면 새 세션이 열린다.
    session_id: str | None = None


class RecordCreate(BaseModel):
    """다이어리 기록 1건 (02 §12 기록 입력 화면).

    이 기록은 "장기 기억"이 아니라 **검색 대상 문서**다 (05 §3).
    """

    pet_id: str = Field(min_length=1, max_length=64)
    species: Species
    #: ISO 8601. **검증하고 정규화한다.**
    #:
    #: 예전에는 그냥 `str` 이었고 주석에만 "ISO 8601" 이라고 적혀 있었다.
    #: `RecordStore.timeline` 은 **문자열 비교**로 기간을 자르므로,
    #: `2026-7-3` 처럼 0을 안 채운 값이 들어오면 `'2026-7-3' > '2026-07-31'` 이 참이 되어
    #: **7월 리포트에서 7월 기록이 사라진다** (2026-08-02 재현). `어제`·`` 도 통과했다.
    #: 건강 다이어리에서 조용한 누락은 오답과 같다.
    recorded_at: str
    note: str = Field(default="", max_length=4000)
    meals: list[str] = Field(default_factory=list)
    symptoms: list[str] = Field(default_factory=list)
    #: 조류 전용 — 배설물 상태 (02 §12). 종이 bird가 아니면 무시된다.
    droppings: str | None = None

    @model_validator(mode="after")
    def _normalize_recorded_at(self) -> RecordCreate:
        """ISO 8601 로 파싱해 **정규화된 문자열로 되돌린다.**

        정규화까지 하는 이유 — 기간 필터가 문자열 비교라서, 파싱만 통과시키고
        원문을 그대로 두면 `2026-7-3` 이 여전히 잘못 정렬된다.
        같은 시각을 가리키는 표기가 여럿이면 **비교가 성립하지 않는다.**
        """
        raw = (self.recorded_at or "").strip()
        try:
            dt = datetime.fromisoformat(raw.replace("Z", "+00:00"))
        except ValueError as e:
            raise ValueError(
                f"recorded_at 이 ISO 8601 이 아니다: {raw!r} "
                "(예: 2026-07-02 · 2026-07-02T09:00 · 2026-07-02T09:00:00+09:00)"
            ) from e
        object.__setattr__(self, "recorded_at", dt.isoformat())
        return self


# ─────────────────────────────────────────────────────────────
# 응답 구성요소
# ─────────────────────────────────────────────────────────────
class _Strict(BaseModel):
    """대입에도 검증이 도는 기반 모델.

    검증기를 생성 시점에만 걸면 ``obj.status = "answered"`` 한 줄로 불변식이 뚫린다.
    안전 불변식을 타입으로 강제하기로 한 이상(D-40) 대입도 막아야 의미가 있다.
    """

    model_config = ConfigDict(validate_assignment=True)


class Citation(_Strict):
    """근거 1건 (02 §12 '근거 보기' 화면).

    `quote` 가 None인 것은 결함이 아니다 — 경로 ②(사실 추출)로 적재된 자료는
    원문을 벡터DB에 담지 않으므로 인용문이 존재하지 않는다 (D-37 · 02 §14).
    프론트는 `route` 를 보고 인용문 대신 출처·위치를 보여준다.
    """

    source_id: str
    publisher: str
    title: str | None = None
    locator: str | None = None
    url: str | None = None
    route: Literal["원문적재", "사실추출"] = "사실추출"
    quote: str | None = None

    @model_validator(mode="after")
    def _quote_only_for_route1(self) -> Citation:
        if self.route == "사실추출" and self.quote:
            raise ValueError(
                f"{self.source_id}: 경로 ②(사실추출) 자료에 원문 인용을 실을 수 없다 (D-37)."
            )
        return self


class TriageResult(_Strict):
    """트리아지 배지 + 감사 정보 (02 §7.2 · D-09 · D-39).

    `rule_level`·`llm_level`·`overridden` 을 응답에 실어 보낸다.
    화면에 안 띄우더라도 **하향 금지 게이트가 실제로 작동했다는 증거**가
    로그와 데모에 남는다 (산출물 ④).
    """

    level: int = Field(ge=1, le=4)

    #: `level` 에서 파생된다. 안 주면 자동으로 채워지고, 주면 대조 후 불일치 시 거부한다.
    #:
    #: 예전에는 둘 다 필수 자유 문자열이었다. 그래서
    #: `TriageResult(level=4, name="MONITOR", badge="관찰")` 이 그대로 통과했다 —
    #: **응답이 스스로 모순인 채로 화면에 나갈 수 있었다** (2026-08-02 검토).
    #: `TriageLevel.badge` 라는 단일 출처가 이미 있는데 계약이 다시 받고 있었다.
    name: str | None = None
    badge: str | None = None

    #: 자유 문자열로 남긴다 — 등급 이름·배지는 UI 계약이지만,
    #: 행동 문장은 상황에 따라 구체화되어야 한다 (예: "3시간 이내에 …").
    message: str
    escalation_conditions: list[str] = Field(default_factory=list)

    rule_level: int | None = None
    llm_level: int | None = None
    overridden: bool = False

    @model_validator(mode="before")
    @classmethod
    def _derive_name_badge(cls, data: Any) -> Any:
        """`level` 에서 이름·배지를 채운다. 이미 있으면 건드리지 않는다 (대조는 아래에서)."""
        if not isinstance(data, dict):
            return data
        raw = data.get("level")
        try:
            lv = TriageLevel(int(raw))
        except (TypeError, ValueError):
            return data  # 범위 밖이면 Field(ge=1, le=4) 가 잡는다
        data.setdefault("name", lv.name)
        data.setdefault("badge", lv.badge)
        return data

    @model_validator(mode="after")
    def _monitor_needs_conditions(self) -> TriageResult:
        if int(self.level) == int(TriageLevel.MONITOR) and not self.escalation_conditions:
            raise ValueError("MONITOR는 상승 조건 없이 응답에 실을 수 없다 (D-39).")
        return self

    @model_validator(mode="after")
    def _level_is_final(self) -> TriageResult:
        """**하향 금지 게이트의 결과를 덮을 수 없다** (D-09).

        `apply_gate()` 안의 `max()` 산술은 옳다. 문제는 그 뒤였다 —
        게이트를 통과한 값을 이 객체에 담을 때 아무도 대조하지 않아서

            TriageResult(level=1, rule_level=4, llm_level=1)   # 통과했다

        가 성립했다. D-09 를 우회하는 길은 *게이트를 안 부르는 것*이 아니라
        **부른 뒤 결과를 덮는 것**이었고, 계약이 그 문을 열어 두고 있었다.
        """
        floor = max(self.rule_level or 0, self.llm_level or 0)
        if self.level < floor:
            raise ValueError(
                f"level={self.level} 이 게이트 바닥 {floor} 보다 낮다 — "
                f"하향 금지 게이트의 결과를 덮을 수 없다 (D-09). "
                f"rule_level={self.rule_level} llm_level={self.llm_level}"
            )
        lv = TriageLevel(int(self.level))
        if self.name != lv.name or self.badge != lv.badge:
            raise ValueError(
                f"등급 표기가 level 과 어긋난다 — level={self.level}({lv.name}/{lv.badge}) "
                f"인데 name={self.name!r} badge={self.badge!r} 이다."
            )
        # `overridden` 은 gate.py 의 정의(`llm < rule`)와 같은 뜻이어야 한다.
        # 어긋나면 감사 정보가 거짓이 된다 — 산출물 ④가 이 값을 근거로 쓴다.
        expected = (
            self.rule_level is not None
            and self.llm_level is not None
            and self.llm_level < self.rule_level
        )
        if self.overridden != expected:
            raise ValueError(
                f"overridden={self.overridden} 이 rule_level={self.rule_level} · "
                f"llm_level={self.llm_level} 과 맞지 않는다 (gate.py 정의: llm < rule)."
            )
        return self


class ClarifyPrompt(_Strict):
    """되묻기 (02 §9). 무엇이 없어서 묻는지를 기계가 읽을 수 있게 남긴다."""

    missing: list[str]  # 예: ["species", "weight_kg"]
    question: str
    turn: int = Field(ge=1)
    max_turns: int = MAX_CLARIFY_TURNS


RefusalReason = Literal[
    "근거없음",  # 검색 결과 없음 / 유사도 임계 미만
    "검증실패",  # 근거 검증 실패 후 재검색도 실패
    "되묻기상한",  # 슬롯을 끝내 못 채움
    "판정불가",  # 규칙·LLM 판정이 모두 없음
    "범위밖",  # 도메인 밖 질문
]


class Refusal(_Strict):
    """거절 (02 §9).

    거절은 실패가 아니라 **설계된 경로**다. 검색 실패와 생성 실패를
    `reason` 으로 구분해 오류 분석에 그대로 쓴다 (00 §7).
    """

    reason: RefusalReason
    message: str
    advice: str = "수의사와 상담하시기 바랍니다."


# ─────────────────────────────────────────────────────────────
# 응답
# ─────────────────────────────────────────────────────────────
class AskResponse(_Strict):
    """질의 응답 1건. `status` 가 화면 분기를 결정한다."""

    status: Literal["answered", "clarify", "refused"]
    session_id: str

    answer: str | None = None
    triage: TriageResult | None = None
    citations: list[Citation] = Field(default_factory=list)
    clarify: ClarifyPrompt | None = None
    refusal: Refusal | None = None

    #: **확인받지 못한 추정 물질** (D-59). 사용자가 물질을 말하지 않았고 되묻기로도
    #: 확정하지 못했을 때, 후보 중 최고 등급으로 답하면서 그 가정을 여기 남긴다.
    #:
    #: 값을 넣으면 `_assumption_must_be_stated` 가 **문장에 그 가정이 실렸는지 확인한다.**
    #: 안 실렸으면 응답을 만들 수 없다 — **밝히지 않은 추정은 곧 환각이다.**
    #:
    #: 타입이 `SubstanceName` 이므로 **폐쇄 목록 밖 이름은 애초에 못 들어온다** (D-59 ①).
    #: 가정을 밝히는 것과 **무엇을 가정할 수 있는지**는 다른 문제이고, 둘 다 계약이 본다.
    assumed_substance: SubstanceName | None = None

    #: **이 답이 무엇에 관한 것인가.** 동정된 물질을 남긴다 (D-59 ①).
    #:
    #: 선택 필드다. 계약이 강제하는 것은 *"넣는다면 반드시 폐쇄 목록 안"* 이고,
    #: *"반드시 넣어야 한다"* 는 아니다 — ② 노드가 아직 없어서 강제할 대상이 없다.
    #: 노드가 들어오면 `graph.state.set_substance` 를 유일한 문으로 두고 거기서 올린다.
    #:
    #: **못 하는 것을 한다고 적지 않는다** (D-58).
    identified_substance: SubstanceName | None = None

    #: 02 §9 — 상태와 무관하게 항상 나간다.
    disclaimer: str = DISCLAIMER

    @computed_field  # type: ignore[prop-decorator]
    @property
    def full_text(self) -> str:
        """화면이 없는 클라이언트(챗봇·음성)를 위한 완성 문장.

        `answer` 만 읽는 소비자가 **상승 조건을 통째로 빠뜨리는 사고**를 막는다.
        조건 누락은 이 도메인에서 과소평가와 같다 (D-39 · 04 §4.1.0).
        """
        parts: list[str] = []
        if self.answer:
            parts.append(self.answer)
        elif self.clarify:
            parts.append(self.clarify.question)
        elif self.refusal:
            parts.append(f"{self.refusal.message} {self.refusal.advice}")
        if self.triage and self.triage.escalation_conditions:
            joined = ", ".join(self.triage.escalation_conditions)
            parts.append(f"다음 증상이 나타나면 즉시 알리세요 — {joined}.")
        parts.append(self.disclaimer)
        return " ".join(parts)

    @model_validator(mode="after")
    def _status_invariants(self) -> AskResponse:
        """상태와 필드가 서로 배타적인지 확인한다.

        예전에는 **있어야 할 것**만 봤다. 그래서 *없어야 할 것*이 함께 실린 조합이
        전부 통과했다 (2026-08-02 검토) —

            refused + triage      → full_text 가 거절 문구 뒤에 상승 조건을 이어 붙인다
            answered + refusal    → 화면 분기와 응답 내용이 어긋난다
            answered + answer="  " → 공백만 있는 "답변"이 근거와 배지를 달고 나간다

        `answered` 의 근거·판정 검사는 이 프로젝트의 존재 이유다. 나머지도 같은 급으로 본다.
        """
        if self.status == "answered":
            if not (self.answer or "").strip():
                raise ValueError("answered 인데 answer 가 비었다 (공백만 있는 것도 빈 것이다).")
            if not self.citations:
                # 이 프로젝트의 존재 이유. 근거 없는 답은 응답 객체조차 못 만든다.
                raise ValueError("answered 인데 근거가 없다 — 거절 경로로 보내야 한다 (02 §9).")
            if self.triage is None:
                raise ValueError("answered 인데 트리아지 판정이 없다.")
            if self.refusal is not None:
                raise ValueError("answered 와 refusal 을 함께 실을 수 없다 (02 §9).")
            if self.clarify is not None:
                raise ValueError("answered 와 되묻기를 함께 실을 수 없다 (02 §9).")
        elif self.status == "clarify":
            if self.clarify is None:
                raise ValueError("clarify 인데 되묻기 내용이 없다.")
            if self.answer:
                raise ValueError("되묻는 중에는 답변을 함께 내지 않는다 (02 §9).")
            if self.triage is not None:
                raise ValueError("되묻는 중에는 트리아지 배지를 내지 않는다 (02 §9).")
            if self.refusal is not None:
                raise ValueError("clarify 와 refusal 을 함께 실을 수 없다.")
            if self.citations:
                raise ValueError("되묻는 중에는 근거를 내보내지 않는다 (02 §9).")
        elif self.status == "refused":
            if self.refusal is None:
                raise ValueError("refused 인데 거절 사유가 없다.")
            if self.answer:
                raise ValueError("거절하면서 답변을 내보낼 수 없다.")
            if self.triage is not None:
                # full_text 가 거절 문구 뒤에 상승 조건을 붙여 문장이 스스로 모순된다.
                raise ValueError("거절하면서 트리아지 배지를 내보낼 수 없다 (02 §9).")
            if self.clarify is not None:
                raise ValueError("refused 와 되묻기를 함께 실을 수 없다.")
        return self

    @model_validator(mode="after")
    def _assumption_must_be_stated(self) -> AskResponse:
        """추정 물질로 답하면서 **그 가정을 숨길 수 없다** (D-59).

        물질을 말하지 않는 질의(*"앵무새 앞에서 프라이팬을 태웠어요"*)에
        후보 중 최고 등급으로 답하는 것은 D-13(과소평가 최우선)에 따른 선택이다.
        그 선택이 정직하려면 **무엇을 가정했는지가 문장에 있어야** 한다.

        이것을 문장 생성에 맡기면 안 된다 — LLM 이 한 줄을 빠뜨리는 순간
        추측이 단정으로 나가고, **그것이 곧 환각이다.** 그래서 계약이 강제한다
        (D-54 와 같은 방식: *지키기로 한 것이 아니라 못 어기는 것*).

        `full_text` 를 보는 이유는 `_no_foreign_contacts` 와 같다 — 화면 없는
        클라이언트가 읽는 것이 그것이고, 가정은 거기에 실려야 의미가 있다.
        """
        if self.assumed_substance and self.assumed_substance not in self.full_text:
            raise ValueError(
                f"추정 물질 {self.assumed_substance!r} 로 답하면서 그 가정을 문장에 밝히지 않았다 "
                "(D-59). 밝히지 않은 추정은 환각이다."
            )
        return self

    @model_validator(mode="after")
    def _no_foreign_contacts(self) -> AskResponse:
        """**최종 안전망** — 국내에서 쓸 수 없는 연락처가 남아 있으면 응답을 못 만든다 (D-47).

        스크럽은 `SafetyEngine` 이 한다. 여기는 그것이 **실제로 돌았는지** 확인하는 자리다.
        정상 경로에서는 절대 걸리지 않는다. 걸렸다면 래퍼 밖에서 응답이 만들어졌다는 뜻이고,
        그때는 `main.py::_install_response_guard` 가 200 + `refused` 로 내리면서
        로그에 ERROR 를 남긴다 — **조용히 나가지 않는다.**

        `answer` 가 아니라 `full_text` 를 보는 이유 —
        `full_text` 는 `escalation_conditions` 를 **뒤에 이어 붙인다.** `answer` 만 검사하면
        스크럽 뒤에 붙는 조건에서 번호가 되살아난다 (2026-08-02 검토에서 실측).
        """
        if has_contact(self.full_text):
            raise ValueError(
                "응답에 국내에서 쓸 수 없는 연락처가 남아 있다 (D-47). "
                "SafetyEngine 을 거치지 않은 경로가 있는지 확인할 것."
            )
        return self


class RecordCreated(BaseModel):
    record_id: str
    pet_id: str
    indexed: bool = False  # 벡터DB 적재 여부 (WS1 연결 후 True)


class ReportResponse(BaseModel):
    """기간 리포트 (02 §12 · 구현 3단계)."""

    pet_id: str
    period_from: str
    period_to: str
    timeline: list[dict] = Field(default_factory=list)
    summary: str = ""
    disclaimer: str = DISCLAIMER


class HealthResponse(BaseModel):
    """기동 상태.

    `engine` != `engine_configured` 이면 **폴백이 일어난 것**이다.
    그 상태로 산출한 평가 지표는 오염이므로 즉시 드러나야 한다 (04 §8).
    """

    status: Literal["ok"]
    engine: str  # 실제로 물려 있는 엔진
    engine_configured: str  # configs 가 요구한 엔진
    profile: str  # PETTRIAGE_PROFILE
    version: str

    #: 임베딩 모델이 메모리에 올라와 있는가 (D-53).
    #: `None` 은 **해당 없음** — `engine=stub` 은 벡터 검색을 하지 않는다.
    #: `False` 면 첫 질의가 로딩(수 초~수십 초)을 맞는다.
    #: **시연 전에 이 값을 확인한다** — 스트리밍이 없어 그 시간이 침묵으로 나타난다 (02 §12.4).
    model_loaded: bool | None = None

    @property
    def degraded(self) -> bool:
        return self.engine != self.engine_configured

    @property
    def cold(self) -> bool:
        """워밍업이 안 된 상태. 첫 질의가 느리다."""
        return self.model_loaded is False


# ─────────────────────────────────────────────────────────────
# 계정 · 반려동물 프로필 (WS5 백엔드)
#
# 라우터 파일 안에 스키마를 두지 않는다 — **계약은 여기 하나다** (D-40 · D-22).
# 처음 들어올 때는 `routes/auth.py`·`routes/pets.py` 가 각자 6종을 들고 있었고,
# `Species` 도 `Literal["dog","cat","bird"]` 로 다시 적혀 있었다 (2026-08-01 흡수).
# ─────────────────────────────────────────────────────────────
#: bcrypt 가 조용히 버리는 경계. **`app.auth` 에서 가져온다** — 두 곳에 적으면 어긋난다.
#: `auth` 는 `bcrypt` 를 임포트하므로 여기서 최상단 임포트를 하지 않는다
#: (`[api]` extra 없이도 계약은 읽혀야 한다).
def _bcrypt_max_bytes() -> int:
    try:
        from .auth import BCRYPT_MAX_BYTES as _n
    except Exception:  # noqa: BLE001 — bcrypt 미설치 구성
        return 72
    return _n


def _check_password_bytes(value: str) -> str:
    """**글자 수가 아니라 바이트로** 잰다.

    상한이 `max_length=64` (글자)로만 걸려 있었다. 그런데 bcrypt 의 한계는
    **72바이트**이고 한글은 3바이트/자다. 그래서 `가`×25 = 25자 / 75바이트가
    글자 검사를 통과해 `auth.hash_password` 까지 내려갔고,
    `PasswordTooLongError` 가 잡히지 않아 **500** 이 나갔다 (2026-08-02 재현).

    같은 파일 docstring 이 *"한글은 3바이트/자라 25자부터"* 라고 이미 적고 있었다 —
    **원인을 알면서 단위를 잘못 골랐다.** 여기서 바이트로 재면 422 로 안내된다.
    """
    limit = _bcrypt_max_bytes()
    n = len(value.encode("utf-8"))
    if n > limit:
        raise ValueError(
            f"비밀번호가 {n}바이트입니다. {limit}바이트까지 쓸 수 있습니다 "
            f"(영문·숫자는 1바이트, 한글은 3바이트 — 한글만 쓰면 24자까지)."
        )
    return value


class SignupRequest(BaseModel):
    """회원가입.

    비밀번호 상한은 **bcrypt 가 72바이트를 넘는 부분을 조용히 버리기** 때문에 있다.
    사용자가 긴 비밀번호를 썼다고 믿는 채 앞부분만 쓰이는 상황을 만들지 않는다.
    `max_length` 는 명백한 남용을 막는 1차 방어이고, **실제 경계는 바이트 검사**다.
    """

    email: EmailStr
    password: Annotated[
        str, Field(min_length=8, max_length=72), AfterValidator(_check_password_bytes)
    ]
    nickname: str = Field(min_length=1, max_length=50)


class SignupResponse(BaseModel):
    user_id: str
    nickname: str
    message: str = "회원가입이 완료되었습니다."


class LoginRequest(BaseModel):
    email: EmailStr
    #: 로그인에도 같은 바이트 검사를 건다 — 없으면 **로그인만 500** 이 난다.
    #: 가입은 막고 로그인은 안 막으면 그 자체가 계정 존재 여부의 신호가 된다.
    password: Annotated[
        str, Field(min_length=1, max_length=72), AfterValidator(_check_password_bytes)
    ]


class LoginResponse(BaseModel):
    access_token: str
    token_type: Literal["bearer"] = "bearer"
    nickname: str


class PetCreate(BaseModel):
    """반려동물 등록.

    **동물등록번호를 받지 않는다** (D-36 조치 1) — 등록번호에는 소유자
    성명·주민등록번호·주소·전화번호가 함께 묶여 있다. 식별은 앱 내부 UUID 로 한다.

    `weight_kg` 단위는 D-17 개정본을 따른다 — 체중은 kg, 섭취량은 g.
    """

    name: str = Field(min_length=1, max_length=50)
    species: Species
    breed: str | None = Field(default=None, max_length=50)
    weight_kg: float | None = Field(default=None, gt=0, le=200)


class PetResponse(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    pet_id: str
    name: str
    species: str
    breed: str | None = None
    weight_kg: float | None = None
    created_at: datetime
