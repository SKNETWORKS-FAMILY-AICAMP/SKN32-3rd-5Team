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

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, computed_field, model_validator

from ..triage.levels import TriageLevel

#: 02 §9 — 모든 응답에 노출한다. 응답 모델의 기본값이므로 누락이 불가능하다.
DISCLAIMER = "본 안내는 참고용이며 수의학적 진단이 아닙니다. 이상이 의심되면 수의사와 상담하세요."

Species = Literal["dog", "cat", "bird"]

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

    pet_id: str = Field(max_length=64)
    species: Species
    recorded_at: str  # ISO 8601
    note: str = Field(default="", max_length=4000)
    meals: list[str] = Field(default_factory=list)
    symptoms: list[str] = Field(default_factory=list)
    #: 조류 전용 — 배설물 상태 (02 §12). 종이 bird가 아니면 무시된다.
    droppings: str | None = None


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
    name: str
    badge: str
    message: str
    escalation_conditions: list[str] = Field(default_factory=list)

    rule_level: int | None = None
    llm_level: int | None = None
    overridden: bool = False

    @model_validator(mode="after")
    def _monitor_needs_conditions(self) -> TriageResult:
        if int(self.level) == int(TriageLevel.MONITOR) and not self.escalation_conditions:
            raise ValueError("MONITOR는 상승 조건 없이 응답에 실을 수 없다 (D-39).")
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
        if self.status == "answered":
            if not self.answer:
                raise ValueError("answered 인데 answer 가 비었다.")
            if not self.citations:
                # 이 프로젝트의 존재 이유. 근거 없는 답은 응답 객체조차 못 만든다.
                raise ValueError("answered 인데 근거가 없다 — 거절 경로로 보내야 한다 (02 §9).")
            if self.triage is None:
                raise ValueError("answered 인데 트리아지 판정이 없다.")
        elif self.status == "clarify":
            if self.clarify is None:
                raise ValueError("clarify 인데 되묻기 내용이 없다.")
            if self.answer:
                raise ValueError("되묻는 중에는 답변을 함께 내지 않는다 (02 §9).")
        elif self.status == "refused":
            if self.refusal is None:
                raise ValueError("refused 인데 거절 사유가 없다.")
            if self.answer:
                raise ValueError("거절하면서 답변을 내보낼 수 없다.")
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

    @property
    def degraded(self) -> bool:
        return self.engine != self.engine_configured
