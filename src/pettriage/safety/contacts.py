"""연락처 차단 — **외국 핫라인 번호가 답변에 나가지 않게 한다.**

설계 근거: docs/06_설계결정기록.md · D-47 (D-38 · D-40 의 연장)

문제
----
코퍼스 45건 중 응급 지침 자료(ASPCA · FDA · Pet Poison Helpline · FOUR PAWS ·
Cornell · Banfield)는 **전부 미국 자료**이고, 하나같이 말미에 24/7 핫라인 번호를 단다.

    Pet Poison Helpline   855-764-7661   (S-027 · S-029 · S-085 · S-100)
    ASPCA APCC            888-426-4435   (S-007 · S-029 · S-100)
    FOUR PAWS 안내         855-289-0358 · 855-454-4130   (S-030)

검수에서 이 번호들이 **서로 상충하는 줄 알았으나 아니었다** — 기관이 다르거나
같은 기관을 다른 자료가 다르게 적은 것이다 (2026-08-01 검수). 진짜 문제는 따로 있다.

    **국내 사용자가 이 번호로 전화하면 아무 일도 일어나지 않는다.**
    응급 상황에서 잘못된 연락처는 오답보다 나쁘다 — 시간을 쓰게 만들기 때문이다.

왜 데이터가 아니라 여기서 막나
--------------------------
번호는 대부분 `note` 에 있고 문장화 템플릿이 `note` 를 쓰지 않으므로
**청크에는 거의 안 들어간다.** 그런데도 여기서 막는 이유는,

  1. LLM이 자기 사전지식으로 `888-426-4435` 를 뱉을 수 있다. 검색 근거에 없어도 나온다.
     ④ 검증이 `근거없음` 으로 잡아야 하지만, **판정에 기대는 것은 보장이 아니다.**
  2. 사실 표에서 행을 지우면 "무엇이 위험한가" 같은 나머지 내용까지 잃는다.

그래서 데이터는 그대로 두고 **출력 직전에 코드가 거른다.** D-40의 계층 분리와 같은 형태다.

방식 — 문장 단위 제거
-------------------
번호만 지우면 `"ASPCA APCC에 연락하세요"` 가 남는다. 국내 사용자에겐 여전히 오답이다.
그래서 **연락처가 든 문장을 통째로 뺀다.** ④ 검증이 이미 쓰는 조치와 같은 방식이다.

    >>> r = scrub_contacts("초콜릿은 개에게 독성이 있다. ASPCA APCC(888-426-4435)에 연락하세요.")
    >>> r.text
    '초콜릿은 개에게 독성이 있다. 연락처는 지역마다 달라 안내해 드리지 않습니다. 가까운 동물병원이나 24시 동물병원으로 바로 연락해 주세요.'
    >>> r.removed
    ['ASPCA APCC(888-426-4435)에 연락하세요.']

전부 지워지면 안내 문장만 남는다. **그래도 틀린 번호를 주는 것보다 낫다.**
"""  # noqa: E501

from __future__ import annotations

import re
from collections.abc import Sequence
from dataclasses import dataclass

#: 제거가 일어났을 때 대신 붙이는 문장. 국내 대체 연락처를 정하기 전까지의 기본값이다.
#:
#: 국내에 미국 APCC 에 대응하는 **공식 중독 상담 창구가 없다.** 있는 것처럼 적으면
#: 그게 곧 환각이므로, 특정 기관 대신 "가까운 동물병원" 으로 돌린다.
#:
#: **존댓말인 이유** — 이 문장은 청크가 아니라 **사용자에게 그대로 나가는 출력**이다.
#: 코퍼스 청크는 검색 대상이라 평서체(`~다`)로 두지만, 화면에 뜨는 문장은 존댓말로 쓴다.
#: 응급 상황에서 보호자가 읽는 말이므로 **명령조가 되면 안 된다** (D-47).
GUIDANCE = (
    "연락처는 지역마다 달라 안내해 드리지 않습니다. "
    "가까운 동물병원이나 24시 동물병원으로 바로 연락해 주세요."
)

#: 구분자 — 하이픈·en dash·em dash·점·공백. **LLM과 한글 편집기가 en dash 를 자주 낸다.**
_SEP = r"[\s.‐-―\-]"

#: 그 자체로 전화번호인 것. 행동 지시어 없이도 막는다.
#:
#: 미국 톨프리(`1-855-764-7661` · `(888) 426-4435` · `(888)426-4435` · `888.426.4435`)와
#: 국내 형식(`02-1234-5678` · `02)123-4567`)을 함께 잡는다.
#: **국내 번호도 막는 이유** — 지금 코퍼스에 검증된 국내 연락처가 한 건도 없다.
#: 나중에 검증된 창구가 생기면 그때 예외를 여기에 명시적으로 뚫는다.
_PHONE = re.compile(
    rf"""
    (?<![\d.])                          # 앞이 숫자·소수점이면 용량 수치다 (2.3 g/kg)
    (?:\+?\d{{1,3}}{_SEP})?             # 국가번호
    (?:\(\d{{2,4}}\)|\d{{2,4}}\)|\d{{2,4}}{_SEP})   # 지역·식별 번호. 괄호 뒤 구분자는 없어도 된다
    {_SEP}?\d{{3,4}}
    {_SEP}\d{{4}}
    (?![\d.])
    """,
    re.VERBOSE,
)

#: **애매한** 숫자 묶음. 연도·용량·범위 표기와 구별이 안 되므로
#: 같은 문장에 행동 지시어(`_ACTION`)가 있을 때만 연락처로 본다.
#:
#:     "8884264435 로 전화하세요"        → 차단 (전화)
#:     "2020-2024 자료를 종합하면"        → 통과
#:     "1588-1234"  단독                 → 통과
#:
#: 예전에는 이 형식을 아예 안 잡았고(`_PHONE` docstring 은 잡는다고 적혀 있었다),
#: 그래서 `8884264435 로 전화하세요.` 가 그대로 나갔다 — 2026-08-02 검토에서 실측.
_PHONE_AMBIGUOUS = re.compile(
    rf"""
    (?<![\d.])
    (?: \d{{10,11}}                     # 구분자 없이 붙여 쓴 10~11자리
      | \d{{4}}{_SEP}\d{{4}}            # 국내 대표번호 1588-1234
    )
    (?![\d.])
    """,
    re.VERBOSE,
)

#: 국내에서 걸 수 없는 해외 상담 창구. **번호를 지워도 기관명이 남으면 소용없다.**
_FOREIGN_ORGS = (
    "Pet Poison Helpline",
    "Animal Poison Control Center",
    "APCC",
    "ASPCA Poison",
    "Poison Control Center",
    "FOUR PAWS",
)

#: URL 은 그 자체가 행동 유도다 — 행동 지시어를 요구하지 않는다.
#: *"petpoisonhelpline.com 을 참고하세요"* 의 "참고"까지 `_ACTION` 에 넣기 시작하면
#: 목록이 끝없이 늘어난다. **주소가 보이면 거기로 가라는 뜻**이라고 보는 편이 짧다.
_FOREIGN_URLS = (
    "petpoisonhelpline",
    "aspca.org/pet-care/animal-poison-control",
    "aspcapro.org",
)

#: 기관명이 나왔더라도 **행동 지시**가 아니면 지우지 않는다.
#: *"ASPCA APCC 자료에 따르면 아보카도는 조류에게 치명적이다"* 는 살려야 한다.
_ACTION = (
    "연락", "전화", "문의", "상담", "신고", "요청", "도움", "방문", "접속", "이용",
    "call", "contact", "reach",
)  # fmt: skip

#: 문장 경계. 한국어 종결 `~다.` · 영문 마침표에 더해 **줄바꿈**도 경계로 본다.
#:
#: 줄바꿈을 넣지 않으면 LLM 이 흔히 내는 불릿 목록이 통째로 **한 문장**이 되어,
#: 마지막 줄에 번호가 하나 있으면 **앞의 용량 정보까지 전부 삭제된다.**
#: 2026-08-02 검토에서 실측 — 3줄짜리 답변이 안내 문장 한 줄로 바뀌었다.
_SENT = re.compile(r"(?<=[.!?。？！])\s+|\n+")

#: 우리가 붙이는 출처 표기. **검사 대상이 아니다.**
#:
#: `(출처: Pet Poison Helpline, S-084)` 는 자료의 출처를 밝히는 고정 형식이지
#: *"거기로 연락하라"* 가 아니다. 빼지 않으면 같은 문장에 "연락"이 있을 때
#: 문장 전체가 지워진다 — 실측: 888청크 중 **12건이 이 이유로 오탐**이었고,
#: 그중에는 에틸렌글리콜·발작·호흡곤란 응급 청크가 들어 있었다.
_CITATION = re.compile(r"\(출처:[^)]*\)")


def _looks_like_dose(text: str, span: tuple[int, int]) -> bool:
    """수치 뒤에 단위가 붙어 있으면 전화번호가 아니라 용량이다."""
    tail = text[span[1] : span[1] + 6]
    return bool(re.match(r"\s*(mg|g|kg|mL|ml|%|IU|kcal)", tail))


def has_contact(sentence: str) -> bool:
    """이 문장에 국내에서 쓸 수 없는 연락처가 들어 있나.

    **출처 표기 `(출처: …, S-xxx)` 는 검사하지 않는다.** 우리가 붙이는 고정 형식이고
    행동 지시가 아니다 (`_CITATION` 주석 참조).

    판정은 세 갈래다.

    ============================  ====================================
    무엇                          조건
    ============================  ====================================
    명백한 전화번호 형식           그 자체로 차단
    URL                           그 자체로 차단
    애매한 숫자 묶음 · 기관명       **행동 지시어가 같은 문장에 있을 때만**
    ============================  ====================================

    애매한 쪽에 행동 지시어를 요구하는 이유는 오탐이 더 위험하기 때문이다 —
    `2020-2024` 나 `8자리 연도` 하나 때문에 응급 안내 문장이 통째로 사라지면
    D-47 이 막으려던 것보다 나쁜 일이 된다.
    """
    text = _CITATION.sub(" ", sentence)
    low = text.lower()
    has_action = any(a.lower() in low for a in _ACTION)

    for m in _PHONE.finditer(text):
        if not _looks_like_dose(text, m.span()):
            return True

    if any(u.lower() in low for u in _FOREIGN_URLS):
        return True

    if not has_action:
        return False

    for m in _PHONE_AMBIGUOUS.finditer(text):
        if not _looks_like_dose(text, m.span()):
            return True

    return any(o.lower() in low for o in _FOREIGN_ORGS)


@dataclass(frozen=True)
class ScrubResult:
    """`removed` 는 로그용이다. **무엇을 뺐는지 남기지 않으면 검증할 수 없다** (04 §8)."""

    text: str
    removed: list[str]

    @property
    def changed(self) -> bool:
        return bool(self.removed)


def scrub_contacts(text: str) -> ScrubResult:
    """연락처가 든 문장을 빼고, 뺐으면 국내 안내 문장을 붙인다.

    ⑤ 평이화 **다음**, 사용자에게 나가기 직전에 부른다.
    앞에서 부르면 이후 단계가 번호를 다시 만들어 넣을 수 있다.
    """
    if not text or not text.strip():
        return ScrubResult(text, [])

    kept: list[str] = []
    removed: list[str] = []
    for sent in _SENT.split(text.strip()):
        (removed if has_contact(sent) else kept).append(sent)

    if not removed:
        return ScrubResult(text, [])

    kept.append(GUIDANCE)
    return ScrubResult(" ".join(s.strip() for s in kept if s.strip()), removed)


def scrub_items(items: Sequence[str]) -> ScrubResult2:
    """목록을 **항목 단위**로 거른다. 문장 단위로 하면 안 되는 자리에 쓴다.

    `TriageResult.escalation_conditions` 가 그런 자리다. 코퍼스에 실제 유출이 있다.

        F-021-012.escalation_conditions
          ['개를 원인 물질에서 즉시 떼어놓기',
           '먹은 것·양·시각 확인하고 포장 보관',
           '임의로 구토를 유도하지 말 것',
           '수의사·응급병원 또는 Pet Poison Helpline (855)-764-7661 에 즉시 연락',   ← 이것만
           '수의사 승인 없이 사람 약을 주지 말 것']

    문장 단위로 처리하면 리스트 전체가 한 덩어리가 되어 **나머지 네 조건까지 사라진다.**
    빠진 자리는 부르는 쪽이 `GUIDANCE` 로 메운다 — 정보 손실이 없다.
    """
    kept: list[str] = []
    removed: list[str] = []
    for item in items:
        (removed if has_contact(item) else kept).append(item)
    return ScrubResult2(tuple(kept), removed)


@dataclass(frozen=True)
class ScrubResult2:
    """`scrub_items` 의 결과. `text` 가 아니라 `items` 라서 타입을 나눴다."""

    items: tuple[str, ...]
    removed: list[str]

    @property
    def changed(self) -> bool:
        return bool(self.removed)
