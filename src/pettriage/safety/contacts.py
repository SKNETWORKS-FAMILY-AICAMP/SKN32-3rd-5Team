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

#: 전화번호로 볼 수 있는 숫자 묶음.
#:
#: 미국 톨프리(`1-855-764-7661` · `(888) 426-4435` · `888.426.4435`)와
#: 국내 형식(`02-1234-5678` · `1588-1234`)을 함께 잡는다.
#: **국내 번호도 막는 이유** — 지금 코퍼스에 검증된 국내 연락처가 한 건도 없다.
#: 나중에 검증된 창구가 생기면 그때 예외를 여기에 명시적으로 뚫는다.
_PHONE = re.compile(
    r"""
    (?<![\d.])                      # 앞이 숫자·소수점이면 용량 수치다 (2.3 g/kg)
    (?:\+?\d{1,3}[\s.\-])?          # 국가번호
    (?:\(\d{2,4}\)|\d{2,4})         # 지역·식별 번호
    [\s.\-]\d{3,4}
    [\s.\-]\d{4}
    (?![\d.])
    """,
    re.VERBOSE,
)

#: 하이픈 없이 붙여 쓴 국내 대표번호(`15881234`)까지는 잡지 않는다 —
#: 4자리+4자리 숫자는 연도·용량과 구별이 안 돼 오탐이 더 위험하다.

#: 국내에서 걸 수 없는 해외 상담 창구. **번호를 지워도 기관명이 남으면 소용없다.**
_FOREIGN_ORGS = (
    "Pet Poison Helpline",
    "Animal Poison Control Center",
    "APCC",
    "ASPCA Poison",
    "Poison Control Center",
    "FOUR PAWS",
    "petpoisonhelpline",
    "aspca.org/pet-care/animal-poison-control",
)

#: 기관명이 나왔더라도 **행동 지시**가 아니면 지우지 않는다.
#: *"ASPCA APCC 자료에 따르면 아보카도는 조류에게 치명적이다"* 는 살려야 한다.
_ACTION = ("연락", "전화", "문의", "상담", "call", "contact", "신고")

#: 문장 경계. 한국어 종결 `~다.` 와 영문 마침표를 함께 본다.
_SENT = re.compile(r"(?<=[.!?。？！])\s+")


def _looks_like_dose(text: str, span: tuple[int, int]) -> bool:
    """수치 뒤에 단위가 붙어 있으면 전화번호가 아니라 용량이다."""
    tail = text[span[1] : span[1] + 6]
    return bool(re.match(r"\s*(mg|g|kg|mL|ml|%|IU|kcal)", tail))


def has_contact(sentence: str) -> bool:
    """이 문장에 국내에서 쓸 수 없는 연락처가 들어 있나."""
    for m in _PHONE.finditer(sentence):
        if not _looks_like_dose(sentence, m.span()):
            return True
    low = sentence.lower()
    if any(o.lower() in low for o in _FOREIGN_ORGS):
        return any(a.lower() in low for a in _ACTION)
    return False


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
