"""판정 근거 — **이 등급이 어디서 나왔는가** (D-81).

    🔴 지금 전화        ← 50 mg/kg 을 계산해서 나온 3
    🔴 지금 전화        ← 양을 몰라 바닥으로 깔아 둔 3

**두 개가 화면에서 똑같이 생겼다.** 등급이라는 형식 자체가 확신을 표현하는데,
그 확신의 출처가 응답 어디에도 없었다. 밑에 한 줄 붙여도 사람은 배지를 읽고
문장은 흘린다 — **밝혔지만 밝힌 것이 아니다.**

그래서 근거를 **각주가 아니라 필드**로 낸다. `basis` 가 `level` 과 같은 급으로 나가고,
계약이 *"문장에도 실렸나"* 를 검사한다 (D-59 ⑤ 가 추정 물질에 한 것과 같은 방식).

## 왜 이 파일이 따로 있나

    태그 문자열(`[확인 안 됨]`)을 **만드는 쪽과 검사하는 쪽이 각자 적으면 어긋난다.**
    어긋나는 순간 계약이 통과시키고, 그러면 강제가 사라진다. 두 곳이 이 모듈 하나를
    부른다 (D-22).

## 얼마나 걸리는 일인가

    골든셋 60건 중 `answered` 기대가 39건인데, **체중과 섭취량이 둘 다 있는 건은
    13건(33%)뿐**이다. 나머지 26건은 결손 상태에서 등급을 낸다 —
    **답변 셋 중 둘이 그렇다.**
"""

from __future__ import annotations

from typing import Literal

#: 판정 근거. **`level` 과 같은 급으로 응답에 나간다.**
Basis = Literal["정량계산", "최악가정", "양미상", "정성", "모델판정"]

#: 근거 → 문장 앞에 세우는 태그. **`_assumption_notice` 의 `[확인되지 않은 가정]` 과
#: 같은 표기 규칙이다.** 계약이 이 태그로 *"밝혔나"* 를 확인한다.
TAGS: dict[str, str] = {
    "정량계산": "[계산]",
    "최악가정": "[가정]",
    "양미상": "[확인 안 됨]",
    "정성": "[근거]",
    "모델판정": "[규칙 없음]",
}

#: **밝히지 않으면 응답을 만들 수 없는 근거.**
#:
#: `정량계산` 은 뺄까 고민했다 — 결손이 아니니 사과할 일이 아니다. 그러나 D-16 이
#: *"수치는 코드가 계산한다"* 로 정한 그 수치가 **응답 문장에 한 번도 나간 적이 없었다**
#: (D-75 에서 필드로는 올렸으나 문장에는 못 넣었다). 계산했다는 증거가 화면에 없으면
#: 산출물로 보일 것이 없다. **넣는다.**
MUST_STATE: frozenset[str] = frozenset(TAGS)


def notice(
    basis: str,
    *,
    active: str | None = None,
    mg_per_kg: float | None = None,
    detail: str | None = None,
    thresholds: str | None = None,
) -> str:
    """문장 앞에 세울 한 문단. **한 문장씩만 쓴다** — 단서를 많이 달면 아무도 안 읽는다.

    ⚠️ *"불확실할 수 있습니다"* 같은 말은 쓰지 않는다. 아무 정보가 없으면서
    신뢰만 깎는다. 무엇을 몰랐고 그래서 무엇을 했는지만 적는다.
    """
    tag = TAGS.get(basis, "")
    if basis == "정량계산":
        head = f"{tag} {active or '유효성분'} 약 {mg_per_kg:.4g} mg/kg"
        if detail:
            head += f" — {detail}"
        return head + "." + (f" 자료의 역치는 {thresholds} 입니다." if thresholds else "")
    if basis == "최악가정":
        return (
            f"{tag} {detail or '종류'}를 확인하지 못해, 함량이 가장 높은 기준으로 "
            "계산했습니다. 종류를 알려주시면 다시 판단해 드립니다."
        )
    if basis == "양미상":
        what = detail or "섭취량"
        return (
            f"{tag} {what}을 확인하지 못해 정량 판정을 하지 못했습니다. "
            "적게 먹었다는 뜻이 아니라, 판단할 수치가 없다는 뜻입니다."
        )
    if basis == "정성":
        return f"{tag} 수치 계산이 아니라 자료의 등급 표에 따른 판정입니다."
    if basis == "모델판정":
        return (
            f"{tag} 규칙 표에서 이 조합의 등급을 찾지 못해, 모델 판단만으로 매긴 "
            "등급입니다. 수치 근거가 없습니다."
        )
    return ""


def stated_in(basis: str | None, text: str) -> bool:
    """그 근거가 문장에 실렸나. **계약이 이것을 부른다.**"""
    if not basis or basis not in MUST_STATE:
        return True
    return TAGS[basis] in (text or "")
