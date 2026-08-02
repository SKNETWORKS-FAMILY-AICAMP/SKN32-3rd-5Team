"""별칭 표 — 보호자가 쓰는 말을 **코퍼스의 물질명으로 잇는다.**

설계 근거: docs/06_설계결정기록.md · D-59 · D-60 · D-61 · D-51 · D-46 · D-58

이 표는 무엇을 하는 곳인가 (2026-08-02 존재의의 재검토)
--------------------------------------------------
> **코드가 놓을 수 없는 다리를 사람이 근거와 함께 놓는 곳이다.**

51행을 하나씩 되물어 보면 표가 하는 일이 넷이고, **그중 하나는 이 표의 일이 아니다.**

    ① 다리   문자열로는 길이 없는 것을 잇는다      21행   프라이팬→PTFE · 소주→알코올
    ② 방어   부분일치가 **틀린 답**을 주는 자리      6행   아래 참조
    ③ 표시   `kind` 로 확정과 추정을 가른다        전 행   D-59 ⑤가 이 열에 의존한다
    ④ 철자   같은 개념의 띄어쓰기·이형태          20행   ← **매처가 할 일을 표가 떠안았다**

**②가 왜 값진가** — 별칭을 끄고 부분일치만 돌리면 이렇게 된다.

    체리 씨       별칭 O → 벚나무속 과일 씨    별칭 X → **체리**      ← 과육이다
    복숭아 씨앗    별칭 O → 벚나무속 과일 씨    별칭 X → **복숭아**     ← 과육이다
    소독용 알코올  별칭 O → 이소프로필알코올     별칭 X → 모호(주류 알코올과 섞인다)

**씨와 과육은 다른 것이다.** 씨는 시안화물이고 과육은 그 얘기가 아니다.
이것을 지키는 것은 `vocabulary.resolve_substance` 의 **순서**(별칭 → 부분일치)뿐이고,
`tests/test_vocabulary.py::test_별칭이_부분일치보다_먼저다` 가 그 순서를 잠근다.

**④는 부채다.** 51행 중 20행(39%)이 표기 변형이다 — `체리씨`·`체리 씨`·`체리씨앗`·
`체리 씨앗` 이 각각 한 행이라 과일 하나에 4행이다. 다리를 놓지 않는데 표를 키우고,
검사를 부풀리고(같은 근거를 20번 확인한다), 유지를 비대칭으로 만든다(새 과일 = 4행,
셋만 쓰면 나머지 하나가 조용히 죽는다). **띄어쓰기는 열린 집합이 아니라 정규화다** —
매칭 전 공백 제거로 20행 → 10행이 된다. 마감(8/4) 뒤 회차로 미뤘다.

왜 필요한가
----------
보호자는 코퍼스의 이름으로 말하지 않는다. 실측 (2026-08-02, 533종 대상) —

    '쪽파' · '실파'      → 코퍼스 0건.  코퍼스는 `알리움류(양파·마늘·리크·차이브)` 로 적는다
    '프라이팬' · '코팅팬' → 코퍼스 0건.  코퍼스는 `PTFE(테플론) 과열 흄` 으로 적는다
    '무설탕껌'           → 코퍼스 0건.  코퍼스는 `자일리톨(무설탕 껌·사탕·…)` 로 적는다

`rules.lookup` 은 부분 문자열로 찾으므로 이 말들은 **한 건도 못 잡는다.**
골든셋 `G-039`(대파/cat)가 실패하던 원인이 이것이다.

낱말 경계로만 걸린다 (D-60)
--------------------------
1차 표는 부분 문자열로 걸었고 **오탐 14건**을 냈다 (2026-08-02 재검토 실측) —

    초코파이 → 알리움류      중성화 수술 → 알코올      파스타 → 알리움류
    양파    → 알리움류      마취술      → 알코올      소주방 → 알코올

막던 방식은 `_BLOCK_AFTER = {"파": "슬프인란"}` 였다. **뒷글자만** 막고, 그것도
**열린 집합**이라 `초코파이`(앞글자가 문제)는 원리적으로 못 막는다.

지금은 반대로 간다 — 무엇을 막을지가 아니라 **무엇이면 낱말인지**를 정한다.

    앞:  문두이거나 한글이 아닐 것            (`초코`파이 · `양`파 · 수`술` → 탈락)
    뒤:  문말이거나 한글이 아닐 것,
         또는 **조사가 와서 낱말을 끝낼 것**   (파스`타` · 소주`방` → 탈락 / 대파`를` → 통과)

조사는 **닫힌 집합**이다. 차단 목록은 아니었다. 그 차이가 유지 가능성을 가른다.

**1글자 별칭은 두지 않는다.** 경계 규칙을 넣어도 `파` 는 `파도`·`파는`(팔다)에
걸린다 — `도`·`는` 이 조사이면서 낱말이기도 하기 때문이다. 1글자는 조사와
구별되지 않으므로 규칙으로는 못 막는다. 그래서 `load_aliases` 가 **거부한다.**
잃는 것(*"파를 먹었어요"*)은 `대파`·`쪽파`·`파김치` 행으로 되찾고, 못 되찾는
것은 못 잡는다 — **놓치면 정성 답변으로 내려가고, 잘못 걸면 우리가 등급을 만든다** (D-46).

미등재 합성어는 걸리지 않는다. `생맥주`·`파김치`·`체리 씨앗` 처럼 **행을 늘려서**
넓힌다. 규칙에 예외를 붙이면 다시 열린 집합이 된다.

경계 규칙이 못 막는 것 — 지명 (2026-08-02 재검토)
------------------------------------------------
`청주`·`양주` 는 술이면서 **도시 이름**이다. 낱말 경계 규칙은 이것을 못 막는다.
*"청주에서 갈 만한 동물병원 있나요"* 의 `에서` 는 정상적인 조사이기 때문이다.

    청주에서 갈 만한 동물병원 있나요   → 알코올   ← 규칙으로는 구별 불가
    양주에 사는데 병원 추천해 주세요    → 알코올

그래서 **표에서 뺐다.** `소주`·`맥주`·`생맥주`·`막걸리`·`위스키`·`와인` 이 남는다.
동음이의어는 규칙이 아니라 **표가 판단한다** — 넣을지 말지가 사람의 결정이다.
`tests/test_aliases.py` 의 오탐 목록이 이 둘을 지키고 있다.

두 종류를 구분한다
-----------------
별칭이라고 다 같지 않다. 확실한 것과 **추정인 것**이 섞여 있고, 섞어 두면
추정이 확정으로 나간다.

    확정   대파 → 알리움류      대파(Allium fistulosum)는 **알리움 속이다.** 분류학적 사실
    추정   프라이팬 → PTFE 흄    **무쇠·스테인리스 팬은 PTFE 를 내지 않는다.** 가정이 들어간다

`kind` 열이 그 구분이고, `추정` 은 D-59 ④ 절차를 탄다 — 확인 1회, 못 정하면
상향 가정 + `AskResponse.assumed_substance` 에 남긴다. **밝히지 않은 추정은 환각이다.**

`basis` 는 무엇인가 (D-61)
-------------------------
**대상 물질(`substance`)이 그 이름으로 실재함을 보이는 사실 행들.** 별칭 자체가
거기 적혀 있을 필요는 없다 — `대파` 는 어느 근거에도 없지만 `알리움류` 는 있다.

이 정의가 없어서 1차 표에 **틀린 근거 3건**이 들어갔다 (2026-08-02 재검토) —

    술 → 알코올            근거 `F-029-002` = **아보카도**(mammal)
    살구씨 → 핵과 씨앗      근거 `F-029-016` = **위장관 폐색 이물(아보카도 씨·옥수수속대·뼈)**

`tests/test_aliases.py` 가 id 의 **실재**만 보고 **관련성**을 안 봤기 때문에
셋 다 초록이었다. 지금은 관련성과 **종 범위**를 함께 본다 (D-58 의 연장):

    관련성   근거 **전부**의 `substance` 안에 대상 물질명이 들어 있을 것
    종 범위  이 행이 적용되는 종을 근거들이 **덮을** 것 (`all` → 전 종, `mammal` → dog·cat)

종 범위를 안 보면 `살구씨` 처럼 된다 — **근거는 조류 자료뿐인데 전 종에 걸렸다.**
`프라이팬` 을 `bird` 로 좁힌 것과 같은 문제인데 거기만 좁혀져 있었다.

역치를 갈아끼우지 않는다 (D-51 의 연장)
------------------------------------
`대파 → 양파` 로 두면 규칙표 적중이 늘어난다 (`양파` 는 dog·cat 둘 다 역치가 있고
`알리움류` 는 cat 만 있다). **그렇게 하지 않는다.**

대파와 양파는 알리움 속의 다른 종이고 티오설페이트 함량이 같다는 근거가 없다.
적중률을 위해 다른 종의 역치를 빌려 오면 **우리가 만든 숫자가 등급을 판정한다** —
D-51 이 `g leaves/kg` 에서 막은 것과 같은 조작이다.

속 수준 별칭(`알리움류`)으로 두면 dog 은 계산 가능한 역치가 없어 **정성 답변**으로
내려간다. 그것이 정직한 결과다 (D-46: 정량 포기는 실패가 아니다).

표를 손으로 채운다
----------------
`정량임계치.csv` 와 달리 **생성물이 아니다.** 사람이 쓰고, 근거(`basis`)에
어느 사실 행이 그 매핑을 뒷받침하는지 적는다. `tests/test_aliases.py` 가
그 근거를 코퍼스와 대조한다.
"""

from __future__ import annotations

import csv
import logging
from dataclasses import dataclass
from functools import lru_cache
from importlib import resources
from pathlib import Path

from .vocabulary import MIN_TERM_LEN, is_word_hit
from .vocabulary import SPECIES as _SPECIES

log = logging.getLogger(__name__)

TABLE_NAME = "별칭.csv"

#: 매핑의 성격. **섞으면 추정이 확정으로 나간다.**
KINDS = ("확정", "추정")

#: 표가 쓸 수 있는 종. 코퍼스의 `all`·`mammal` 은 **종이 아니라 묶음**이라
#: 표에는 쓰지 않는다 — 표는 질의가 들고 오는 값(개·고양이·앵무새)으로 적는다.
#:
#: **정의는 `vocabulary` 에 있다.** 여기서 다시 적으면 두 곳이 어긋난다 (P2).
SPECIES = _SPECIES

#: 별칭 최소 길이. **1글자는 조사와 구별되지 않는다** (D-60).
#:
#: `파` 는 낱말 경계 규칙을 넣어도 `파도`·`파는`(팔다)에 걸린다.
#: `도`·`는` 이 조사이면서 그 자체로 낱말의 일부이기 때문이다.
#: 규칙으로 못 막는 것은 **표에 두지 않는다.**
#:
#: **값의 정의는 `vocabulary.MIN_TERM_LEN` 하나뿐이다** — 부분 일치도 같은 이유로
#: 같은 하한을 쓴다. 두 곳에 적으면 어긋난다 (P2).
MIN_ALIAS_LEN = MIN_TERM_LEN


class AliasTableMissingError(RuntimeError):
    """표를 못 찾았다. **조용히 빈 표로 넘어가지 않는다.**

    빈 표로 돌면 별칭이 하나도 안 걸리는데 그 사실이 아무 데도 안 드러난다 —
    `rules.lookup` 은 그냥 0건을 돌려주고, 그것이 *"코퍼스에 근거가 없다"* 로 읽힌다.
    """


class AliasTableInvalidError(ValueError):
    """표의 한 행이 규칙을 어겼다. **그 행만 버리지 않고 전체를 세운다.**

    한 행을 조용히 건너뛰면 그 별칭이 안 걸리는 이유를 아무도 모른다.
    표는 사람이 쓰는 것이고, 사람이 쓴 것이 틀렸으면 **쓴 사람이 알아야 한다.**
    """


@dataclass(frozen=True)
class Alias:
    """별칭 한 행."""

    alias: str
    substance: str
    kind: str
    species: tuple[str, ...]  # 비어 있으면 전 종
    basis: tuple[str, ...]
    note: str

    @property
    def is_assumption(self) -> bool:
        """추정인가. 참이면 **사용자에게 밝혀야 한다** (D-59 ⑤)."""
        return self.kind == "추정"

    @property
    def covers(self) -> frozenset[str]:
        """이 행이 실제로 적용되는 종. 빈 `species` 는 전 종이다."""
        return frozenset(self.species) if self.species else frozenset(SPECIES)

    def applies_to(self, species: str | None) -> bool:
        """이 별칭이 그 종에 적용되나. `species` 가 비어 있으면 전 종에 적용된다."""
        return not species or species in self.covers


def _table_path() -> Path:
    """설치 형태와 무관하게 표를 찾는다 (`rules._table_path` 와 같은 방식)."""
    p = resources.files("pettriage.compute") / "tables" / TABLE_NAME
    with resources.as_file(p) as real:
        if real.exists():
            return real
    raise AliasTableMissingError(f"{TABLE_NAME} 를 찾지 못했다 (compute/tables/).")


@lru_cache(maxsize=1)
def load_aliases() -> tuple[Alias, ...]:
    """표 전체. **긴 별칭부터** 정렬해 돌려준다.

    정렬이 중요하다 — `파김치` 보다 짧은 것이 먼저 걸리면 근거로 남는 별칭이 달라진다.
    길이가 같으면 별칭 문자열로 다시 정렬해 **순서를 결정적으로** 만든다.

    규칙을 어긴 행은 `AliasTableInvalidError` 로 **전체를 세운다.**
    """
    path = _table_path()
    out: list[Alias] = []
    with path.open(encoding="utf-8-sig", newline="") as f:
        for r in csv.DictReader(f):
            alias = (r.get("alias") or "").strip()
            if not alias:
                continue
            if len(alias) < MIN_ALIAS_LEN:
                raise AliasTableInvalidError(
                    f"{TABLE_NAME}: {alias!r} 는 {MIN_ALIAS_LEN}글자 미만이다. "
                    "1글자 별칭은 조사와 구별되지 않아 오탐을 낸다 (D-60). "
                    "합성어를 행으로 나눠 적는다 — 예: '파' 대신 '대파'·'쪽파'·'파김치'."
                )
            kind = (r.get("kind") or "").strip()
            if kind not in KINDS:
                raise AliasTableInvalidError(
                    f"{TABLE_NAME}: {alias!r} 의 kind 가 {KINDS} 가 아니다: {kind!r}"
                )
            species = tuple(s for s in (r.get("species") or "").replace(",", "|").split("|") if s)
            unknown = [s for s in species if s not in SPECIES]
            if unknown:
                raise AliasTableInvalidError(
                    f"{TABLE_NAME}: {alias!r} 의 species 에 모르는 값이 있다: {unknown}. "
                    f"쓸 수 있는 값은 {SPECIES} 이고, 비우면 전 종이다."
                )
            out.append(
                Alias(
                    alias=alias,
                    substance=(r.get("substance") or "").strip(),
                    kind=kind,
                    species=species,
                    basis=tuple(b for b in (r.get("basis") or "").split("|") if b),
                    note=(r.get("note") or "").strip(),
                )
            )
    if not out:
        raise AliasTableMissingError(f"{path} 가 비었다.")
    return tuple(sorted(out, key=lambda a: (-len(a.alias), a.alias)))


def resolve(text: str, species: str | None = None) -> list[Alias]:
    """질의 문장에서 별칭을 찾는다. **긴 것부터, 대상이 겹치면 하나만.**

    Args:
        text: 보호자 질의 원문. 낱말이 아니라 문장을 그대로 넣어도 된다
        species: 종. 종이 다른 별칭은 걸러진다 (`프라이팬` 은 조류 자료뿐이다)

    Returns:
        걸린 별칭들. **`확정` 이 `추정` 보다 앞에 온다** — 부르는 쪽이
        앞에서부터 쓰면 확실한 것을 먼저 시도하게 된다.

    ⚠️ **확정 우선이 길이 우선보다 먼저다.** 대상이 같으면 하나만 남기는데,
    길이순으로만 돌면 `"프라이팬으로 코팅팬을 덮었어요"` 에서 `프라이팬`(4글자·**추정**)이
    `코팅팬`(3글자·**확정**)을 밀어낸다. 남는 근거가 추정으로 바뀌고,
    `AskResponse.assumed_substance` 가 필요 없는 자리에 가정이 붙는다 (D-59 ⑤).
    2026-08-02 재검토에서 잡았다.
    """
    if not text:
        return []
    found: list[Alias] = []
    seen: set[str] = set()
    # 안정 정렬이므로 `확정` 안에서는 `load_aliases()` 의 길이순이 유지된다.
    for a in sorted(load_aliases(), key=lambda a: a.is_assumption):
        if a.substance in seen or not a.applies_to(species):
            continue
        if is_word_hit(text, a.alias):
            found.append(a)
            seen.add(a.substance)
    return found


def substances_for(text: str, species: str | None = None) -> list[str]:
    """`resolve` 의 결과에서 대상 물질명만. 검색어 보강에 쓴다."""
    return [a.substance for a in resolve(text, species)]
