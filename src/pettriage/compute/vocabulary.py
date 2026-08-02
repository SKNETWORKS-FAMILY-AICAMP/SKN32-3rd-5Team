"""물질 어휘 — **폐쇄 목록.** 이 밖의 이름은 우리 것이 아니다.

설계 근거: docs/06_설계결정기록.md · D-59 ① · D-40 · D-38 · D-22

무엇을 막는가
------------
D-59 ①이 정한 것은 이것이다.

    ✗  "이 상황의 물질은?"                   → 생성 → 환각 가능
    ✓  "다음 중 어느 것인가? 없으면 '없음'"    → 선택 → 환각을 **막는다**

그런데 그때 정한 것은 **프롬프트가 그렇게 부탁한다**까지였다. 부탁은 어길 수 있다.
모델이 `일산화탄소` 라고 답하면 — 코퍼스에 없는 물질이다 — 지금까지는 아무것도
그것을 막지 않았다. D-40 이 말한 그대로다.

    **지키기로 한 것이 아니라 못 어기는 것.**

그래서 목록을 파일로 두고, 두 겹으로 쓴다.

    정규화   `resolve_substance`   표면형을 목록 위로 **올린다.** 못 올리면 되묻기
    안전망   `contracts.SubstanceName`   목록 밖 이름으로는 **응답이 안 만들어진다**

⚠️ **거부기가 아니라 정규화기가 먼저다.** 처음에는 계약만 넣었는데, ②가 뽑는 것은
보호자의 **표면형**이라 흔한 표현 30개 중 12개가 목록 밖이었다 (`커피`·`우유`·**`대파`**).
그러면 정상 질의가 계약에서 죽는다. 그리고 05 §6 은 ②의 검증 실패를 **되묻기**로,
02 §6 그래프는 `결측·물질미상 → ask_clarify` 로 이미 정해 놨다 —
**예외는 그 경로를 우회한다.** 2026-08-02 재검토에서 방향을 바꿨다.

정규화를 거치면 계약은 `_no_foreign_contacts` 와 같은 지위가 된다 —
정상 경로에서는 안 걸리고, 걸렸다면 **정규화를 안 거쳤다는 뜻**이다.

무엇을 못 막는가 — 정직하게
--------------------------
폐쇄 목록은 **환각을 막지 오분류를 막지 못한다.**

    환각    코퍼스에 **없는** 것을 만든다      → 이 파일이 막는다. 범위가 무한이라 막아야 한다
    오분류  있는 것 중 **틀린** 것을 고른다    → 못 막는다. 538종 안이므로 **골든셋으로 잰다**

D-59 ③이 나눈 것이 이 둘이고, 여기는 앞의 절반만 담당한다.

목록은 어디서 오나
-----------------
두 파일의 합집합이다. **둘 다 사람이 쓰거나 사람이 쓴 것에서 파생한다.**

    물질어휘.csv   코퍼스 물질명 533종.  `facts_*.csv` 에서 생성 (`make vocab`)
    별칭.csv       별칭이 가리키는 계열명 7종. 사람이 쓰고 근거를 붙인다

별칭 대상을 함께 넣는 이유 — `대파 → 알리움류` 의 `알리움류` 는 코퍼스에
그 이름 그대로 있지 않다 (코퍼스는 `알리움류(양파·마늘·리크·차이브)` 로 적는다).
`tests/test_aliases.py` 가 **그 계열명이 코퍼스에 실재하는지**를 이미 대조하므로
근거 없는 이름이 아니다. 두 목록 모두 **닫혀 있고 파일에서 온다.**

⚠️ **다만 그 5개는 코퍼스 어휘가 아니라 우리 어휘다** (2026-08-02 존재의의 재검토).

    PTFE            ← 코퍼스: 'PTFE(테플론) 과열 흄'
    알리움류          ← 코퍼스: '알리움류(양파·마늘·리크·차이브)'
    벚나무속 과일 씨    ← 코퍼스: '벚나무속 과일 씨(체리·천도복숭아·복숭아·자두)'
    에센셜 오일        ← 코퍼스: '에센셜 오일(정유) 전반 — …'
    이소프로필알코올     ← 코퍼스: '이소프로필알코올(소독용 알코올)'

전부 코퍼스 이름의 **부분 문자열**이고 검사가 대조하므로 지어낸 이름은 아니다.
그러나 `identified_substance="PTFE"` 로 나가면 **출처 문서에 그 이름은 없다.**
*"폐쇄 목록 = 코퍼스"* 는 정확한 문장이 아니다 — **533 + 우리 축약형 5** 다.

왜 코퍼스를 런타임에 안 읽나
--------------------------
`data/facts/facts_*.csv` 는 **패키지에 안 들어간다.** 설치본에는 `compute/tables/`
만 실린다. 어휘를 런타임에 코퍼스에서 만들면 설치 환경에서 어휘가 **조용히 비고**,
검사가 전부 통과하거나 전부 실패한다. **둘 다 신호가 아니다** (D-58).
`정량임계치.csv` 와 같은 방식으로 생성물을 파일에 고정한다 (D-22).
"""

from __future__ import annotations

import csv
import re
from dataclasses import dataclass
from functools import lru_cache
from importlib import resources
from pathlib import Path

TABLE_NAME = "물질어휘.csv"

#: 질의가 들고 오는 종. 코퍼스의 `all`·`mammal` 은 **종이 아니라 묶음**이다.
#:
#: 여기가 단일 출처다 — `aliases.py` 도 `tests/` 도 이것을 가져다 쓴다.
#: 예전에는 테스트가 자기 사본을 들고 있었다 (P2: 두 곳에 적으면 반드시 어긋난다).
SPECIES: tuple[str, ...] = ("dog", "cat", "bird")

#: 코퍼스의 `species` 값이 **실제로 덮는 종.**
_COVERS: dict[str, frozenset[str]] = {
    "all": frozenset(SPECIES),
    "mammal": frozenset({"dog", "cat"}),
    "dog": frozenset({"dog"}),
    "cat": frozenset({"cat"}),
    "bird": frozenset({"bird"}),
}

#: **정상 선택지다.** 목록에 없으면 이것을 고르고, D-49 되묻기로 간다.
#: 거절이 아니다 — 물질이 미확정이어도 정성 판정은 가능하다 (D-59 ④).
NONE = "없음"

#: 한국어에서 **한 글자는 식별력이 없다** (D-60). 별칭 표의 하한이자 부분 일치의 하한.
#:
#: `파` 는 낱말 경계 규칙을 넣어도 `파도`·`파는`(팔다)에 걸리고, 코퍼스 물질명 `인`
#: (무기질)은 역방향 포함으로 `카페인` 에 걸린다. **둘 다 같은 병이고 같은 하한이 막는다.**
MIN_TERM_LEN = 2

_HANGUL = re.compile(r"[가-힣]")

#: 별칭 **뒤에 붙어도 낱말이 끝나는** 것들 — 조사와 서술격 어미.
#:
#: 한국어 조사는 **닫힌 집합**이다. 이 목록이 유지 가능한 이유가 그것이고,
#: 1차 표의 `_BLOCK_AFTER`(무엇을 막을지 나열)가 유지 불가능했던 이유도 그것이다.
#:
#: **순서는 상관없다.** `is_word_hit` 이 *"조사 뒤가 한글이 아닐 것"* 까지 함께 보므로
#: `이랑` 을 `이` 로 잘라도 남은 `랑` 이 한글이라 통과하지 않는다 —
#: 짧은 것이 긴 것을 가로채지 못한다. 길이순 정렬은 **사람이 읽기 위한 것**이다.
_PARTICLES: tuple[str, ...] = tuple(
    sorted(
        (
            "이라도",
            "이라는",
            "이랑",
            "에서",
            "에게",
            "한테",
            "부터",
            "까지",
            "보다",
            "처럼",
            "같이",
            "조차",
            "마저",
            "밖에",
            "이나",
            "이든",
            "으로",
            "라도",
            "라는",
            "하고",
            "이야",
            "인데",
            "이며",
            "이고",
            "을",
            "를",
            "은",
            "는",
            "이",
            "가",
            "도",
            "만",
            "의",
            "에",
            "와",
            "과",
            "랑",
            "로",
            "나",
            "야",
            "요",
            "께",
        ),
        key=len,
        reverse=True,
    )
)


def is_word_hit(text: str, alias: str) -> bool:
    """`alias` 가 `text` 안에서 **낱말로** 나타나나 (D-60).

    앞은 문두이거나 한글이 아니어야 하고, 뒤는 문말이거나 한글이 아니거나
    **조사가 와서 낱말을 끝내야** 한다.

        is_word_hit("초코파이", "파")         → False   앞이 한글(`코`)
        is_word_hit("파스타를", "파")         → False   뒤가 한글인데 `스타` 는 조사가 아니다
        is_word_hit("대파를 먹었어요", "대파")  → True    `를` 뒤가 공백
        is_word_hit("소주방", "소주")         → False   `방` 은 조사가 아니다

    형태소 분석은 하지 않는다. **표가 작을 때 규칙도 작아야 유지된다.**
    """
    n = len(alias)
    start = 0
    while (i := text.find(alias, start)) != -1:
        start = i + 1
        if i > 0 and _HANGUL.match(text[i - 1]):
            continue  # 앞이 한글 — 다른 낱말의 꼬리다
        rest = text[i + n :]
        if not rest or not _HANGUL.match(rest[0]):
            return True  # 문말이거나 공백·부호·영문·숫자
        if any(
            rest.startswith(p) and (len(rest) == len(p) or not _HANGUL.match(rest[len(p)]))
            for p in _PARTICLES
        ):
            return True  # 조사가 낱말을 끝냈다
    return False


class VocabularyMissingError(RuntimeError):
    """어휘 표를 못 찾았다. **빈 목록으로 넘어가지 않는다.**

    빈 목록이면 폐쇄 목록 검사가 **모든 이름을 거부**한다. 그 편이
    조용히 전부 통과시키는 것보다 낫지만, 어느 쪽이든 원인을 여기서 말한다.
    """


class UnknownSubstanceError(ValueError):
    """폐쇄 목록 밖의 물질명. **코퍼스에 없는 것으로는 답하지 않는다** (D-59 ①)."""


@dataclass(frozen=True)
class Term:
    """어휘 한 항목."""

    substance: str
    species: tuple[str, ...]  # 코퍼스 표기 그대로 (all · mammal · dog · cat · bird)
    n_facts: int

    @property
    def covers(self) -> frozenset[str]:
        """이 물질에 자료가 있는 종."""
        out: set[str] = set()
        for s in self.species:
            out |= _COVERS.get(s, frozenset())
        return frozenset(out)


def covers_of(corpus_species: str) -> frozenset[str]:
    """코퍼스 `species` 값 하나가 덮는 종. 모르는 값은 빈 집합이다."""
    return _COVERS.get(corpus_species, frozenset())


def _table_path() -> Path:
    p = resources.files("pettriage.compute") / "tables" / TABLE_NAME
    with resources.as_file(p) as real:
        if real.exists():
            return real
    raise VocabularyMissingError(
        f"{TABLE_NAME} 를 찾지 못했다 (compute/tables/). `make vocab` 으로 생성한다."
    )


@lru_cache(maxsize=1)
def load_vocabulary() -> tuple[Term, ...]:
    """어휘 표 전체. **생성물이다 — 손으로 고치지 않는다** (`make vocab`)."""
    path = _table_path()
    out: list[Term] = []
    with path.open(encoding="utf-8-sig", newline="") as f:
        for r in csv.DictReader(f):
            name = (r.get("substance") or "").strip()
            if not name:
                continue
            out.append(
                Term(
                    substance=name,
                    species=tuple(s for s in (r.get("species") or "").split("|") if s),
                    n_facts=int(r.get("n_facts") or 0),
                )
            )
    if not out:
        raise VocabularyMissingError(f"{path} 가 비었다. `make vocab` 으로 다시 만든다.")
    return tuple(out)


@lru_cache(maxsize=1)
def known_substances() -> frozenset[str]:
    """**폐쇄 목록 전체.** 코퍼스 물질명 + 별칭 표의 계열명.

    ⚠️ `aliases` 를 함수 안에서 임포트한다. 이 모듈이 어휘의 단일 출처라
    `aliases` 가 여기서 `SPECIES` 를 가져가고, 최상단에서 마주 임포트하면 순환이 된다.
    **어느 쪽이 아래인지 정해 두는 것**이 지연 임포트보다 중요하다 — 아래는 여기다.
    """
    from .aliases import load_aliases

    names = {t.substance for t in load_vocabulary()}
    names |= {a.substance for a in load_aliases()}
    return frozenset(names)


def is_known(name: str) -> bool:
    """폐쇄 목록 안인가. `'없음'` 도 참이다 — **정상 선택지**이기 때문이다."""
    return name == NONE or name in known_substances()


def check_substance(name: str) -> str:
    """폐쇄 목록 검사. 통과하면 그대로 돌려주고, 아니면 `UnknownSubstanceError`.

    부르는 쪽이 `if` 로 감싸지 않게 **값을 돌려주는 형태**로 둔다 —
    `AfterValidator` 가 이 모양을 요구하고, 계약에 그대로 꽂힌다.
    """
    if is_known(name):
        return name
    raise UnknownSubstanceError(
        f"{name!r} 는 코퍼스 물질 어휘 {len(known_substances())}종에 없다. "
        "물질 동정은 생성이 아니라 폐쇄 목록에서의 선택이다 (D-59 ①). "
        f"고를 것이 없으면 {NONE!r} 이 정상 선택지다."
    )


@dataclass(frozen=True)
class Resolution:
    """표면형 하나를 폐쇄 목록 위로 올린 결과. **못 올렸으면 `name` 이 `None` 이다.**"""

    surface: str
    name: str | None
    how: str  # 직접 · 별칭 · 부분일치 · 모호 · 없음
    assumption: bool = False  # 추정 별칭을 탔나 → `assumed_substance` 에 남겨야 한다 (D-59 ⑤)
    candidates: tuple[str, ...] = ()  # `모호` 일 때 무엇들 사이에서 못 정했나

    @property
    def ok(self) -> bool:
        return self.name is not None


def resolve_substance(surface: str, species: str | None = None) -> Resolution:
    """보호자 표면형 → **폐쇄 목록 안의 이름.** 못 정하면 `None` 이다. **예외를 던지지 않는다.**

    ⚠️ **이 함수는 거부기가 아니라 정규화기다.** 2026-08-02 재검토에서 방향을 바꿨다.

        처음에는 목록 밖이면 `UnknownSubstanceError` 를 던지게 만들었다.
        그런데 ②가 발화에서 뽑는 것은 **표면형**이고, 실측하니 흔한 표면형 30개 중
        **12개가 목록 밖**이었다 — `커피`·`우유`·`이부프로펜`, 그리고 **`대파`**.
        골든셋 `G-039` 가 바로 그 질의다. 던지면 **정상 질의가 계약에서 죽는다.**

        그리고 실패 방식이 문서와 달랐다. 05 §6 은 ①분류를
        *"허용목록에 있는가 → **폴백 라벨 적용 후 로그**"*, ②슬롯을
        *"검증 실패 → **되묻기**"* 로 정해 뒀고, 02 §6 그래프도
        `extract_slots ─ 결측·**물질미상** → ask_clarify` 로 그려 놨다.
        **예외는 그 경로를 우회한다.**

    올리는 순서 — `rules.lookup` 과 **같은 규칙 위에 선다.**

        ① 직접     목록에 그대로 있다              초콜릿 → 초콜릿
        ② 별칭     별칭 표가 가리킨다 (확정 먼저)    대파   → 알리움류
        ③ 부분일치  코퍼스 이름 하나가 품는다        카페인 → 초콜릿(테오브로민+카페인)
        ④ 모호     둘 이상이 품는다                 → None. 하나를 고르면 그게 진단이다 (D-11)
        ⑤ 없음     아무것도 없다                    커피   → None

    ④와 ⑤가 `None` 인 것이 정상 동작이다. 부르는 쪽은 `missing_slots` 에 넣고
    `ask_clarify` 로 보낸다 — **거절이 아니다** (D-49 · D-59 ④).

    ⚠️ **낱말을 넣는다. 문장을 넣지 않는다.**

    문장을 넣어도 예외는 안 나지만 **조용히 덜 찾는다** — 역방향(코퍼스 이름 ⊂ 입력)만
    걸리므로 코퍼스 이름이 **글자 그대로 들어 있는 것**만 잡힌다. 실측 —

        '고양이가 우유랑 초콜릿을 먹었어요'  →  초콜릿 **하나만.** 우유는 조용히 빠진다
                                            (코퍼스는 `우유(모유 대용)`·`우유·유제품`)

    문장에서 찾는 일은 `aliases.resolve`/`substances_for` 의 몫이다 — 그쪽이 문장을
    받도록 설계돼 있다. 여기는 ②가 **뽑아 준 낱말**을 받는다 (05 §4 · D-38).

    Args:
        surface: 발화에서 뽑은 **물질 표현 하나.** `''`·`'없음'` 은 미확정으로 본다
        species: 종. 주면 **그 종에 자료가 있는 이름으로만** 올린다 (D-58)

    Returns:
        `Resolution`. `assumption` 이 참이면 부르는 쪽이 `assumed_substance` 에
        남겨야 한다 — **밝히지 않은 추정은 환각이다** (D-59 ⑤).
    """
    s = (surface or "").strip()
    if not s or s == NONE:
        return Resolution(surface=s, name=None, how="없음")

    if s in known_substances():
        return Resolution(surface=s, name=s, how="직접")

    from .aliases import resolve as resolve_alias

    hits = resolve_alias(s, species)
    if hits:
        a = hits[0]  # 확정이 앞에 온다
        return Resolution(surface=s, name=a.substance, how="별칭", assumption=a.is_assumption)

    if len(s) < MIN_TERM_LEN:
        return Resolution(surface=s, name=None, how="없음")
    near = _near(s, species)
    if len(near) == 1:
        return Resolution(surface=s, name=near[0], how="부분일치")
    if near:
        return Resolution(surface=s, name=None, how="모호", candidates=near)
    return Resolution(surface=s, name=None, how="없음")


def _near(surface: str, species: str | None) -> tuple[str, ...]:
    """표면형을 품거나 표면형에 품히는 코퍼스 이름들. **낱말 경계로 본다** (D-60).

    부분 일치를 그냥 두면 두 가지가 샌다. 실측 (2026-08-02) —

        '카페인'      → 후보 10건.  그 안에 코퍼스 물질명 **`인`**(무기질)이 있었다.
                                  역방향 포함(`인` ⊂ `카페인`)이 만든 것이다
        '에틸렌글리콜'  → 후보 2건.   `에틸렌글리콜(부동액)` 과
                                  **`디에틸렌글리콜(브레이크액)`** — 다른 화합물이다

    그래서 별칭 표가 쓰는 것과 **같은 낱말 경계 규칙**을 여기서도 쓴다.
    `디`에틸렌글리콜은 앞이 한글이라 탈락하고, `인` 은 최소 길이에서 탈락한다.
    규칙을 새로 만들지 않는다 — **한 개념은 한 곳에서만 정의한다** (P2).

    ⚠️ `rules._match` 는 아직 이 경계를 안 본다. 같은 형태의 오탐이 거기에도 있다
    (`lookup('카페인')` 이 `인` 행에 걸린다). 이 회차에서는 손대지 않고 **기록만 한다** —
    규칙 테이블 조회를 바꾸면 골든셋 판정이 함께 움직여서, 별칭 재검토와 섞으면
    무엇이 무엇을 바꿨는지 못 가른다.
    """
    out: set[str] = set()
    for t in load_vocabulary():
        if species and species not in t.covers:
            continue
        if is_word_hit(t.substance, surface):  # 표면형이 코퍼스 이름 안에 **낱말로**
            out.add(t.substance)
        elif len(t.substance) >= MIN_TERM_LEN and is_word_hit(surface, t.substance):
            out.add(t.substance)  # 역방향 — 짧은 코퍼스 이름은 보지 않는다
    return tuple(sorted(out))


def candidates_for(species: str | None = None) -> tuple[str, ...]:
    """그 종에 **자료가 있는** 물질명. ② 슬롯 프롬프트의 선택지다 (D-59 ①).

    종을 주면 좁힌다 — 개 보호자에게 조류 전용 물질을 고르게 하면
    **근거 없는 종으로 답이 넘어간다** (D-58 · 별칭 표의 `프라이팬` 과 같은 이유).
    """
    if not species:
        return tuple(t.substance for t in load_vocabulary())
    return tuple(t.substance for t in load_vocabulary() if species in t.covers)
