"""별칭 표 — **표가 코퍼스에 뿌리를 두는가**, 그리고 **엉뚱한 데 걸리지 않는가.**

`정량임계치.csv` 와 달리 이 표는 **사람이 쓴다.** 생성물이 아니므로
사실 표가 바뀌어도 자동으로 따라오지 않는다. 그래서 여기서 대조한다.

    별칭이 가리키는 물질이 코퍼스에서 사라지면 → 그 별칭은 **조용히 아무것도 안 한다.**
    `lookup` 은 0건을 돌려주고, 그것이 *"근거가 없다"* 로 읽힌다.

**없는 것을 가리키는 별칭은 있는 것보다 나쁘다** — 있으면 고치고, 없으면 못 고친다.

2026-08-02 재검토가 남긴 것
--------------------------
1차 표의 검사는 근거 id 의 **실재**만 봤다. 그래서 셋 다 초록인 채로

    술 → 알코올        근거 `F-029-002` = **아보카도**
    살구씨 → 핵과 씨앗  근거 `F-029-016` = **위장관 폐색 이물**

가 통과했고, `살구씨` 는 **근거가 조류 자료뿐인데 전 종에 걸려** 있었다.
그래서 여기서 세 가지를 더 본다 — **관련성 · 종 범위 · 오탐** (D-58 · D-60 · D-61).
"""

from __future__ import annotations

import csv
import dataclasses
from pathlib import Path

import pytest

from pettriage.compute.aliases import (
    KINDS,
    MIN_ALIAS_LEN,
    SPECIES,
    Alias,
    is_word_hit,
    load_aliases,
    resolve,
    substances_for,
)
from pettriage.compute.rules import lookup
from pettriage.ingest.facts_io import load_facts
from pettriage.paths import find_root

#: 코퍼스의 `species` 값이 **실제로 덮는 종.** `all`·`mammal` 은 종이 아니라 묶음이다.
_COVERS: dict[str, frozenset[str]] = {
    "all": frozenset(SPECIES),
    "mammal": frozenset({"dog", "cat"}),
    "dog": frozenset({"dog"}),
    "cat": frozenset({"cat"}),
    "bird": frozenset({"bird"}),
}


@pytest.fixture(scope="module")
def corpus():
    root = find_root() or Path.cwd()
    return load_facts(root / "data" / "facts" / "facts_ohb.csv")


@pytest.fixture(scope="module")
def by_id(corpus):
    return {f.fact_id: f for f in corpus}


class TestTableIsGrounded:
    """표의 모든 행이 **코퍼스에 뿌리를 두는가.**"""

    def test_대상_물질이_코퍼스에_실재한다(self, corpus):
        substances = {f.substance for f in corpus}
        for a in load_aliases():
            hit = [s for s in substances if a.substance in s]
            assert hit, f"{a.alias!r} → {a.substance!r} 가 코퍼스 어디에도 없다"

    def test_근거_fact_id_가_실재한다(self, by_id):
        for a in load_aliases():
            assert a.basis, f"{a.alias!r} 에 근거(basis)가 없다 — 어느 사실이 이 매핑을 받치나"
            missing = [b for b in a.basis if b not in by_id]
            assert not missing, f"{a.alias!r} 의 근거가 코퍼스에 없다: {missing}"

    def test_근거가_대상_물질과_관련_있다(self, by_id):
        """**실재하는 것만으로는 부족하다** (D-61).

        1차 표는 `술 → 알코올` 의 근거를 `F-029-002`(아보카도)로,
        `살구씨 → 핵과 씨앗` 의 근거를 `F-029-016`(위장관 폐색 이물)로 적어 두었다.
        둘 다 실재하는 id 였으므로 예전 검사는 초록이었다.

        `basis` 의 정의는 *"대상 물질이 그 이름으로 실재함을 보이는 사실 행"* 이다.
        그러면 근거 행의 `substance` 안에 대상 물질명이 **들어 있어야** 한다.
        """
        for a in load_aliases():
            for b in a.basis:
                got = by_id[b].substance
                assert (
                    a.substance in got
                ), f"{a.alias!r} → {a.substance!r} 의 근거 {b} 가 다른 물질이다: {got!r}"

    def test_근거가_이_행이_적용되는_종을_덮는다(self, by_id):
        """**근거 없는 종에 걸리지 않는다** (D-58 의 연장).

        1차 표의 `살구씨` 는 `species` 가 비어 전 종이었는데, 대상 `핵과 씨앗` 의
        근거는 `F-005-005`·`F-091-005` **둘 다 조류**였다. 개 보호자에게 줄 근거가
        없는데 걸렸다. `프라이팬` 을 `bird` 로 좁힌 것과 같은 문제인데 거기만
        좁혀져 있었다.
        """
        for a in load_aliases():
            covered: set[str] = set()
            for b in a.basis:
                covered |= _COVERS[by_id[b].species]
            missing = a.covers - covered
            assert not missing, (
                f"{a.alias!r} 는 {sorted(a.covers)} 에 걸리는데 "
                f"근거가 덮는 종은 {sorted(covered)} 다 — {sorted(missing)} 에 줄 근거가 없다"
            )

    def test_kind_는_확정_이나_추정_뿐이다(self):
        for a in load_aliases():
            assert a.kind in KINDS, f"{a.alias!r} 의 kind={a.kind!r}"

    def test_추정에는_무엇을_가정하는지_적혀_있다(self):
        """추정이면 **무엇을 가정했는지**가 표에 있어야 한다 (D-59 ⑤).

        가정을 안 적으면 나중에 그 매핑을 보는 사람이 확정으로 읽는다.
        """
        for a in load_aliases():
            if a.is_assumption:
                assert a.note.strip(), f"{a.alias!r} 는 추정인데 note 가 비었다"

    def test_별칭이_중복되지_않는다(self):
        seen = [a.alias for a in load_aliases()]
        assert len(seen) == len(set(seen)), f"중복 별칭: {[x for x in seen if seen.count(x) > 1]}"

    def test_한글자_별칭은_없다(self):
        """**1글자는 조사와 구별되지 않는다** (D-60).

        `파` 는 낱말 경계 규칙을 넣어도 `파도`·`파는`(팔다)에 걸린다.
        규칙으로 못 막는 것은 표에 두지 않는다 — `load_aliases` 도 거부한다.
        """
        short = [a.alias for a in load_aliases() if len(a.alias) < MIN_ALIAS_LEN]
        assert not short, f"1글자 별칭: {short}"

    def test_한글자_별칭을_넣으면_표가_안_열린다(self, tmp_path, monkeypatch):
        from pettriage.compute import aliases as mod

        bad = tmp_path / "별칭.csv"
        bad.write_text(
            "alias,substance,kind,species,basis,note\n파,알리움류,확정,,F-014-001,\n",
            encoding="utf-8",
        )
        monkeypatch.setattr(mod, "_table_path", lambda: bad)
        mod.load_aliases.cache_clear()
        try:
            with pytest.raises(mod.AliasTableInvalidError, match="1글자"):
                mod.load_aliases()
        finally:
            mod.load_aliases.cache_clear()


class TestLongestFirst:
    """긴 별칭이 먼저 걸려야 근거가 정확해진다."""

    def test_정렬이_길이_내림차순이다(self):
        lens = [len(a.alias) for a in load_aliases()]
        assert lens == sorted(lens, reverse=True)

    def test_천도복숭아씨가_복숭아씨보다_먼저_걸린다(self):
        got = resolve("천도복숭아씨를 삼켰어요", "dog")
        assert got and got[0].alias == "천도복숭아씨", [a.alias for a in got]


class TestWordBoundary:
    """별칭이 **다른 낱말 안에 묻혀** 걸리면 안 된다 (D-60).

    1차 표는 부분 문자열로 걸어 **오탐 14건**을 냈다. 그 전부를
    걸려야 하는 것들과 같은 자리에 둔다 — **한쪽만 보면 다른 쪽이 무너진다.**
    """

    #: (문장, 별칭, 걸려야 하나)
    CASES = [
        # ── 걸려야 한다 ──────────────────────────────
        ("대파를 먹었어요", "대파", True),
        ("쪽파 먹였어요", "쪽파", True),
        ("대파랑 양파 같이 먹었어요", "대파", True),
        ("냄비, 프라이팬 다 태웠어요", "프라이팬", True),
        ("소주를 핥았어요", "소주", True),
        ("맥주 한 모금", "맥주", True),
        ("무설탕껌을 삼켰어요", "무설탕껌", True),
        ("자두씨는 어떤가요", "자두씨", True),
        ("테팔 팬 과열", "테팔", True),
        ("파김치도 위험한가요", "파김치", True),
        # ── 1차 표가 잘못 걸던 자리 ────────────────────
        ("초코파이를 먹었어요", "파김치", False),
        ("양파를 먹었어요", "대파", False),
        ("파스타를 먹었어요", "파김치", False),
        ("파마산 치즈를 먹었어요", "파김치", False),
        ("파파야를 먹어도 되나요", "파김치", False),
        ("파운드케이크", "파김치", False),
        ("중성화 수술 후에 안 먹어요", "소주", False),
        ("수술 자국이 부었어요", "소주", False),
        ("대파하듯 지쳐요", "대파", False),
        ("소주방 근처에서", "소주", False),
        ("서양주목을 씹었어요", "양주", False),
        ("압력냄비를 태웠어요", "냄비", False),
    ]

    @pytest.mark.parametrize(("text", "alias", "want"), CASES)
    def test_경계(self, text, alias, want):
        assert is_word_hit(text, alias) is want, f"{alias!r} in {text!r}"

    def test_표_전체로_돌려도_오탐이_없다(self):
        """`is_word_hit` 단위가 아니라 **표 전체**로 돌린다.

        단위로 통과해도 다른 행이 걸리면 결과는 같다 —
        *"초코파이 → 알리움류"* 는 `파` 행이 만든 것이었다.
        """
        clean = [
            "초코파이를 먹었어요",
            "양파를 먹었어요",
            "파스타를 먹었어요",
            "파마산 치즈",
            "파파야를 먹어도 되나요",
            "파운드케이크",
            "파슬리를 먹었어요",
            "파프리카",
            "파인애플",
            "파란 토사물",
            "파도 소리가 들려요",
            "파는 곳이 어디예요",
            "중성화 수술 후에 안 먹어요",
            "수술 자국이 부었어요",
            "미용 기술이 좋은 곳",
            "마취술 후 회복",
            "예술적인 미용",
            "시술 부위가 부었어요",
            "술기운",
            "소주방 근처에서",
            "서양주목을 씹었어요",
            # 지명 — 경계 규칙으로는 못 막는다. **표에서 뺐다** (2026-08-02 재검토)
            "양주에 사는데 병원 추천해 주세요",
            "청주에서 갈 만한 동물병원 있나요",
            "충북 청주 24시 병원",
            "와인색 러그를 씹었어요",
            "캣타워 추천해 주세요",
            "반려동물 보험료 계산",
            "오늘 날씨 어때요",
        ]
        bad = {t: substances_for(t, None) for t in clean if substances_for(t, None)}
        assert not bad, f"오탐: {bad}"


class TestRealSentences:
    """**낱말이 아니라 문장으로** 부딪힌다.

    별칭 단위로 통과해도 실제 문장에서는 다르게 걸린다. 여기 있는 것은 전부
    2026-08-02 재검토에서 **실제로 실패했던** 문장이다 — 표기 변형(`체리 씨앗`)과
    합성어(`논스틱팬`)가 전부 0건이었다.
    """

    CASES = [
        ("5kg 고양이가 대파를 40g쯤 뜯어 먹었어요", "cat", "알리움류"),
        ("강아지가 파김치를 한 입 먹었어요", "dog", "알리움류"),
        ("고양이가 쪽파 다듬다 흘린 걸 먹었어요", "cat", "알리움류"),
        ("강아지가 소주를 핥았어요", "dog", "알코올"),
        ("고양이가 와인 쏟은 걸 핥았어요", "cat", "알코올"),
        ("강아지가 생맥주를 핥았어요", "dog", "알코올"),
        ("강아지가 손소독제를 핥았어요", "dog", "이소프로필알코올"),
        ("강아지가 소독용 알코올을 핥았어요", "dog", "이소프로필알코올"),
        ("강아지가 무설탕 껌을 삼켰어요", "dog", "자일리톨"),
        ("강아지가 무설탕껌 하나를 먹었어요", "dog", "자일리톨"),
        ("고양이가 구강청결제를 마셨어요", "cat", "자일리톨"),
        ("앵무새 앞에서 프라이팬을 태웠어요", "bird", "PTFE"),
        ("앵무새가 논스틱팬 연기를 마셨어요", "bird", "PTFE"),
        ("앵무새 방에서 디퓨저를 켜뒀어요", "bird", "에센셜 오일"),
        ("강아지가 자두씨를 삼켰어요", "dog", "벚나무속 과일 씨"),
        ("강아지가 자두 씨를 삼켰어요", "dog", "벚나무속 과일 씨"),
        ("강아지가 체리 씨앗을 먹었어요", "dog", "벚나무속 과일 씨"),
        ("강아지가 복숭아 씨앗을 삼켰어요", "dog", "벚나무속 과일 씨"),
        ("강아지가 살구씨앗을 먹었어요", "dog", "벚나무속 과일 씨"),
    ]

    @pytest.mark.parametrize(("q", "species", "want"), CASES)
    def test_문장에서_걸린다(self, q, species, want):
        assert want in substances_for(q, species), q

    def test_별칭_하나하나가_조사와_붙어도_걸린다(self):
        """표의 **모든 행**을 흔한 조사와 붙여 본다.

        경계 규칙이 조사를 하나 빠뜨리면 그 형태의 질의가 통째로 죽는데,
        표본 몇 개로는 안 드러난다.
        """
        tails = [
            "을 먹었어요",
            "를 먹었어요",
            "이 위험한가요",
            "가 위험한가요",
            " 먹었어요",
            "은 어때요",
            "는 어때요",
            "도 안되나요",
            "이랑 같이 먹었어요",
            "?",
        ]
        dead = [
            a.alias + t
            for a in load_aliases()
            for t in tails
            if a.substance not in substances_for(a.alias + t, None)
        ]
        assert not dead, f"조사가 붙으면 죽는 형태: {dead[:15]}"


class TestSpeciesScope:
    """종이 다른 별칭은 걸리지 않는다."""

    def test_프라이팬은_조류에만(self):
        """코퍼스의 PTFE 자료가 전부 `bird` 다. 개 보호자에게 줄 근거가 없다."""
        assert substances_for("프라이팬을 태웠어요", "bird") == ["PTFE"]
        assert substances_for("프라이팬을 태웠어요", "dog") == []

    def test_대파는_개와_고양이에만(self):
        """`알리움류` 라는 이름을 쓰는 근거가 `dog`·`cat` 뿐이다.

        새에게도 양파·마늘 자료는 있지만(`양파·마늘(Allium)`) **그 이름이 아니다.**
        `알리움류` 로 검색해서는 안 걸린다 — 없는 근거를 있다고 하지 않는다.
        """
        assert substances_for("대파를 먹었어요", "cat") == ["알리움류"]
        assert substances_for("대파를 먹었어요", "dog") == ["알리움류"]
        assert substances_for("대파를 먹었어요", "bird") == []

    def test_종을_안_주면_전부_본다(self):
        assert "PTFE" in substances_for("프라이팬을 태웠어요", None)


class TestConfirmedBeforeAssumption:
    """`확정` 이 `추정` 보다 앞에 온다 — 부르는 쪽이 확실한 것을 먼저 쓴다."""

    def test_순서(self):
        got = resolve("코팅팬 프라이팬", "bird")
        kinds = [a.kind for a in got]
        assert kinds == sorted(kinds, key=lambda k: k == "추정"), kinds

    def test_확정_우선이_길이_우선보다_먼저다(self):
        """대상이 같으면 하나만 남는데, **무엇이 남느냐**가 근거를 바꾼다.

        길이순으로만 돌면 `프라이팬`(4글자·추정)이 `코팅팬`(3글자·확정)을 밀어낸다.
        그러면 확정으로 답할 수 있는 자리에 **가정이 붙는다** (D-59 ⑤).
        2026-08-02 재검토에서 잡았다.
        """
        got = resolve("프라이팬으로 코팅팬을 덮었어요", "bird")
        assert [a.alias for a in got] == ["코팅팬"], [(a.alias, a.kind) for a in got]
        assert got[0].kind == "확정"

    def test_테팔은_추정이다(self):
        """브랜드명이다. 같은 브랜드가 무쇠·스테인리스 라인도 판다.

        1차 표는 `확정` 으로 뒀다 — 코퍼스 밖 상식을 확정으로 적은 것이다.
        """
        got = {a.kind for a in load_aliases() if a.alias == "테팔"}
        assert got == {"추정"}, got

    def test_살구씨는_추정이다(self):
        """코퍼스 목록(체리·천도복숭아·복숭아·자두)에 **살구가 없다.**

        살구(Prunus armeniaca)가 벚나무속인 것은 분류학적 사실이지만
        코퍼스가 명시하지 않는다. 명시하지 않은 것은 확정으로 적지 않는다 (D-38).
        """
        got = {(a.kind, a.substance) for a in load_aliases() if a.alias == "살구씨"}
        assert got == {("추정", "벚나무속 과일 씨")}, got


class TestRulesIntegration:
    """`lookup` 이 별칭을 쓰되 **직접 일치를 가로채지 않는가.**"""

    def test_G_039_대파_고양이가_풀린다(self):
        """골든셋 `G-039` 가 실패하던 원인이 이 별칭 결손이었다.

        `rules.lookup` 은 부분 문자열로 찾는데 `대파` 는
        `알리움류(양파·마늘·리크·차이브)` 에 걸리지 않는다.
        """
        got = lookup("대파", "cat")
        assert got, "대파/cat 이 여전히 0건이다"
        assert got[0].fact_id == "F-014-001"

    def test_직접_일치가_우선이다(self):
        """표에 이름이 그대로 있으면 별칭이 가로채지 않는다."""
        assert [r.fact_id for r in lookup("양파", "dog")] == ["F-034-001"]
        assert [r.fact_id for r in lookup("초콜릿", "dog")] == [
            "F-034-020",
            "F-034-021",
            "F-034-022",
        ]

    def test_역치를_갈아끼우지_않는다(self):
        """`대파 → 양파` 로 두면 적중은 늘지만 **다른 종의 역치를 빌려 오는 것**이다.

        대파(Allium fistulosum)와 양파(A. cepa)의 티오설페이트 함량이 같다는 근거가 없다.
        속 수준(`알리움류`)으로 두면 dog 은 계산 가능한 역치가 없어 정성 답변으로
        내려간다 — 그것이 정직한 결과다 (D-46 · D-51).
        """
        target = {a.substance for a in load_aliases() if a.alias == "대파"}
        assert target == {"알리움류"}, target
        assert lookup("대파", "dog") == [], "dog 에 없는 역치가 생겼다면 별칭이 종을 넘었다"


class TestNoMatchIsSilent:
    """모르는 말에는 **아무것도 지어내지 않는다.**"""

    @pytest.mark.parametrize("text", ["", "오늘 날씨 어때요", "캣타워 추천해 주세요"])
    def test_빈_결과(self, text):
        assert resolve(text, "cat") == []

    def test_알_수_없는_별칭은_예외가_아니라_빈_결과다(self):
        assert isinstance(resolve("존재하지않는물질", "dog"), list)


def test_표를_사람이_읽을_수_있다():
    """CSV 가 열리고 필수 열이 있는가. **표는 사람이 쓰는 것이다.**"""
    from pettriage.compute.aliases import _table_path

    with _table_path().open(encoding="utf-8-sig", newline="") as f:
        rows = list(csv.DictReader(f))
    assert rows
    for col in ("alias", "substance", "kind", "species", "basis", "note"):
        assert col in rows[0], f"열 {col!r} 이 없다"


def test_Alias_는_불변이다():
    a = load_aliases()[0]
    assert isinstance(a, Alias)
    with pytest.raises(dataclasses.FrozenInstanceError):
        a.substance = "바꿀 수 없다"  # type: ignore[misc]
