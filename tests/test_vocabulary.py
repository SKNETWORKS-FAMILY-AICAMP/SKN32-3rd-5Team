"""물질 어휘 폐쇄 목록 — **목록이 코퍼스와 같은가**, 그리고 **밖이 정말 막히는가.**

D-59 ①은 *"물질 동정은 생성이 아니라 폐쇄 목록에서의 선택"* 으로 정했다.
그때 만든 것은 **프롬프트가 그렇게 부탁하는 것**까지였고, 부탁은 어길 수 있다.

여기서 보는 것은 둘이다.

    목록이 맞나    `물질어휘.csv` 는 생성물이다. 코퍼스가 바뀌면 **어긋난다**
    막히기는 하나   계약이 목록 밖 이름으로 응답을 만들지 못하는가 (D-40)
"""

from __future__ import annotations

import csv
from pathlib import Path

import pytest
from pydantic import ValidationError

from pettriage.app.contracts import (  # noqa: F401
    AskResponse,
    Citation,
    SubstanceName,
    TriageResult,
)
from pettriage.compute.vocabulary import (
    NONE,
    SPECIES,
    Term,
    UnknownSubstanceError,
    VocabularyMissingError,
    candidates_for,
    check_substance,
    covers_of,
    is_known,
    known_substances,
    load_vocabulary,
    resolve_substance,
)
from pettriage.ingest.facts_io import load_all
from pettriage.paths import find_root


@pytest.fixture(scope="module")
def corpus():
    root = find_root() or Path.cwd()
    return load_all(root / "data" / "facts")


class TestTableIsFresh:
    """**생성물이다.** 코퍼스가 바뀌었는데 표가 안 바뀌면 목록이 거짓말을 한다."""

    def test_표가_코퍼스와_같다(self, corpus):
        """`make vocab` 을 안 돌리고 사실 표만 고치면 여기서 걸린다.

        `정량임계치.csv` 에는 이 검사가 없어서 **어느 쪽이 진짜인지 알 수 없는 상태**가
        한 번 만들어졌다 (`build_rule_table.py` 머리말의 마카다미아 건).
        같은 일을 두 번 하지 않는다.
        """
        from_corpus = {(f.substance or "").strip() for f in corpus}
        from_corpus.discard("")
        from_table = {t.substance for t in load_vocabulary()}
        missing = sorted(from_corpus - from_table)
        extra = sorted(from_table - from_corpus)
        assert (
            not missing
        ), f"코퍼스에 있는데 표에 없다 ({len(missing)}종) — `make vocab`: {missing[:5]}"
        assert not extra, f"표에 있는데 코퍼스에 없다 ({len(extra)}종) — `make vocab`: {extra[:5]}"

    def test_종_표기가_코퍼스와_같다(self, corpus):
        want: dict[str, set[str]] = {}
        for f in corpus:
            want.setdefault((f.substance or "").strip(), set()).add(f.species)
        for t in load_vocabulary():
            assert set(t.species) == want[t.substance], t.substance

    def test_건수가_맞는다(self, corpus):
        n = sum(t.n_facts for t in load_vocabulary())
        assert n == len([f for f in corpus if (f.substance or "").strip()])

    def test_표를_사람이_읽을_수_있다(self):
        from pettriage.compute.vocabulary import _table_path

        with _table_path().open(encoding="utf-8-sig", newline="") as f:
            rows = list(csv.DictReader(f))
        assert rows
        for col in ("substance", "species", "n_facts"):
            assert col in rows[0], f"열 {col!r} 이 없다"


class TestClosedList:
    """목록의 성질."""

    def test_별칭_대상이_전부_목록_안이다(self):
        """`대파 → 알리움류` 의 `알리움류` 로 답할 수 있어야 한다.

        코퍼스는 `알리움류(양파·마늘·리크·차이브)` 로 적으므로 코퍼스 이름만으로는
        목록 밖이 된다. 별칭 표의 계열명을 합집합에 넣는 이유가 이것이고,
        **넣지 않으면 별칭이 걸린 순간 계약이 응답을 거부한다.**
        """
        from pettriage.compute.aliases import load_aliases

        for a in load_aliases():
            assert is_known(a.substance), f"별칭 대상 {a.substance!r} 가 폐쇄 목록 밖이다"

    def test_별칭_계열명이_목록을_넓힌다(self):
        """폐쇄 목록 = 코퍼스 물질명 ∪ 별칭 계열명. **둘 다 파일에서 온다.**"""
        corpus_names = {t.substance for t in load_vocabulary()}
        assert known_substances() >= corpus_names
        assert NONE not in known_substances(), "'없음' 은 목록의 항목이 아니라 선택지다"

    def test_없음은_정상_선택지다(self):
        """고를 것이 없다는 것은 실패가 아니다 (D-59 ④ · D-49)."""
        assert is_known(NONE)
        assert check_substance(NONE) == NONE

    @pytest.mark.parametrize(
        "name",
        [
            "일산화탄소",  # 그럴듯하지만 코퍼스에 없다 — D-59 ②가 든 오동정 예시
            "테플론 가스",
            "아세톤",
            "",
            "없음.",
            "알리움",  # 부분 문자열은 통과하지 않는다
        ],
    )
    def test_목록_밖은_거부된다(self, name):
        assert not is_known(name)
        with pytest.raises(UnknownSubstanceError):
            check_substance(name)

    def test_부분_일치로_통과하지_않는다(self):
        """`is_known` 은 **정확 일치**다.

        느슨하게 두면 `'아'` 같은 한 글자가 수백 개에 걸려 목록이 사실상 열린다.
        보호자 어휘와의 다리는 `별칭.csv` 가 놓는다 — 거기는 근거가 붙는다.
        """
        assert is_known("양파")
        assert not is_known("양")
        assert not is_known("양파를")


class TestNormalizer:
    """**거부기가 아니라 정규화기다.** 2026-08-02 재검토에서 방향을 바꿨다.

    처음에는 폐쇄 목록 밖이면 예외를 던지게 만들었다. 그런데 ②가 뽑는 것은
    보호자의 **표면형**이고, 실측하니 흔한 표현 30개 중 **12개가 목록 밖**이었다 —
    `커피`·`우유`·`이부프로펜`, 그리고 **`대파`**(골든셋 `G-039` 의 질의다).
    던지면 정상 질의가 계약에서 죽는다.
    """

    @pytest.mark.parametrize(
        ("surface", "species", "how", "name"),
        [
            ("초콜릿", "dog", "직접", "초콜릿"),
            ("양파", "dog", "직접", "양파"),
            ("대파", "cat", "별칭", "알리움류"),  # G-039
            ("무설탕껌", "dog", "별칭", "자일리톨"),
            ("자몽", "dog", "부분일치", "감귤류(레몬·자몽·라임)"),
            ("에틸렌글리콜", "dog", "부분일치", "에틸렌글리콜(부동액)"),
        ],
    )
    def test_표면형이_목록_위로_올라간다(self, surface, species, how, name):
        r = resolve_substance(surface, species)
        assert (r.how, r.name) == (how, name), r

    @pytest.mark.parametrize(
        ("surface", "species"),
        [("커피", "dog"), ("우유", "dog"), ("살서제", "dog")],
    )
    def test_둘_이상이면_고르지_않는다(self, surface, species):
        """하나를 고르면 나머지를 배제한 것이고, **그 배제가 곧 진단이다** (D-11 · D-49).

        후보는 버리지 않는다 — 검색어로 전부 넘기고 **LLM 이 다 읽게** 한다 (D-58).
        """
        r = resolve_substance(surface, species)
        assert r.how == "모호" and r.name is None
        assert len(r.candidates) >= 2, r.candidates

    @pytest.mark.parametrize(
        "surface", ["일산화탄소", "테플론 가스", "초코파이", "", "없음", "닭뼈"]
    )
    def test_없으면_없다고_한다(self, surface):
        r = resolve_substance(surface, "dog")
        assert r.name is None and r.how in ("없음", "모호")

    def test_예외를_던지지_않는다(self):
        """05 §6 — ②의 검증 실패는 **되묻기**다. 예외는 그 경로를 우회한다."""
        for s in ["일산화탄소", "", "!!", "커피", "가"]:
            assert resolve_substance(s, "dog").name is None

    def test_문장을_넣으면_조용히_덜_찾는다(self):
        """**이 함수는 낱말을 받는다.** 문장을 넣으면 예외 없이 덜 찾는다.

        역방향(코퍼스 이름 ⊂ 입력)만 걸려서 코퍼스 이름이 **글자 그대로** 들어 있는
        것만 잡힌다. 여기에 고정해 두는 이유는 **누군가 문장을 넣을 것**이기 때문이다 —
        그때 이 테스트가 그 사실을 말해 준다. 문장은 `aliases.substances_for` 가 받는다.
        """
        r = resolve_substance("고양이가 우유랑 초콜릿을 먹었어요", "cat")
        assert r.name == "초콜릿", r  # 우유는 조용히 빠진다
        assert resolve_substance("우유", "dog").how == "모호"  # 낱말로 넣으면 후보가 보인다

    @pytest.mark.parametrize(
        ("surface", "wrong"),
        [
            ("체리 씨", "체리"),
            ("체리씨", "체리"),
            ("복숭아 씨앗", "복숭아"),
            ("복숭아씨", "복숭아"),
        ],
    )
    def test_별칭이_부분일치보다_먼저다(self, surface, wrong):
        """**씨와 과육은 다른 것이다.** 씨는 시안화물이고 과육은 그 얘기가 아니다.

        `resolve_substance` 는 별칭 → 부분일치 순으로 돈다. 순서를 뒤집으면
        *"체리 씨를 삼켰어요"* 가 **`체리`(과육) 근거로 답한다** — 조용히, 그럴듯하게.

            체리 씨   별칭 O → 벚나무속 과일 씨      별칭 X → 체리   ← 과육이다

        2026-08-02 존재의의 재검토에서 **이 안전 속성을 지키는 것이 순서 하나뿐이고
        테스트가 없다**는 것이 드러났다. D-51(역치를 갈아끼우지 않는다)이 막으려던 것과
        같은 종류인데 여기는 잠겨 있지 않았다. 여기서 잠근다.
        """
        got = resolve_substance(surface, "dog")
        assert got.name == "벚나무속 과일 씨", got
        assert got.name != wrong

    def test_소독용_알코올이_주류_알코올로_안_떨어진다(self):
        """별칭이 없으면 `알코올` 과 `이소프로필알코올` 사이에서 **모호**해진다.

        둘은 다른 물질이다. 표가 그 갈림을 사람의 판단으로 미리 닫아 둔 자리다.
        """
        assert resolve_substance("소독용 알코올", "dog").name == "이소프로필알코올"
        assert resolve_substance("소주", "dog").name == "알코올"

    def test_추정_별칭은_그_사실이_따라온다(self):
        """`assumption` 이 참이면 부르는 쪽이 `assumed_substance` 에 남겨야 한다 (D-59 ⑤)."""
        assert resolve_substance("프라이팬", "bird").assumption is True
        assert resolve_substance("코팅팬", "bird").assumption is False

    def test_종을_넘지_않는다(self):
        assert resolve_substance("프라이팬", "bird").name == "PTFE"
        assert resolve_substance("프라이팬", "dog").name is None

    def test_한_글자_코퍼스_이름이_섞이지_않는다(self):
        """역방향 포함이 만든 오탐 — `'인'`(무기질) ⊂ `'카페인'`.

        `rules._match` 는 아직 이 경계를 안 본다. 같은 오탐이 거기에도 있고,
        이번 회차에서는 **기록만 한다** (`_near` 주석).
        """
        r = resolve_substance("카페인", "dog")
        assert "인" not in r.candidates, r.candidates

    def test_다른_화합물을_끌어오지_않는다(self):
        """`에틸렌글리콜` ⊂ `디에틸렌글리콜(브레이크액)` — **다른 화합물이다.**

        낱말 경계 규칙(D-60)이 앞의 `디` 에서 막는다. 규칙을 새로 만들지 않았다.
        """
        r = resolve_substance("에틸렌글리콜", "dog")
        assert r.name == "에틸렌글리콜(부동액)"

    def test_정규화_결과는_계약을_통과한다(self):
        """**이것이 이 설계의 요점이다** — 정규화를 거치면 계약이 안 걸린다.

        계약(`SubstanceName`)은 `_no_foreign_contacts` 와 같은 지위가 된다:
        정상 경로에서는 절대 걸리지 않고, 걸렸다면 정규화를 안 거쳤다는 뜻이다.
        """
        for surface in ["초콜릿", "대파", "자몽", "무설탕껌", "에틸렌글리콜"]:
            got = resolve_substance(surface, "dog") or resolve_substance(surface, "cat")
            if got.name:
                assert check_substance(got.name) == got.name


class TestSpeciesScope:
    """종을 넘겨 답하지 않는다 (D-58 · 별칭 표의 `프라이팬` 과 같은 이유)."""

    def test_covers_of(self):
        assert covers_of("all") == frozenset(SPECIES)
        assert covers_of("mammal") == frozenset({"dog", "cat"})
        assert covers_of("bird") == frozenset({"bird"})
        assert covers_of("모르는값") == frozenset()

    def test_후보를_종으로_좁힌다(self):
        every = candidates_for(None)
        for sp in SPECIES:
            got = candidates_for(sp)
            assert got, sp
            assert set(got) <= set(every)
        assert len(candidates_for("bird")) < len(every)

    def test_PTFE_는_조류_후보에만(self):
        ptfe = [t for t in load_vocabulary() if "PTFE" in t.substance]
        assert ptfe, "코퍼스에서 PTFE 가 사라졌다 — 별칭 표도 함께 본다"
        for t in ptfe:
            assert t.covers == frozenset({"bird"}), (t.substance, t.species)


class TestContractEnforces:
    """**말이 아니라 계약이 막는가** (D-40 — 지키기로 한 것이 아니라 못 어기는 것)."""

    def _ok(self, **kw):
        return {
            "status": "answered",
            "session_id": "s1",
            "answer": "테스트 답변입니다.",
            "triage": TriageResult(level=3, message="지금 병원에 연락하세요."),
            "citations": [Citation(source_id="S-001", publisher="p")],
            **kw,
        }

    def test_목록_밖_추정물질로는_응답이_안_만들어진다(self):
        with pytest.raises(ValidationError):
            AskResponse(
                **self._ok(answer="일산화탄소로 추정됩니다.", assumed_substance="일산화탄소")
            )

    def test_목록_안이면_가정을_밝힌_경우에만_통과한다(self):
        # 밝히면 통과
        AskResponse(
            **self._ok(
                answer="PTFE(테플론) 과열 흄으로 가정하고 안내합니다.",
                assumed_substance="PTFE(테플론) 과열 흄",
            )
        )
        # 안 밝히면 기존 검증기가 막는다 (D-59 ⑤)
        with pytest.raises(ValidationError):
            AskResponse(**self._ok(assumed_substance="PTFE(테플론) 과열 흄"))

    def test_동정물질도_목록_밖이면_거부된다(self):
        AskResponse(**self._ok(identified_substance="양파"))
        with pytest.raises(ValidationError):
            AskResponse(**self._ok(identified_substance="일산화탄소"))

    def test_별칭_계열명으로도_응답이_만들어진다(self):
        """`대파` 질의의 답은 `알리움류` 에 관한 것이다. 그것이 막히면 안 된다."""
        AskResponse(**self._ok(identified_substance="알리움류"))


class TestGraphDoor:
    """② 노드가 쓸 단일 문 (`graph.state.set_substance`)."""

    def test_목록_안은_들어간다(self):
        from pettriage.graph.state import set_substance

        assert set_substance({}, "양파")["substance"] == "양파"

    def test_없음과_None은_키를_안_둔다(self):
        """D-10 — 없는 값은 키를 두지 않는다. 되묻기가 그 뒤를 받는다."""
        from pettriage.graph.state import set_substance

        assert "substance" not in set_substance({"substance": "양파"}, None)
        assert "substance" not in set_substance({"substance": "양파"}, NONE)

    def test_목록_밖은_터지지_않고_키가_안_생긴다(self):
        """**예외가 아니라 되묻기다** (05 §6 · 02 §6).

        처음에는 `UnknownSubstanceError` 를 던지게 만들었다. 05 §6 은
        ①분류를 *"허용목록에 없으면 폴백 + 로그"*, ②슬롯을 *"검증 실패 시 되묻기"* 로
        정해 뒀고 02 §6 그래프도 `결측·물질미상 → ask_clarify` 로 그려 놨다.
        **예외는 그 경로를 우회한다.** 2026-08-02 재검토에서 뒤집었다 (D-57).
        """
        from pettriage.graph.state import set_substance

        assert "substance" not in set_substance({}, "일산화탄소")
        assert "substance" not in set_substance({}, "커피", "dog")  # 모호도 마찬가지다

    def test_표면형이_정규화되어_들어간다(self):
        """② 는 표면형을 뽑고, **정규화는 코드가 한다** (D-38)."""
        from pettriage.graph.state import set_substance

        assert set_substance({}, "대파", "cat")["substance"] == "알리움류"
        assert set_substance({}, "자몽", "dog")["substance"] == "감귤류(레몬·자몽·라임)"

    def test_종을_slots_에서_읽는다(self):
        from pettriage.graph.state import set_substance

        assert set_substance({"species": "bird"}, "프라이팬")["substance"] == "PTFE"
        assert "substance" not in set_substance({"species": "dog"}, "프라이팬")

    def test_추정이_슬롯에_남는다(self):
        """**추정이라는 사실을 여기서 잃으면 도약이 확정처럼 나간다** (D-59 ⑤).

        `프라이팬 → PTFE` 는 도약이다 — 무쇠·스테인리스 팬은 PTFE 를 내지 않는다.
        `resolve_substance` 는 그것을 알고 있었는데 `Slots` 에 담을 자리가 없어
        **버려지고 있었다** (2026-08-02 존재의의 재검토).

        `_assumption_must_be_stated` 는 그 도약을 막으려고 만든 계약인데,
        **필드를 채워 줄 경로가 끊겨 있어 발동하지 않았다** — 계약이 있어도
        아무도 안 부르면 없는 것과 같다 (D-40 · D-47 에서 한 번 겪었다).
        """
        from pettriage.graph.state import set_substance

        assumed = set_substance({"species": "bird"}, "프라이팬")
        assert assumed["substance"] == "PTFE"
        assert assumed["substance_is_assumed"] is True

        certain = set_substance({"species": "bird"}, "코팅팬")
        assert certain["substance"] == "PTFE"
        assert "substance_is_assumed" not in certain, "확정인데 추정 표시가 붙었다"

    def test_확정으로_바뀌면_추정_표시가_지워진다(self):
        """되묻기로 확정되면 표시가 남아 있으면 안 된다 — **남으면 없는 가정을 밝힌다.**"""
        from pettriage.graph.state import set_substance

        before = {"species": "bird", "substance": "PTFE", "substance_is_assumed": True}
        assert "substance_is_assumed" not in set_substance(before, "코팅팬")

    def test_원본_slots_를_안_바꾼다(self):
        from pettriage.graph.state import set_substance

        before = {"species": "dog"}
        after = set_substance(before, "양파")
        assert "substance" not in before and after["substance"] == "양파"


def test_표가_없으면_조용히_비지_않는다(tmp_path, monkeypatch):
    """빈 목록으로 넘어가면 검사가 전부 통과하거나 전부 실패한다. **둘 다 신호가 아니다** (D-58)."""
    from pettriage.compute import vocabulary as mod

    empty = tmp_path / "물질어휘.csv"
    empty.write_text("substance,species,n_facts\n", encoding="utf-8")
    monkeypatch.setattr(mod, "_table_path", lambda: empty)
    mod.load_vocabulary.cache_clear()
    mod.known_substances.cache_clear()
    try:
        with pytest.raises(VocabularyMissingError):
            mod.load_vocabulary()
    finally:
        mod.load_vocabulary.cache_clear()
        mod.known_substances.cache_clear()


def test_Term_은_불변이다():
    import dataclasses

    t = load_vocabulary()[0]
    assert isinstance(t, Term)
    with pytest.raises(dataclasses.FrozenInstanceError):
        t.substance = "바꿀 수 없다"  # type: ignore[misc]


# ── 종은 물질이 아니다 (D-67) ─────────────────────────────────
class TestSpeciesIsNotASubstance:
    """2026-08-02 1차 평가에서 드러난 오탐. 코퍼스 이름의 **종 한정어**가
    물질 부품으로 떨어져 나왔다 — `췌장염(고양이)` → 부품 `고양이`.
    """

    def test_species_word_is_not_a_mention(self):
        from pettriage.compute.vocabulary import mention_in

        # 도메인 밖 질의가 "물질을 말했다" 로 잡혀 D-46 의 범위밖 방어를 통과했다.
        assert mention_in("고양이 캣타워 추천 좀 해주세요", assumptions=False) is None

    def test_species_word_does_not_shadow_a_real_substance(self):
        """오탐 하나가 **정탐 하나를 지웠다.**

        `고양이` 가 먼저 잡혀서 코퍼스에 실제로 있는 `향초·왁스멜트·인센스` 를
        못 찾았다. 첫 매칭을 반환하던 것도 함께 고쳤다 — 이제 가장 긴 것을 고른다.
        """
        from pettriage.compute.vocabulary import mention_in

        assert mention_in("고양이가 향초를 핥았는데 위험할까요", assumptions=False) == "향초"

    def test_species_cannot_enter_the_closed_list(self):
        """②가 종을 물질로 올려도 **문에서 막힌다** (D-59)."""
        from pettriage.compute.vocabulary import resolve_substance

        for w in ("고양이", "강아지", "앵무새", "조류", "반려동물"):
            assert resolve_substance(w, None).name is None, w

    def test_real_substances_still_resolve(self):
        """막느라 정상 물질을 죽이지 않았다."""
        from pettriage.compute.vocabulary import mention_in

        for q, want in (
            ("고양이가 백합을 씹었어요", "백합"),
            ("강아지가 초콜릿을 먹었어요", "초콜릿"),
            ("고양이한테 우유를 매일 조금씩 줘도 되나요", "우유"),
        ):
            assert mention_in(q, assumptions=False) == want, q

    def test_species_vocabulary_has_one_source(self):
        """`slots.py` 가 따로 들고 있어서 어휘 쪽에서 걸러낼 수 없었다 (P2)."""
        from pettriage.compute.vocabulary import SPECIES_WORDS
        from pettriage.graph.nodes.slots import _SPECIES_KEYWORDS

        assert _SPECIES_KEYWORDS is SPECIES_WORDS
