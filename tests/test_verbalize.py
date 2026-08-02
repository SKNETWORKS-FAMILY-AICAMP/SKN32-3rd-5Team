"""문장화 — 결측 필드가 절을 통째로 생략하는지, LLM 없이 결정론적인지.

D-38의 핵심 두 가지를 테스트가 지킨다.
"""

from __future__ import annotations

import pytest

from pettriage.ingest import templates
from pettriage.ingest.verbalize import to_chunk, verbalize
from pettriage.schemas import Fact
from pettriage.triage.levels import FeedingLevel


def _dog_chocolate(**over) -> Fact:
    base = dict(
        fact_id="F-001",
        source_id="S-034",
        publisher="Frontiers in Veterinary Science",
        doc_type="toxicity_food",
        species="dog",
        substance="초콜릿(테오브로민)",
        threshold_type="임상징후 발현",
        dose="20",
        unit="mg/kg",
        feeding_level=FeedingLevel.NEVER,
        effect="경증 임상징후",
        signs=["구토", "다음", "안절부절"],
        onset="2–4시간",
    )
    base.update(over)
    return Fact(**base)


class TestQuantitativeClause:
    def test_dose_present_renders_quantitative_sentence(self):
        text = verbalize(_dog_chocolate())
        assert "20mg/kg 이상 섭취 시" in text
        assert "출처: Frontiers in Veterinary Science, S-034" in text

    def test_weight_based_unit_is_not_doubled(self):
        """`mg/kg` 은 이미 체중당이다. "체중 1kg당"을 덧붙이면 수치가 왜곡된다."""
        text = verbalize(_dog_chocolate())
        assert "체중 1kg당 20mg/kg" not in text

    def test_기준을_모르면_기준을_붙이지_않는다(self):
        """⚠️ 이 테스트는 2026-08-02 에 **정반대로 뒤집혔다.**

            # 예전
            def test_absolute_unit_gets_per_weight_prefix(self):
                '''체중당 단위가 아니면 "체중 1kg당"을 붙여야 의미가 산다.'''
                text = verbalize(_dog_chocolate(dose="2.3", unit="g"))
                assert "체중 1kg당 2.3g 이상 섭취 시" in text

        `unit="g"` 로만 검사해서 그럴듯해 보였지만, `_dose_phrase` 는 `/kg` 이 없으면
        **무조건** `"체중 1kg당"` 을 붙이고 있었다. 실제 코퍼스에서 2건이 왜곡됐다 —
        `"0.5% of body weight"` 가 `"체중 1kg당 0.5%"` 로, `"8.2kg 개가 포도 4-5알"` 이
        `"체중 1kg당 4-5알"`(약 8배)로 나갔다.

        **테스트가 왜곡을 정답으로 고정하고 있었다.** 지금은 기준을 모르면
        수치만 말한다 — 틀린 기준보다 낫다 (D-38).
        """
        text = verbalize(_dog_chocolate(dose="2.3", unit="g"))
        assert "2.3g 이상 섭취 시" in text
        assert "체중 1kg당" not in text

    def test_출처가_밝힌_기준은_그대로_쓴다(self):
        """`basis` 칸에 기준이 있으면 **그것을** 붙인다. 만들지 않는다."""
        text = verbalize(_dog_chocolate(dose="0.5", unit="%", basis="체중 대비"))
        assert "0.5%(체중 대비) 이상 섭취 시" in text

    def test_코퍼스에_지어낸_기준이_남아있지_않다(self):
        """전수 회귀 — 888청크 어디에도 `체중 1kg당` 이 없어야 한다.

        이 절은 **출처에 없는 말**이다. 한 건이라도 있으면 D-38 위반이다.
        """
        from pathlib import Path

        from pettriage.ingest.facts_io import load_facts
        from pettriage.paths import find_root

        root = find_root() or Path.cwd()
        facts = load_facts(root / "data" / "facts" / "facts_ohb.csv")
        bad = [f.fact_id for f in facts if "체중 1kg당" in verbalize(f)]
        assert not bad, bad

    def test_파이프_구분자가_문장으로_새지_않는다(self):
        """`|` 는 CSV 직렬화 구분자이지 문장 부호가 아니다.

        `toxic_part`·`effect` 가 `LIST_FIELDS` 에 없어서 그대로 나갔다 —
        *"잎|꽃잎|꽃가루|꽃병 물이 독성 부위다"* 가 18청크에서 실측됐다.
        """
        from pathlib import Path

        from pettriage.ingest.facts_io import load_facts
        from pettriage.paths import find_root

        root = find_root() or Path.cwd()
        facts = load_facts(root / "data" / "facts" / "facts_ohb.csv")
        bad = [f.fact_id for f in facts if "|" in verbalize(f)]
        assert not bad, bad

    def test_조사를_병기하지_않는다(self):
        """`아보카도은(는)` 같은 표기는 검색 임베딩과 가독성을 함께 해친다.

        `_josa` 기계가 이미 있었는데 네 자리가 하드코딩이었다 —
        `안전로`(→안전으로) 7건 · `과(와)` 15건 · `g가 권장` 142건 · `g다.` 9건.
        """
        from pathlib import Path

        from pettriage.ingest.facts_io import load_facts
        from pettriage.paths import find_root

        root = find_root() or Path.cwd()
        facts = load_facts(root / "data" / "facts" / "facts_ohb.csv")
        both = ("은(는)", "이(가)", "과(와)", "와(과)", "다(이다)", "안전로")
        bad = [(f.fact_id, p) for f in facts for p in both if p in verbalize(f)]
        assert not bad, bad[:10]

    def test_missing_dose_omits_the_clause_entirely(self):
        """조류는 정량 임계치가 0건 → 정량 절이 자동으로 사라진다."""
        text = verbalize(_dog_chocolate(dose=None, unit=None))
        assert "이상 섭취 시" not in text
        assert "정보 없음" not in text  # 빈 값을 문장으로 만들지 않는다
        assert "주요 증상은" in text  # 다른 절은 살아 있다


class TestThresholdTypeGate:
    """`증례 보고 범위` 는 역치가 아니다 — 역치 문장으로 만들면 안 된다 (schemas.py)."""

    def test_reported_range_is_not_stated_as_a_threshold(self):
        text = verbalize(
            _dog_chocolate(
                substance="포도",
                threshold_type="증례 보고 범위",
                dose="3",
                unit="g/kg",
                effect="급성 신부전",
            )
        )
        assert "이상 섭취 시" not in text, "증례 보고 범위를 역치로 주장하면 안 된다"
        assert "증례 보고에서" in text

    def test_no_threshold_type_produces_no_quantitative_sentence(self):
        """성격이 확인되지 않은 수치는 문장으로 만들지 않는다."""
        text = verbalize(_dog_chocolate(threshold_type=None))
        assert "이상 섭취 시" not in text

    def test_missing_feeding_level_does_not_invent_a_grade(self):
        """등급이 없으면 "주의 대상" 같은 기본값을 지어내지 않는다."""
        text = verbalize(_dog_chocolate(feeding_level=None))
        assert "주의 대상" not in text
        assert "분류된다" not in text


class TestBirdBecomesQualitative:
    def test_bird_fact_has_no_quantitative_sentence(self):
        bird = _dog_chocolate(
            fact_id="F-002",
            source_id="S-005",
            publisher="Lafeber Vet",
            species="bird",
            substance="아보카도",
            dose=None,
            unit=None,
            threshold_type=None,
            onset="12시간",
        )
        text = verbalize(bird)
        assert text.startswith("앵무새에게 아보카도")
        assert "체중 1kg당" not in text


class TestNoThresholdSubstance:
    """포도 — 용량-반응이 성립하지 않는다고 원문이 명시한 경우."""

    def test_renders_explicit_no_threshold_sentence(self):
        grape = _dog_chocolate(
            fact_id="F-003",
            substance="포도·건포도",
            threshold_type="역치 없음",
            dose=None,
            unit=None,
            signs=["구토", "식욕부진"],
            onset="24시간",
        )
        text = verbalize(grape)
        assert "안전한 최소 섭취량이 확립되어 있지 않아" in text
        assert "체중 1kg당" not in text


class TestDeterminism:
    def test_same_fact_yields_identical_text(self):
        """축① — 같은 입력에 항상 같은 출력. LLM이면 성립하지 않는다."""
        f = _dog_chocolate()
        assert len({verbalize(f) for _ in range(20)}) == 1


class TestChunk:
    def test_route2_chunk_has_no_quote(self):
        """경로 ②는 원문을 담지 않는다 (D-37)."""
        chunk = to_chunk(_dog_chocolate())
        assert chunk.route == "사실추출"
        assert chunk.quote is None
        assert chunk.source_id == "S-034"  # 역추적은 source_id로
        assert chunk.fact_ids == ["F-001"]


# ── 조사 선택 (D-38 — 문장화도 검증 대상) ──────────────────


class TestJosa:
    """벡터DB에 들어가는 문장이자 **사용자가 읽는 문장**이다.

    `아보카도은(는)` 같은 표기는 검색 임베딩과 가독성을 함께 해친다.
    """

    @pytest.mark.parametrize(
        ("word", "expected"),
        [
            ("백합", "백합은"),
            ("아보카도", "아보카도는"),
            ("초콜릿", "초콜릿은"),
            ("포도", "포도는"),
            # 끝의 괄호는 조사 판단에서 제외한다 — 조사는 그 앞의 말로 고른다
            ("주목(Yew)", "주목(Yew)은"),
            ("사람용 진통제(이부프로펜)", "사람용 진통제(이부프로펜)는"),
            ("저철분 식이(로리·투칸)", "저철분 식이(로리·투칸)는"),
            # 알파벳도 읽는 소리로 고른다 — 자일리톨(L=엘)은 받침으로 끝난다
            ("Xylitol", "Xylitol은"),
            ("Vitamin A", "Vitamin A는"),
            # 숫자로 끝나면 읽는 소리로 — 삼은 받침이 있다
            ("Ω-3", "Ω-3은"),
            # 그 밖의 기호로 끝나면 그때만 병기한다 — 틀린 조사를 붙이지 않는다
            ("Ω", "Ω은(는)"),
        ],
    )
    def test_eun_neun(self, word: str, expected: str) -> None:
        assert templates._eun(word) == expected

    @pytest.mark.parametrize(
        ("word", "expected"),
        [
            ("반려동물", "반려동물과"),
            ("개", "개와"),
            ("고양이", "고양이와"),
            ("앵무새", "앵무새와"),
        ],
    )
    def test_wa_gwa(self, word: str, expected: str) -> None:
        """`반려동물와 부동액에 관한 자료다` 가 실제로 나갔다 — 하드코딩된 `와` 였다."""
        assert templates._wa(word) == expected

    def test_ga(self) -> None:
        assert templates._ga("급성 신부전") == "급성 신부전이"
        assert templates._ga("구토") == "구토가"

    def test_rendered_sentence_has_no_literal_placeholder(self) -> None:
        """한글 물질명이면 `은(는)` 병기가 문장에 남지 않아야 한다."""
        text = verbalize(_dog_chocolate(substance="아보카도", threshold_type="", dose="", unit=""))
        assert "아보카도는" in text
        assert "은(는)" not in text


class TestComposition:
    """`성분 함량` 은 **권장량도 섭취 역치도 아니다** (D-38 층 0).

    구분하지 않으면 "어류기름 최소 100.71%가 권장된다" 같은 문장이 나온다 —
    원문에 없는 주장이고 그대로 벡터DB에 들어가면 그 자체가 환각의 출처다.
    """

    def test_nutrition_composition_is_not_a_recommendation(self) -> None:
        f = _dog_chocolate(
            doc_type="nutrition",
            substance="어류기름(어유)",
            threshold_type="성분 함량",
            dose="100.71",
            unit="%",
            basis="건물 기준",
            feeding_level=None,
            effect="",
            signs=[],
            onset="",
        )
        text = verbalize(f)
        assert "권장" not in text
        assert "성분 함량 정보" in text
        assert "100.71% 수준으로 보고되었다" in text

    def test_nutrition_recommendation_keeps_predicate(self) -> None:
        """`기준은 …당다` 처럼 서술격 조사가 깨지지 않아야 한다."""
        f = _dog_chocolate(
            doc_type="nutrition",
            substance="비타민 D",
            threshold_type="",
            dose="125",
            unit="IU",
            basis="1,000kcal 대사에너지 당",
            life_stage="성견",
            feeding_level=None,
            effect="",
            signs=[],
            onset="",
        )
        text = verbalize(f)
        assert "최소 125IU가 권장된다" in text
        assert "기준은 1,000kcal 대사에너지 당이다" in text
        assert "당다" not in text

    def test_toxicity_composition_is_not_dropped(self) -> None:
        """초콜릿 종류별 테오브로민 함량이 문장에서 사라지면 안 된다.

        "다크가 왜 더 위험한가"의 근거가 그 수치다.
        역치가 아니므로 "이상 섭취 시" 로 말해서도 안 된다.
        """
        f = _dog_chocolate(
            substance="세미스위트 다크 초콜릿",
            threshold_type="성분 함량",
            dose="5",
            unit="mg/g",
            effect="테오브로민 함유",
            signs=[],
            onset="",
        )
        text = verbalize(f)
        assert "5mg/g 수준의 함량이 보고되었다" in text
        assert "이상 섭취 시" not in text

    def test_empty_effect_does_not_leak_default(self) -> None:
        """`effect_ko` 는 비었을 때 "임상 징후"를 돌려준다 — 성분 조성에 그 말은 없다."""
        f = _dog_chocolate(
            doc_type="nutrition",
            substance="귀리(연맥)",
            threshold_type="성분 함량",
            dose="10.98",
            unit="%",
            basis="건물 기준",
            feeding_level=None,
            effect="",
            signs=[],
            onset="",
        )
        assert "임상 징후" not in verbalize(f)


class TestFeedingLevelNotDropped:
    """급여 가부(축 B)를 문장에서 빠뜨리지 않는다.

    **여기 없으면 `NEVER` 가 "권장량이다"로 뒤집힌다.** 실제로 그랬다 —
    S-003 의 소금 블록(앵무새 급여 금지)이

        "앵무새 전 생애단계의 소금 블록(salt block) 권장량이다."

    로 적재돼 있었다. 원문은 **주지 말라**고 한다.
    2026-08-01 검색 점검 확장 중 발견. 04 §2.5.1 문장화 충실도 목표는 **0** 이다.
    """

    def _fact(self, **kw):
        from pettriage.schemas import Fact

        base = dict(
            fact_id="F-TEST-001",
            source_id="S-003",
            publisher="AAV",
            doc_type="nutrition",
            species="bird",
            substance="소금 블록(salt block)",
        )
        base.update(kw)
        return Fact(**base)

    def _render(self, f) -> str:
        from pettriage.ingest.templates import TEMPLATES

        return TEMPLATES["nutrition"].render(f)

    def test_금지가_권장으로_뒤집히지_않는다(self):
        from pettriage.triage.levels import FeedingLevel

        out = self._render(self._fact(feeding_level=FeedingLevel.NEVER))
        assert "급여 금지" in out
        assert "권장량이다" not in out, out

    def test_조건부는_조건부라고_말한다(self):
        from pettriage.triage.levels import FeedingLevel

        out = self._render(self._fact(feeding_level=FeedingLevel.CAUTION))
        assert "조건부 급여" in out
        assert "권장량이다" not in out, out

    def test_급여가부가_없고_수량이_있으면_권장량_문장(self):
        out = self._render(
            self._fact(substance="트립토판", feeding_level=None, dose="0.50", unit="g")
        )
        assert "권장량이다" in out

    def test_수량도_급여가부도_없으면_권장이라고_말하지_않는다(self):
        """**"권장량이다" 라고 단언하면 안 되는 행이 45건 있었다.**

        `구리 간병증 식단에서 **제한하는** 식품 권장량이다` 처럼
        substance 자체가 부정문인데 권장으로 나갔다.
        `대사에너지 계산식(1단계 총에너지) 권장량이다` 는 계산식을 급여량이라 한 것이다.
        """
        out = self._render(
            self._fact(substance="구리 간병증 식단에서 제한하는 식품", feeding_level=None)
        )
        assert "권장량이다" not in out, out
        assert "영양 지침이다" in out

    def test_두_축을_섞지_않는다(self):
        """`조건부 급여다. 권장량이다.` 는 축 A·B 를 뒤섞은 말이다 (levels.py)."""
        from pettriage.triage.levels import FeedingLevel

        out = self._render(self._fact(feeding_level=FeedingLevel.CAUTION))
        assert out.count("급여") >= 1
        assert "권장량이다" not in out

    def test_생애단계가_없으면_절을_뺀다(self):
        """`앵무새 전 생애단계에게` 는 사람이 쓰는 말이 아니다."""
        from pettriage.triage.levels import FeedingLevel

        out = self._render(self._fact(feeding_level=FeedingLevel.NEVER))
        assert "전 생애단계에게" not in out


class TestAmountWithoutUnit:
    """단위 없는 수량을 **버리지 않는다.**

    `max_value="얇은 조각 1~2쪽"` 처럼 값에 단위가 들어 있으면 `unit` 이 빈다.
    예전 템플릿은 `bool(f.unit)` 로 막아 **급여량이 통째로 사라졌다** (S-047 과일 6종).
    """

    def _render(self, **kw) -> str:
        from pettriage.ingest.templates import TEMPLATES
        from pettriage.schemas import Fact
        from pettriage.triage.levels import FeedingLevel

        base = dict(
            fact_id="F-TEST-002",
            source_id="S-047",
            publisher="Hill's",
            doc_type="nutrition",
            species="dog",
            substance="사과",
            feeding_level=FeedingLevel.CAUTION,
        )
        base.update(kw)
        return TEMPLATES["nutrition"].render(Fact(**base))

    def test_단위가_없어도_수량이_나온다(self):
        out = self._render(max_value="얇은 조각 1~2쪽", unit=None)
        assert "얇은 조각 1~2쪽" in out, out

    def test_단위가_있으면_붙인다(self):
        out = self._render(max_value="10", unit="%")
        assert "10%" in out, out

    def test_수량이_없으면_절이_없다(self):
        out = self._render(max_value=None)
        assert "권장량은" not in out


class TestUnitJosa:
    """단위를 **읽는 소리**로 조사를 고른다.

    `기준은 10%이다(다).` 가 실제로 나갔다 — `%` 가 기호라 판단 불가가 되어
    병기가 붙은 것이다. 사람은 "십 퍼센트다" 라고 읽는다.
    """

    @pytest.mark.parametrize(
        ("word", "expect"),
        [
            ("10%", "10%이다"),
            ("0.5g", "0.5g이다"),
            ("2.3kg", "2.3kg이다"),
            ("20mg", "20mg이다"),
            ("100mL", "100mL다"),
            ("300kcal", "300kcal다"),
            ("5IU", "5IU다"),
        ],
    )
    def test_이다_다(self, word: str, expect: str) -> None:
        from pettriage.ingest.templates import _ida

        assert _ida(word) == expect

    def test_병기가_남지_않는다(self) -> None:
        """`이다(다)` 가 문장에 보이면 판정에 실패한 것이다."""
        from pettriage.ingest.templates import _ida

        for w in ("10%", "0.5g", "100mL", "300kcal"):
            assert "(" not in _ida(w)

    @pytest.mark.parametrize(
        ("word", "expect"),
        [("BER", "BER이다"), ("RER", "RER이다"), ("ME", "ME다"), ("DHA", "DHA다")],
    )
    def test_알파벳도_읽는_소리로(self, word: str, expect: str) -> None:
        """`RER = 1.25 × BER이다(다)` 가 실제로 나갔다.

        받침으로 끝나는 알파벳은 **L·M·N·R** 넷뿐이다 (엘·엠·엔·알).
        """
        from pettriage.ingest.templates import _ida

        assert _ida(word) == expect

    def test_전체_코퍼스에_병기가_없다(self) -> None:
        """888청크 전수 검사. **하나라도 남으면 사람이 쓴 글로 안 보인다.**"""
        from pettriage import paths
        from pettriage.ingest.facts_io import build_chunks, load_all

        root = paths.find_root()
        if root is None or not (root / "data" / "facts").is_dir():
            pytest.skip("사실 표 없음")
        chunks = build_chunks(load_all(root / "data" / "facts"))
        bad = [
            c.chunk_id for c in chunks if any(x in c.text for x in ("이다(다)", "은(는)", "이(가)"))
        ]
        assert not bad, bad[:5]


class TestTriageNotInvented:
    """등급이 없는 자료에 **등급을 붙이지 않는다.**

    `triage_ko` 가 `None` 일 때 `"확인 필요"` 를 돌려줬고, 그래서
    `emergency` 131행 중 **82행**이 *"…는 확인 필요 상황이다"* 로 나갔다.
    **4등급 어디에도 없는 말이고, 출처가 주지 않은 분류다.**

    `TOXICITY_FOOD` 의 급여 등급 절에는 같은 규율이 이미 적혀 있었다 —
    *"기본값을 채워 넣으면 출처에 없는 분류를 주장하게 된다."* `EMERGENCY` 에만 빠져 있었다.
    """

    def _fact(self, **kw):
        from pettriage.schemas import Fact

        base = dict(
            fact_id="F-TEST-003",
            source_id="S-078",
            publisher="테스트",
            doc_type="emergency",
            species="bird",
            substance="향초·플러그인 방향제(VOC)",
        )
        base.update(kw)
        return Fact(**base)

    def _render(self, f) -> str:
        from pettriage.ingest.templates import TEMPLATES

        return TEMPLATES["emergency"].render(f)

    def test_등급이_없으면_등급을_말하지_않는다(self):
        out = self._render(self._fact(triage_level=None))
        assert "확인 필요" not in out, out
        assert "상황이다" not in out, out
        assert "응급 안전 정보다" in out

    def test_등급이_있으면_그대로_쓴다(self):
        from pettriage.triage.levels import TriageLevel

        out = self._render(self._fact(triage_level=TriageLevel.EMERGENCY))
        assert "응급 상황이다" in out

    def test_triage_ko_는_기본값을_만들지_않는다(self):
        assert self._fact(triage_level=None).triage_ko == ""

    def test_증상은_등급과_무관하게_남는다(self):
        """등급이 없어도 **증상은 원문에 있다** — 그것까지 잃으면 안 된다."""
        out = self._render(self._fact(triage_level=None, signs=["호흡곤란", "개구호흡"]))
        assert "호흡곤란" in out
