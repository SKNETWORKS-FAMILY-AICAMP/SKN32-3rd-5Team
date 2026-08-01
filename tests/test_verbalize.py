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

    def test_absolute_unit_gets_per_weight_prefix(self):
        """체중당 단위가 아니면 "체중 1kg당"을 붙여야 의미가 산다."""
        text = verbalize(_dog_chocolate(dose="2.3", unit="g"))
        assert "체중 1kg당 2.3g 이상 섭취 시" in text

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
            # 판단할 수 없으면 병기한다 — 틀린 조사를 붙이지 않는다
            ("Xylitol", "Xylitol은(는)"),
        ],
    )
    def test_eun_neun(self, word: str, expected: str) -> None:
        assert templates._eun(word) == expected

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
