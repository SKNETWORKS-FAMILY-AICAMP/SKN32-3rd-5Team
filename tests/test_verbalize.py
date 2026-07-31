"""문장화 — 결측 필드가 절을 통째로 생략하는지, LLM 없이 결정론적인지.

D-38의 핵심 두 가지를 테스트가 지킨다.
"""

from __future__ import annotations

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
