"""사실 표 → 청크 → 검색 — 파이프라인 배선이 살아 있는가.

벡터DB·모델 없이 돈다. 사실 표가 채워지는 즉시 이 경로가 동작해야 한다.
"""

from __future__ import annotations

import pytest

from pettriage.ingest.facts_io import (
    DuplicateFactId,
    build_chunks,
    load_all,
    load_facts,
    row_to_fact,
    summarize,
)
from pettriage.retrieval import HashEmbedder, InMemoryStore, filter_by_threshold
from pettriage.triage.levels import FeedingLevel

HEADER = (
    "fact_id,source_id,publisher,doc_type,species,substance,scientific_name,toxic_part,"
    "threshold_type,dose,unit,max_value,basis,life_stage,feeding_level,triage_level,"
    "effect,signs,onset,escalation_conditions,quote,locator,accessed_at,citation,"
    "extracted_by,note"
)
ROW_DOG = (
    "F-034-001,S-034,Veterinary Sciences,toxicity_food,dog,초콜릿,,,임상징후 발현,20,mg/kg,,,,"
    "NEVER,,경증 임상징후,구토|다음,2-4시간,,,Table 1,2026-08-01,,ohb,"
)
ROW_BIRD = (
    "F-005-001,S-005,Lafeber,toxicity_food,bird,아보카도,Persea americana,과육,,,,,,,"
    "NEVER,,심근 괴사,호흡곤란,12시간,,,Never 등급,2026-08-01,,lgj,"
)


def _write(tmp_path, name, *rows):
    p = tmp_path / name
    p.write_text("\n".join([HEADER, *rows]) + "\n", encoding="utf-8")
    return p


def test_row_becomes_fact_with_enums():
    fact = row_to_fact(dict(zip(HEADER.split(","), ROW_DOG.split(","), strict=True)))
    assert fact.fact_id == "F-034-001"
    assert fact.feeding_level is FeedingLevel.NEVER
    assert fact.signs == ["구토", "다음"]


def test_blank_cells_stay_none():
    """빈 칸을 기본값으로 채우면 템플릿이 없는 사실을 문장으로 만든다."""
    fact = row_to_fact(dict(zip(HEADER.split(","), ROW_BIRD.split(","), strict=True)))
    assert fact.dose is None and fact.threshold_type is None
    assert fact.triage_level is None


def test_bird_chunk_has_no_quantitative_sentence(tmp_path):
    """조류는 임계치가 0건 → 정량 문장이 자동으로 사라진다 (D-09 · D-38)."""
    facts = load_facts(_write(tmp_path, "facts_lgj.csv", ROW_BIRD))
    (chunk,) = build_chunks(facts)
    assert "이상 섭취 시" not in chunk.text
    assert "아보카도" in chunk.text


def test_route2_chunk_has_no_quote(tmp_path):
    """경로 ② 청크는 원문을 담지 않는다 (D-37)."""
    facts = load_facts(_write(tmp_path, "facts_ohb.csv", ROW_DOG))
    (chunk,) = build_chunks(facts)
    assert chunk.quote is None
    assert chunk.route == "사실추출"


def test_duplicate_fact_id_raises(tmp_path):
    """조용히 덮으면 어느 쪽이 적재됐는지 추적할 수 없다."""
    _write(tmp_path, "facts_ohb.csv", ROW_DOG)
    _write(tmp_path, "facts_lgj.csv", ROW_DOG)
    with pytest.raises(DuplicateFactId):
        load_all(tmp_path)


def test_merge_and_summarize(tmp_path):
    _write(tmp_path, "facts_ohb.csv", ROW_DOG)
    _write(tmp_path, "facts_lgj.csv", ROW_BIRD)
    facts = load_all(tmp_path)
    assert len(facts) == 2
    s = summarize(facts)
    assert s["species"] == {"dog": 1, "bird": 1}
    assert s["threshold_type"]["(없음)"] == 1


def test_end_to_end_species_filter(tmp_path):
    """적재 → 종 필터 검색. 종을 섞으면 치명적이다 (D-10)."""
    _write(tmp_path, "facts_ohb.csv", ROW_DOG)
    _write(tmp_path, "facts_lgj.csv", ROW_BIRD)
    store = InMemoryStore(embedder=HashEmbedder())
    store.add(build_chunks(load_all(tmp_path)))
    assert store.count() == 2

    hits = store.search("초콜릿 섭취", top_k=5, where={"species": "bird"})
    assert all(h.chunk.species == "bird" for h in hits)


def test_threshold_cuts_low_similarity():
    """임계 미만은 거절 신호다 (02 §8.3)."""
    store = InMemoryStore(embedder=HashEmbedder())
    from pettriage.schemas import Chunk

    store.add(
        [
            Chunk(
                chunk_id="c1",
                text="개와 초콜릿",
                source_id="S-034",
                species="dog",
                doc_type="toxicity_food",
            )
        ]
    )
    hits = store.search("완전히 다른 이야기", top_k=5)
    assert filter_by_threshold(hits, 0.9) == []
