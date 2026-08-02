"""검색 결과 중복 접기 (D-14 · D-22 · D-39 · 04 §2.5.6).

`양파` 청크가 코퍼스에 **8건**이다 (S-010·S-019·S-029·S-034 ×2·S-063·S-098 ×2).
고양이 질의는 D-39 병합(`cat`+`mammal`+`all`)으로 그중 4건을 함께 본다.

    실측: '고양이가 베란다에 둔 파란 알갱이를 주워 먹었어요'
          → 상위 5 = 사람 음식 · **양파 · 양파 · 양파** · 토란

**`top_k=5` 가 실질 3종이 된다.** 검색이 틀린 것은 아니고,
문맥 예산을 같은 말로 채우는 것이 문제다.
"""

from __future__ import annotations

import pytest

from pettriage.retrieval import Hit, dedupe_by_substance
from pettriage.schemas import Chunk


def chunk(substance: str, species: str, source_id: str) -> Chunk:
    return Chunk(
        chunk_id=f"c-{source_id}-{substance}",
        text=f"{substance} 청크",
        source_id=source_id,
        species=species,
        doc_type="toxicity_food",
        substance=substance,
        fact_ids=[f"F-{source_id}-001"],
    )


def hit(substance: str, species: str, source_id: str, score: float) -> Hit:
    return Hit(chunk(substance, species, source_id), score)


class TestDedupe:
    def test_같은_물질_같은_종을_접는다(self) -> None:
        hits = [
            hit("사람 음식", "cat", "S-1", 0.62),
            hit("양파", "cat", "S-010", 0.61),
            hit("양파", "cat", "S-034", 0.60),
            hit("양파", "cat", "S-098", 0.59),
            hit("토란", "cat", "S-9", 0.58),
        ]
        out = dedupe_by_substance(hits)
        assert [h.chunk.substance for h in out] == ["사람 음식", "양파", "토란"]

    def test_점수가_가장_높은_것을_남긴다(self) -> None:
        out = dedupe_by_substance(
            [hit("양파", "cat", "S-010", 0.61), hit("양파", "cat", "S-034", 0.60)]
        )
        assert out[0].source_id == "S-010"
        assert out[0].score == pytest.approx(0.61)

    def test_흡수한_출처를_버리지_않는다(self) -> None:
        """**근거가 하나뿐인 주장과 넷이 같은 말을 하는 주장은 무게가 다르다** (02 §12)."""
        out = dedupe_by_substance(
            [
                hit("양파", "cat", "S-010", 0.61),
                hit("양파", "cat", "S-034", 0.60),
                hit("양파", "cat", "S-098", 0.59),
            ]
        )
        assert out[0].all_sources == ["S-010", "S-034", "S-098"]

    def test_종이_다르면_접지_않는다(self) -> None:
        """개 양파 15-30 g/kg · 고양이 양파 5 g/kg — **다른 수치다** (D-39)."""
        out = dedupe_by_substance(
            [hit("양파", "dog", "S-034", 0.70), hit("양파", "cat", "S-034", 0.60)]
        )
        assert len(out) == 2

    def test_물질명이_없으면_접지_않는다(self) -> None:
        """응급 지침처럼 `substance` 가 빈 청크를 묶으면 서로 다른 내용이 사라진다."""
        out = dedupe_by_substance(
            [hit("", "cat", "S-1", 0.6), hit("", "cat", "S-2", 0.5), hit("", "cat", "S-3", 0.4)]
        )
        assert len(out) == 3

    def test_같은_출처가_두_번_들어가지_않는다(self) -> None:
        """S-034 는 양파를 두 행으로 적는다 — 같은 자료가 인용에 중복되면 안 된다."""
        out = dedupe_by_substance(
            [hit("양파", "cat", "S-034", 0.61), hit("양파", "cat", "S-034", 0.60)]
        )
        assert out[0].all_sources == ["S-034"]

    def test_순서를_유지한다(self) -> None:
        hits = [
            hit("A", "cat", "S-1", 0.9),
            hit("B", "cat", "S-2", 0.8),
            hit("C", "cat", "S-3", 0.7),
        ]
        assert [h.chunk.substance for h in dedupe_by_substance(hits)] == ["A", "B", "C"]

    def test_빈_입력(self) -> None:
        assert dedupe_by_substance([]) == []

    def test_원본_히트를_망가뜨리지_않는다(self) -> None:
        """접기는 `merged_sources` 만 채운다 — 점수·청크는 그대로다."""
        a = hit("양파", "cat", "S-010", 0.61)
        dedupe_by_substance([a, hit("양파", "cat", "S-034", 0.60)])
        assert a.score == pytest.approx(0.61)
        assert a.chunk.source_id == "S-010"
