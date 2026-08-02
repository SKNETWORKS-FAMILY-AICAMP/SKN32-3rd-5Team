"""검색 필터 번역 — 목록 필터가 Chroma 에서 그대로 깨지지 않는지.

그래프 노드는 `{"species": ["cat", "mammal", "all"]}` 를 그대로 넘긴다 (D-39 병합 검색).
`InMemoryStore` 는 이걸 받지만 **Chroma 는 `$in` 을 요구하고 목록을 주면 ValueError 를 낸다.**
통합 시점에 이 경로에서 처음 터졌던 것을 테스트로 못 박는다.
"""

from __future__ import annotations

import pytest

from pettriage.retrieval.embedder import get_embedder
from pettriage.retrieval.store import (
    EmptyFilter,
    InMemoryStore,
    _matches,
    to_chroma_where,
)
from pettriage.schemas import Chunk


class TestToChromaWhere:
    def test_none_and_empty(self) -> None:
        assert to_chroma_where(None) is None
        assert to_chroma_where({}) is None

    def test_scalar_stays_scalar(self) -> None:
        assert to_chroma_where({"species": "dog"}) == {"species": "dog"}

    def test_list_becomes_in(self) -> None:
        """D-39 — 고양이는 `mammal`·`all` 을 함께 봐야 한다."""
        got = to_chroma_where({"species": ["cat", "mammal", "all"]})
        assert got == {"species": {"$in": ["cat", "mammal", "all"]}}

    def test_single_item_list_collapses(self) -> None:
        """원소가 하나면 `$in` 을 쓰지 않는다 — 같은 뜻이고 더 단순하다."""
        assert to_chroma_where({"species": ["dog"]}) == {"species": "dog"}

    def test_multiple_keys_wrapped_in_and(self) -> None:
        """Chroma 는 조건이 둘 이상이면 `$and` 를 요구한다."""
        got = to_chroma_where({"species": ["cat", "all"], "doc_type": "toxicity_plant"})
        assert got == {
            "$and": [
                {"species": {"$in": ["cat", "all"]}},
                {"doc_type": "toxicity_plant"},
            ]
        }

    def test_partially_empty_list_keeps_real_values(self) -> None:
        """빈 값이 섞여 있어도 **실제 값이 하나라도 있으면** 그것으로 거른다."""
        assert to_chroma_where({"species": ["dog", ""]}) == {"species": "dog"}

    def test_all_empty_raises(self) -> None:
        """값이 전부 비면 `EmptyFilter` 다 — **필터 없음이 아니다** (D-10).

        ⚠️ 이 테스트는 2026-08-02 에 **정반대로 뒤집혔다.**

            # 예전
            assert to_chroma_where({"species": ["", None]}) is None
            #  ↑ 주석: "빈 종 값이 섞여 들어와도 필터가 깨지지 않는다"

        필터가 깨진 게 아니라 **사라진** 것이었다. `None` 은 Chroma 에서
        *"조건 없이 전부 검색"* 을 뜻하므로, 종이 미확인일 때 개 보호자에게
        고양이 백합 청크가 나가는 경로였다 — D-10 의 정확한 반대다.
        `InMemoryStore` 는 같은 입력에 0건을 냈으므로 두 구현이 갈려 있기도 했다.

        **테스트가 틀린 동작을 정답으로 고정하고 있었다.** 그래서 그때까지
        아무도 이것을 버그로 보지 못했다.
        """
        with pytest.raises(EmptyFilter):
            to_chroma_where({"species": []})
        with pytest.raises(EmptyFilter):
            to_chroma_where({"species": ["", None]})


class TestInMemoryMatchesSameSemantics:
    """두 구현이 같은 뜻으로 동작해야 한다. 안 그러면 저장소를 갈 때 결과가 바뀐다."""

    def test_empty_list_means_zero_hits_in_both(self) -> None:
        """빈 목록 — **두 구현이 갈리던 유일한 지점.** 이제 둘 다 0건이다.

        `ChromaStore.search` 는 `EmptyFilter` 를 잡아 `[]` 를 내고,
        `InMemoryStore` 는 `_matches` 가 `False` 라 애초에 0건이다.
        """
        assert not _matches({"species": "cat"}, {"species": []})
        assert not _matches({"species": "cat"}, {"species": ["", None]})

        store = InMemoryStore(get_embedder("hash-test"))
        store.add(
            [
                Chunk(
                    chunk_id="c1",
                    text="고양이에게 백합은 치명적이다",
                    source_id="S-030",
                    publisher="ASPCA",
                    species="cat",
                    doc_type="toxicity_plant",
                )
            ]
        )
        assert store.search("백합", where={"species": []}) == []
        assert len(store.search("백합", where={"species": ["cat"]})) == 1

    def test_list_membership(self) -> None:
        assert _matches({"species": "mammal"}, {"species": ["cat", "mammal", "all"]})
        assert not _matches({"species": "bird"}, {"species": ["cat", "mammal", "all"]})

    def test_scalar(self) -> None:
        assert _matches({"species": "dog"}, {"species": "dog"})
        assert not _matches({"species": "cat"}, {"species": "dog"})

    def test_multiple_keys_are_and(self) -> None:
        meta = {"species": "dog", "doc_type": "toxicity_food"}
        assert _matches(meta, {"species": "dog", "doc_type": "toxicity_food"})
        assert not _matches(meta, {"species": "dog", "doc_type": "nutrition"})
