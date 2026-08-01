"""검색 필터 번역 — 목록 필터가 Chroma 에서 그대로 깨지지 않는지.

그래프 노드는 `{"species": ["cat", "mammal", "all"]}` 를 그대로 넘긴다 (D-39 병합 검색).
`InMemoryStore` 는 이걸 받지만 **Chroma 는 `$in` 을 요구하고 목록을 주면 ValueError 를 낸다.**
통합 시점에 이 경로에서 처음 터졌던 것을 테스트로 못 박는다.
"""

from __future__ import annotations

from pettriage.retrieval.store import _matches, to_chroma_where


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

    def test_empty_values_are_dropped(self) -> None:
        """빈 종 값이 섞여 들어와도 필터가 깨지지 않는다 — 종 미확인 경로가 있다 (D-10)."""
        assert to_chroma_where({"species": ["", None]}) is None
        assert to_chroma_where({"species": ["dog", ""]}) == {"species": "dog"}


class TestInMemoryMatchesSameSemantics:
    """두 구현이 같은 뜻으로 동작해야 한다. 안 그러면 저장소를 갈 때 결과가 바뀐다."""

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
