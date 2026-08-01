"""검색 계층 (WS2).

```
embedder.py   Embedder 프로토콜 · HashEmbedder(테스트) · BGEEmbedder(bge-m3)
store.py      VectorStore 프로토콜 · InMemoryStore(테스트) · ChromaStore
```

그래프 노드는 **프로토콜만** 안다. Chroma → pgvector 교체가 `store.py` 안에서 끝난다.

검색 노드가 부르는 순서는 **검색 → 임계값 → 접기** 다.

```python
hits = store.search(q, top_k=k, where=where)
hits = filter_by_threshold(hits, threshold)
hits = dedupe_by_substance(hits)
```

마지막 단계를 빠뜨리면 **같은 물질이 상위를 채운다** — 실측에서 `top_k=5` 가
실질 3종이 됐다 (04 §2.5.6).
"""

from .embedder import BGEEmbedder, Embedder, HashEmbedder, get_embedder
from .store import (
    ChromaStore,
    Hit,
    InMemoryStore,
    VectorStore,
    dedupe_by_substance,
    filter_by_threshold,
    to_chroma_where,
)

__all__ = [
    "BGEEmbedder",
    "ChromaStore",
    "Embedder",
    "HashEmbedder",
    "Hit",
    "InMemoryStore",
    "VectorStore",
    "dedupe_by_substance",
    "filter_by_threshold",
    "get_embedder",
    "to_chroma_where",
]
