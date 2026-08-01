"""검색 계층 (WS2).

```
embedder.py   Embedder 프로토콜 · HashEmbedder(테스트) · BGEEmbedder(bge-m3)
store.py      VectorStore 프로토콜 · InMemoryStore(테스트) · ChromaStore
```

그래프 노드는 **프로토콜만** 안다. Chroma → pgvector 교체가 `store.py` 안에서 끝난다.
"""

from .embedder import BGEEmbedder, Embedder, HashEmbedder, get_embedder
from .store import ChromaStore, Hit, InMemoryStore, VectorStore, filter_by_threshold

__all__ = [
    "BGEEmbedder",
    "ChromaStore",
    "Embedder",
    "HashEmbedder",
    "Hit",
    "InMemoryStore",
    "VectorStore",
    "filter_by_threshold",
    "get_embedder",
]
