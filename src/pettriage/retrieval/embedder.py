"""임베딩 — 프로토콜과 두 구현.

설계 근거: docs/02_시스템-아키텍처.md §8 · docs/06 D-19

    다국어를 결정하는 것은 벡터DB가 아니라 **임베딩 모델**이다.
    자료의 90%가 영문인데 질의는 한국어라, cross-lingual 성능이
    검색 성패를 가른다 (D-19 · 04 E2).

`HashEmbedder` 는 모델 없이 도는 결정론적 구현이다. 테스트와 CI 가
GPU·네트워크 없이 파이프라인 전체를 검증할 수 있어야 하기 때문이다.
**검색 품질 실험에는 쓰지 않는다** — 의미를 담지 않는다.
"""

from __future__ import annotations

import hashlib
import math
from typing import Protocol, runtime_checkable


@runtime_checkable
class Embedder(Protocol):
    """텍스트 → 벡터. 벡터DB 계층은 이것만 안다."""

    name: str
    dim: int

    def encode(self, texts: list[str]) -> list[list[float]]: ...


class HashEmbedder:
    """해시 기반 결정론적 임베딩 — **테스트 전용**.

    같은 문자열은 항상 같은 벡터가 되고 모델을 내려받지 않는다.
    파이프라인 배선(적재 → 검색 → 응답)이 살아 있는지만 확인한다.
    """

    name = "hash-test"

    def __init__(self, dim: int = 64) -> None:
        self.dim = dim

    def encode(self, texts: list[str]) -> list[list[float]]:
        out: list[list[float]] = []
        for t in texts:
            vec = [0.0] * self.dim
            # 문자 3-gram 을 해시해 버킷에 더한다 — 부분 일치가 반영되게
            s = t.strip()
            for i in range(max(len(s) - 2, 1)):
                g = s[i : i + 3]
                h = int(hashlib.blake2b(g.encode("utf-8"), digest_size=8).hexdigest(), 16)
                vec[h % self.dim] += 1.0
            norm = math.sqrt(sum(v * v for v in vec)) or 1.0
            out.append([v / norm for v in vec])
        return out


class BGEEmbedder:
    """`BAAI/bge-m3` — 실제 검색에 쓰는 다국어 임베딩 (D-19).

    무거운 임포트를 함수 안에서 한다. GPU 없는 팀원과 CI 가 깨지면 안 된다.
    """

    def __init__(self, model_id: str = "BAAI/bge-m3", device: str | None = None) -> None:
        self.name = model_id
        self._device = device
        self._model = None
        self.dim = 1024  # bge-m3 dense 차원

    def _ensure(self) -> None:
        if self._model is None:
            from sentence_transformers import SentenceTransformer

            self._model = SentenceTransformer(self.name, device=self._device)
            self.dim = self._model.get_sentence_embedding_dimension()

    def encode(self, texts: list[str]) -> list[list[float]]:
        self._ensure()
        assert self._model is not None
        vecs = self._model.encode(texts, normalize_embeddings=True)
        return [list(map(float, v)) for v in vecs]


def get_embedder(name: str = "hash-test") -> Embedder:
    """설정값 → 구현. `configs/*.yaml` 의 `retrieval.embedding_model` 이 들어온다."""
    if name in ("hash-test", "test"):
        return HashEmbedder()
    return BGEEmbedder(name)
