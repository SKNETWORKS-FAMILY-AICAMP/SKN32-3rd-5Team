"""임베딩 — 프로토콜과 두 구현.

설계 근거: docs/02_시스템-아키텍처.md §8 · docs/06 D-19

    ⚠️ 이 문단은 2026-08-02에 정정됐다. 이전 판은 *"자료의 90%가 영문이라
    cross-lingual 성능이 검색 성패를 가른다"* 였는데, **그 전제가 사라졌다** —
    경로②(D-37·D-38·D-45)로 **벡터DB 청크가 전부 한국어**가 됐다.
    `bge-m3` 는 그대로 쓰지만 근거는 다국어가 아니라 **한국어 검색 성능**이다
    (D-19 후속 · 03 §2.3).

`HashEmbedder` 는 모델 없이 도는 결정론적 구현이다. 테스트와 CI 가
GPU·네트워크 없이 파이프라인 전체를 검증할 수 있어야 하기 때문이다.
**검색 품질 실험에는 쓰지 않는다** — 의미를 담지 않는다.
"""

from __future__ import annotations

import hashlib
import logging
import math
import time
from functools import lru_cache
from typing import Protocol, runtime_checkable

log = logging.getLogger(__name__)


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

    @property
    def loaded(self) -> bool:
        """모델이 메모리에 올라와 있는가. `/api/health` 와 워밍업이 본다."""
        return self._model is not None

    def _ensure(self) -> None:
        if self._model is None:
            from sentence_transformers import SentenceTransformer

            # 로딩 시간을 **로그로 남긴다.** 이 프로젝트에서 아직 측정된 적이 없고,
            # 추정으로 "수십 초" 라고 적어 두면 그게 곧 지어낸 숫자가 된다 (D-53).
            t0 = time.perf_counter()
            self._model = SentenceTransformer(self.name, device=self._device)
            self.dim = self._model.get_sentence_embedding_dimension()
            log.info("임베딩 모델 로드 완료 — %s · %.1fs", self.name, time.perf_counter() - t0)

    def encode(self, texts: list[str]) -> list[list[float]]:
        self._ensure()
        assert self._model is not None
        vecs = self._model.encode(texts, normalize_embeddings=True)
        return [list(map(float, v)) for v in vecs]


@lru_cache(maxsize=4)
def get_embedder(name: str = "hash-test") -> Embedder:
    """설정값 → 구현. `configs/*.yaml` 의 `retrieval.embedding_model` 이 들어온다.

    ## 🔴 `lru_cache` 가 여기 붙어 있는 이유 (D-53)

    붙이기 전에는 **부를 때마다 새 `BGEEmbedder` 인스턴스**가 나왔다.
    `_ensure()` 가 인스턴스 안에서는 한 번만 로드하므로 안전해 **보이지만**,
    인스턴스가 매번 새로 생기면 그 캐시가 아무 소용이 없다.

    ```python
    # ✗ 노드 안에서 이렇게 쓰면 — 그리고 이게 자연스러운 코드다
    def retrieve(state):
        emb = get_embedder(cfg.retrieval.embedding_model)   # 매번 새 인스턴스
        hits = store.search(...)                            # → 매 질의마다 모델 재로드
    ```

    **문서에 "모델을 상주시켜라" 라고 적는 것으로는 이걸 막지 못한다.**
    D-40 이 *"지키기로 한 것이 아니라 못 어기는 것"* 이라고 한 그 자리다.
    같은 이름이면 **같은 인스턴스**를 돌려주므로, 노드가 어디서 몇 번을 부르든
    로드는 프로세스당 한 번이다.

    `maxsize=4` — 실제로 쓰는 이름은 `bge-m3` 와 `hash-test` 둘뿐이고,
    실험에서 모델을 바꿔 끼워도 넷이면 넉넉하다. 무한 캐시로 두면
    오타 난 모델명이 쌓여도 아무도 모른다.

    ⚠️ **테스트에서 임베더를 바꿔 끼울 때는 `get_embedder.cache_clear()` 를 부른다.**
    """
    if name in ("hash-test", "test"):
        return HashEmbedder()
    return BGEEmbedder(name)


def warm_up(name: str) -> float | None:
    """모델을 미리 올린다. **첫 사용자가 로딩을 맞지 않게 한다** (D-53).

    Returns:
        로딩에 걸린 초. 이미 올라와 있었으면 `0.0`.
        테스트 임베더처럼 로드가 필요 없으면 **`None`** —
        0.0 과 구분한다. *"안 쟀다"* 와 *"쟀는데 0"* 은 다르다.
    """
    emb = get_embedder(name)
    if not isinstance(emb, BGEEmbedder):
        return None
    if emb.loaded:
        return 0.0
    t0 = time.perf_counter()
    emb.encode(["워밍업"])  # 실제 인코딩까지 한 번 — 로드만으로는 첫 호출이 여전히 느리다
    return time.perf_counter() - t0
