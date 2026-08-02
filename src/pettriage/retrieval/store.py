"""벡터 저장소 — 프로토콜과 두 구현.

설계 근거: docs/02_시스템-아키텍처.md §8 · §11 · docs/06 D-14 · D-20

    **메타데이터 필터가 이 도메인의 핵심이다.** 종을 잘못 섞으면 치명적이라
    (D-10) 검색 단계에서 `species` 로 먼저 잘라낸다. 필터는 LLM이 아니라
    코드가 구성한다 (05 §4).

    ⚠️ **유사도 임계값은 거절을 만들지 못한다** (D-46). 실측에서 근거 있음(0.547~0.733)과
    없음(0.494~0.659) 분포가 겹쳤다. `filter_by_threshold` 는 순위를 다듬는 장치이지
    거절 장치가 아니다 — **결과가 0건인 것**만이 거절이다.
    (2026-08-02 정정. 이 문단은 원래 "임계 미만은 거절"이라고 적혀 있었다.)

`InMemoryStore` 는 의존성 없이 도는 구현이다. CI 와 테스트가 여기서 돈다.
`ChromaStore` 는 실제 적재용이며 무거운 임포트를 함수 안에서 한다.
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass, field
from typing import Any, Protocol, runtime_checkable

from ..schemas import Chunk
from .embedder import Embedder

log = logging.getLogger(__name__)


@dataclass
class Hit:
    """검색 결과 1건. `score` 는 코사인 유사도(0~1)."""

    chunk: Chunk
    score: float
    #: 중복 접기로 흡수한 **다른 자료의 `source_id`** (D-22 · 02 §12 근거 보기).
    #:
    #: 같은 물질이 여러 자료에 있다는 사실은 **버리면 안 되는 정보다** —
    #: 근거가 하나뿐인 주장과 넷이 같은 말을 하는 주장은 무게가 다르고,
    #: 인용 화면은 그것을 보여줘야 한다.
    merged_sources: list[str] = field(default_factory=list)

    @property
    def source_id(self) -> str:
        return self.chunk.source_id

    @property
    def all_sources(self) -> list[str]:
        """이 히트가 대표하는 **모든** 출처. 인용 목록은 이것을 쓴다."""
        return [self.source_id, *self.merged_sources]


@runtime_checkable
class VectorStore(Protocol):
    """그래프 노드는 이것만 안다. Chroma·pgvector 교체가 여기서 끝난다."""

    name: str

    def add(self, chunks: list[Chunk]) -> int: ...

    def search(
        self, query: str, *, top_k: int = 5, where: dict[str, Any] | None = None
    ) -> list[Hit]: ...

    def count(self) -> int: ...


def _meta(chunk: Chunk) -> dict[str, Any]:
    """청크 → 저장소 메타데이터.

    `species` 는 반드시 들어간다 — 종 필터가 D-10 의 구현부다.

    **필터용만 담으면 안 된다.** `publisher`·`locator` 를 빠뜨렸더니
    검색 결과로 `Citation` 을 만들 방법이 없었다 — `Citation.publisher` 는 필수인데
    청크 어디에도 없었고, `source_id → publisher` 를 되찾는 코드 경로도 없었다.
    `answered` 는 근거 ≥1 을 요구하므로, **그래프 엔진이 붙는 순간 publisher 를
    지어내거나 전부 거절하거나** 둘 중 하나가 됐을 것이다 (2026-08-02 검토).

    Chroma 메타는 스칼라만 받는다 — `fact_ids` 는 `|` 로 이어 붙인다.
    """
    return {
        "source_id": chunk.source_id,
        "publisher": chunk.publisher,
        "species": chunk.species,
        "doc_type": chunk.doc_type,
        "substance": chunk.substance or "",
        "route": chunk.route,
        "locator": chunk.locator or "",
        "fact_ids": "|".join(chunk.fact_ids),
    }


class EmptyFilter(ValueError):
    """필터 값이 비어 **어떤 문서도 만족할 수 없다.** 0건이 정답이다 (D-10).

    왜 조건을 조용히 빼지 않고 예외를 던지나
    -------------------------------------
    예전에는 빈 목록을 만나면 그 조건을 `continue` 로 건너뛰었고, 조건이 하나도
    안 남으면 `None`(= **필터 없음**)을 냈다. 실측 결과:

        {"species": []}        → ChromaStore  : 전 종 검색 (필터가 사라짐)
                                 InMemoryStore: 0건
        {"species": ["", None]} → 위와 동일

    **종이 미확인일 때 개 보호자에게 고양이 백합 청크가 나가는 경로**였다.
    D-10("종 미확인이면 답하지 않는다")의 정확한 반대다.

    번역기는 *"이 조건은 무시해도 된다"* 를 판단할 자격이 없다. 판단은 저장소가 한다 —
    `search()` 가 이 예외를 잡아 빈 리스트를 낸다. 그러면 두 저장소가 같아지고,
    `retrieve` 노드는 "결과 0건 = 거절"(D-46)을 이미 갖고 있으므로 손댈 것이 없다.
    """


def to_chroma_where(where: dict[str, Any] | None) -> dict[str, Any] | None:
    """우리 필터 표기를 Chroma 문법으로 옮긴다.

    그래프 노드는 `{"species": ["cat", "mammal", "all"]}` 처럼 **목록을 그대로** 넘긴다
    (D-39 고양이 병합 검색). Chroma 는 이걸 못 받고 `{"$in": [...]}` 를 요구하며,
    조건이 둘 이상이면 `$and` 로 묶어야 한다.

        {"species": ["cat", "mammal"], "doc_type": "toxicity_food"}
        → {"$and": [{"species": {"$in": [...]}}, {"doc_type": "toxicity_food"}]}

    **번역을 여기 한 곳에 둔다.** 노드가 저장소 문법을 알면 pgvector 로 옮길 때
    노드를 전부 고쳐야 한다 — `VectorStore` Protocol 을 둔 이유가 그것이다.
    """
    if not where:
        return None
    clauses: list[dict[str, Any]] = []
    for k, v in where.items():
        if isinstance(v, list | tuple | set):
            vals = [x for x in v if x is not None and x != ""]
            if not vals:
                raise EmptyFilter(
                    f"{k} 필터가 비었다 — 조건을 만족하는 문서가 없다 (D-10). "
                    f"검색을 건너뛰고 0건으로 처리할 것."
                )
            clauses.append({k: {"$in": list(vals)}} if len(vals) > 1 else {k: vals[0]})
        else:
            clauses.append({k: v})
    return clauses[0] if len(clauses) == 1 else {"$and": clauses}


def _matches(meta: dict[str, Any], where: dict[str, Any] | None) -> bool:
    if not where:
        return True
    for k, v in where.items():
        got = meta.get(k)
        if isinstance(v, list | tuple | set):
            if got not in v:
                return False
        elif got != v:
            return False
    return True


@dataclass
class InMemoryStore:
    """의존성 없는 저장소 — 테스트·CI 전용.

    벡터DB 를 켜지 않고도 **적재 → 필터 → 검색 → 임계값 거절**을 검증한다.
    """

    embedder: Embedder
    name: str = "memory"
    _chunks: list[Chunk] = field(default_factory=list)
    _vecs: list[list[float]] = field(default_factory=list)

    def add(self, chunks: list[Chunk]) -> int:
        if not chunks:
            return 0
        vecs = self.embedder.encode([c.text for c in chunks])
        self._chunks.extend(chunks)
        self._vecs.extend(vecs)
        return len(chunks)

    def search(
        self, query: str, *, top_k: int = 5, where: dict[str, Any] | None = None
    ) -> list[Hit]:
        if not self._chunks:
            return []
        q = self.embedder.encode([query])[0]
        hits: list[Hit] = []
        for c, v in zip(self._chunks, self._vecs, strict=True):
            if not _matches(_meta(c), where):
                continue
            hits.append(Hit(chunk=c, score=_cosine(q, v)))
        hits.sort(key=lambda h: h.score, reverse=True)
        return hits[:top_k]

    def count(self) -> int:
        return len(self._chunks)


def _cosine(a: list[float], b: list[float]) -> float:
    num = sum(x * y for x, y in zip(a, b, strict=True))
    da = math.sqrt(sum(x * x for x in a)) or 1.0
    db = math.sqrt(sum(y * y for y in b)) or 1.0
    return num / (da * db)


class ChromaStore:
    """Chroma 구현 (02 §11).

    pgvector 로 옮길 때는 이 클래스만 다시 쓴다 — 그래프 노드는 손대지 않는다.
    """

    name = "chroma"

    def __init__(
        self, embedder: Embedder, persist_dir: str = ".chroma", collection: str = "external"
    ):
        self.embedder = embedder
        self._dir = persist_dir
        self._name = collection
        self._col = None

    def _ensure(self):
        if self._col is None:
            import logging as _logging

            # chromadb 0.6.3 은 기동 시점 이벤트 2건에 대해 아래 설정을 무시하고
            # posthog 를 호출한다. 최신 posthog 와 시그니처가 달라 호출 자체가
            # 실패하므로 **밖으로 나가는 것은 없지만** 로그가 시끄럽다.
            # 실패 경고를 지운다 — 진짜 오류가 이 소음에 묻히면 안 된다.
            _logging.getLogger("chromadb.telemetry.product.posthog").setLevel(_logging.CRITICAL)

            import chromadb
            from chromadb.config import Settings

            # 텔레메트리를 끈다 — 개인 기록을 다루는 시스템이 외부로 사용 신호를
            # 내보내면 안 된다 (D-36 조치 5·6 과 같은 취지).
            client = chromadb.PersistentClient(
                path=self._dir, settings=Settings(anonymized_telemetry=False)
            )
            self._col = client.get_or_create_collection(
                self._name, metadata={"hnsw:space": "cosine"}
            )
        return self._col

    def add(self, chunks: list[Chunk]) -> int:
        if not chunks:
            return 0
        col = self._ensure()
        col.upsert(  # D-20 증분 인덱싱 — chunk_id 기준 upsert
            ids=[c.chunk_id for c in chunks],
            documents=[c.text for c in chunks],
            embeddings=self.embedder.encode([c.text for c in chunks]),
            metadatas=[_meta(c) for c in chunks],
        )
        return len(chunks)

    def search(
        self, query: str, *, top_k: int = 5, where: dict[str, Any] | None = None
    ) -> list[Hit]:
        try:
            # 목록 필터는 Chroma 문법으로 옮겨야 한다 — 그대로 넘기면 ValueError 다
            chroma_where = to_chroma_where(where)
        except EmptyFilter as e:
            # 조건을 만족하는 문서가 없다. **필터를 빼고 검색하지 않는다** (D-10).
            log.info("검색 생략 — %s", e)
            return []
        col = self._ensure()
        res = col.query(
            query_embeddings=self.embedder.encode([query]),
            n_results=top_k,
            where=chroma_where,
        )
        hits: list[Hit] = []
        for i, doc in enumerate(res["documents"][0]):
            m = res["metadatas"][0][i]
            hits.append(
                Hit(
                    chunk=Chunk(
                        chunk_id=res["ids"][0][i],
                        text=doc,
                        source_id=m.get("source_id", ""),
                        publisher=m.get("publisher", ""),
                        species=m.get("species", "all"),
                        doc_type=m.get("doc_type", "symptom"),
                        substance=m.get("substance") or None,
                        route=m.get("route", "사실추출"),
                        locator=m.get("locator") or None,
                        fact_ids=[x for x in (m.get("fact_ids") or "").split("|") if x],
                    ),
                    # Chroma 는 거리(distance)를 준다. 코사인 거리 → 유사도
                    score=1.0 - float(res["distances"][0][i]),
                )
            )
        return hits

    def count(self) -> int:
        return self._ensure().count()


def filter_by_threshold(hits: list[Hit], threshold: float) -> list[Hit]:
    """임계값 미만을 잘라낸다.

    **비어서 돌아오면 그건 실패가 아니라 거절 신호다** (02 §8.3 · §9).
    부르는 쪽이 `refused / 근거없음` 으로 보내야 한다.
    """
    return [h for h in hits if h.score >= threshold]


def dedupe_by_substance(hits: list[Hit]) -> list[Hit]:
    """같은 (물질 × 종) 히트를 **하나로 접는다.** 점수가 가장 높은 것을 남긴다.

    설계 근거: docs/06 D-14 · D-22 · D-39 · docs/04 §2.5.6

    **왜 필요한가** — 같은 물질이 여러 자료에 흩어져 있다.

        `양파` 청크가 8건이다 (S-010 · S-019 · S-029 · S-034 ×2 · S-063 · S-098 ×2).
        고양이 질의는 D-39 병합(`cat`+`mammal`+`all`)으로 그중 4건을 함께 본다.

        실측: `고양이가 베란다에 둔 파란 알갱이를 주워 먹었어요`
              → 상위 5 = 사람 음식 · **양파 · 양파 · 양파** · 토란
              **`top_k=5` 가 실질 3종이 됐다.**

    검색이 틀린 것은 아니다. **문맥 예산을 같은 말로 채우는 것**이 문제다 —
    압축·생성으로 넘어가는 근거가 그만큼 좁아진다.

    **접되 버리지 않는다.** 흡수한 자료의 `source_id` 는 `merged_sources` 에 남긴다.
    근거가 하나뿐인 주장과 넷이 같은 말을 하는 주장은 무게가 다르고,
    02 §12의 '근거 보기' 화면은 그 차이를 보여줘야 한다.

    **종을 열쇠에 넣는 이유** — 개 양파(15-30 g/kg)와 고양이 양파(5 g/kg)는
    **다른 수치**다 (D-39). 종까지 같을 때만 접는다.

    **물질명이 없는 청크는 접지 않는다.** 응급 지침처럼 `substance` 가 빈 청크를
    한 덩어리로 묶으면 서로 다른 내용이 사라진다.

    입력 순서(=점수 내림차순)를 유지한다.
    """
    out: list[Hit] = []
    seen: dict[tuple[str, str], Hit] = {}
    for h in hits:
        substance = (h.chunk.substance or "").strip()
        if not substance:
            out.append(h)
            continue
        key = (substance, h.chunk.species)
        first = seen.get(key)
        if first is None:
            seen[key] = h
            out.append(h)
        elif h.source_id not in first.all_sources:
            first.merged_sources.append(h.source_id)
    return out
