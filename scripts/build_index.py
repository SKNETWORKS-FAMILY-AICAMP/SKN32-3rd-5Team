#!/usr/bin/env python
"""사실 표 → 청크 → 벡터DB 적재.

    python scripts/build_index.py                  # 청크만 만들어 보고 (기본)
    python scripts/build_index.py --store chroma   # 실제 적재 + 검색 점검

설계 근거: docs/02 §3·§11 · docs/06 D-14 · D-20 · D-38 · D-44

**문장화는 코드가 한다.** 여기서 LLM 을 부르지 않는다.
표가 맞으면 문장도 맞는다 — 그래서 검증은 표 단계에서 끝난다 (01e).

적재만 하고 끝내지 않는다. **적재 직후 한국어 질의로 검색을 점검한다** —
"들어갔다"와 "찾아진다"는 다른 문제이고, 후자가 안 되면 그래프도 평가도 돌지 않는다.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from pettriage import paths  # noqa: E402
from pettriage.config import get_config  # noqa: E402
from pettriage.ingest.facts_io import build_chunks, load_all, summarize  # noqa: E402

#: 적재 직후 돌리는 점검 질의 — (보호자가 쓸 법한 말, 물어와야 할 것, 기대 종).
#:
#: 임베딩을 갈거나 문장 템플릿을 고치면 **여기가 먼저 깨진다.**
#: 고양이 질의는 D-39의 병합 검색을 건드린다 —
#: `cat` 자체 자료가 2단계뿐이라 `mammal`·`all` 을 함께 봐야 한다.
PROBES: tuple[tuple[str, str, str], ...] = (
    ("강아지가 초콜릿을 먹었어요", "초콜릿", "dog"),
    ("우리 개가 포도를 먹었는데 괜찮을까요", "포도", "dog"),
    ("강아지가 자일리톨 껌을 삼켰어요", "자일리톨", "dog"),
    ("고양이가 백합을 씹었어요", "백합", "cat"),
    ("고양이가 양파 들어간 음식을 먹었어요", "양파", "cat"),
    ("앵무새가 아보카도를 먹었어요", "아보카도", "bird"),
    ("앵무새 앞에서 프라이팬을 태웠어요", "PTFE", "bird"),
)


def probe(store, threshold: float, top_k: int) -> int:
    """점검 질의를 돌려 **실패 건수**를 돌려준다.

    통과 기준 두 가지 —
      1. 기대한 것이 상위 `top_k` 안에 있다
      2. 1위 점수가 `score_threshold` 를 넘는다.
         못 넘으면 파이프라인이 그 질의를 **거절로 보낸다** (02 §8.3)
    """
    print(f"\n검색 점검 — top_k={top_k} · score_threshold={threshold}")
    fails = 0
    for q, expect, species in PROBES:
        where = {"species": [species, "mammal", "all"]} if species else None
        hits = store.search(q, top_k=top_k, where=where)
        if not hits:
            print(f"  ✗ {q!r} → 검색 결과 없음")
            fails += 1
            continue
        top = hits[0]
        found = any(expect in h.chunk.substance or expect in h.chunk.text for h in hits)
        ok_score = top.score >= threshold
        ok = found and ok_score
        fails += 0 if ok else 1
        print(f"  {'✓' if ok else '✗'} {q!r}")
        print(f"      1위 {top.score:.3f} · {top.chunk.substance} ({top.source_id})")
        if not found:
            print(f"      기대한 {expect!r} 가 상위 {top_k} 안에 없다")
        if not ok_score:
            print(f"      1위 점수가 임계값 {threshold} 미만 — 이 질의는 거절로 간다")
    print(f"\n  → 점검 {len(PROBES)}건 중 실패 {fails}건")
    if fails:
        print("     임베딩·문장 템플릿·score_threshold 중 하나를 봐야 한다 (configs/default.yaml)")
    return fails


def main() -> int:
    cfg = get_config()
    ap = argparse.ArgumentParser(description="사실 표를 벡터DB에 적재한다")
    ap.add_argument("--facts-dir", type=Path, default=None)
    ap.add_argument(
        "--store",
        choices=["dry-run", "memory", "chroma"],
        default="dry-run",
        help="chroma 는 configs 의 retrieval.persist_dir 에 적재한다 (D-44)",
    )
    ap.add_argument("--out", type=Path, help="청크를 JSONL 로 저장 (검수용)")
    ap.add_argument("--no-probe", action="store_true", help="적재 후 검색 점검을 건너뛴다")
    args = ap.parse_args()

    root = paths.find_root() or Path.cwd()
    facts_dir = args.facts_dir or root / "data" / "facts"
    if not facts_dir.is_dir():
        print(f"✗ 사실 표 폴더가 없다: {facts_dir}")
        return 1

    facts = load_all(facts_dir)
    if not facts:
        print(f"✗ {facts_dir} 에 facts_*.csv 가 없다. 양식: data/facts/사실표_양식.csv")
        return 1

    print(f"사실 {len(facts)}건\n")
    dist = summarize(facts)
    for k, v in dist.items():
        print(f"  {k:<16} {v}")

    # 종별 쏠림은 골든셋 설계를 무너뜨린다 (04 §2.3)
    if dist["species"].get("bird", 0) == 0:
        print("\n  ⚠ 조류 0건 — 04 §2.3 종별 최소 건수를 만족할 수 없다")

    chunks = build_chunks(facts)
    print(f"\n청크 {len(chunks)}건 생성 (물질 단위 · D-14)")

    empty = [c for c in chunks if len(c.text.strip()) < 20]
    if empty:
        print(f"  ⚠ 문장이 지나치게 짧은 청크 {len(empty)}건 — 사실 표가 비어 있을 수 있다")
        for c in empty[:3]:
            print(f"      {c.chunk_id}: {c.text!r}")

    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        with args.out.open("w", encoding="utf-8") as f:
            for c in chunks:
                f.write(json.dumps(c.__dict__, ensure_ascii=False) + "\n")
        print(f"  → {args.out}")

    if args.store == "dry-run":
        print("\n예시 문장 3건")
        for c in chunks[:3]:
            print(f"  [{c.species}/{c.doc_type}] {c.text}")
        print("\n적재하려면 --store chroma")
        return 0

    from pettriage.retrieval import ChromaStore, HashEmbedder, InMemoryStore, get_embedder

    r = cfg.retrieval
    if args.store == "memory":
        # 모델 없이 배선만 확인한다. **검색 품질은 이 경로로 판단할 수 없다.**
        store = InMemoryStore(embedder=HashEmbedder())
        print("\n⚠ HashEmbedder — 배선 확인용이다. 검색 품질 판단에 쓰지 않는다")
    else:
        print(f"\n임베딩 모델 로딩: {r.embedding_model}")
        print("  처음이면 모델을 내려받는다 (bge-m3 약 2.2GB). 몇 분 걸릴 수 있다.")
        store = ChromaStore(
            embedder=get_embedder(r.embedding_model),
            persist_dir=str(root / r.persist_dir),
            collection=r.collection,
        )

    t0 = time.time()
    n = store.add(chunks)
    print(f"적재 {n}건 → {store.name} (총 {store.count()}건) · {time.time() - t0:.1f}s")
    if args.store == "chroma":
        print(f"  위치: {root / r.persist_dir}  ·  컬렉션: {r.collection}")

    if args.no_probe:
        return 0
    return 1 if probe(store, r.score_threshold, r.top_k) else 0


if __name__ == "__main__":
    sys.exit(main())
