#!/usr/bin/env python
"""사실 표 → 청크 → 벡터DB 적재.

    python scripts/build_index.py --dry-run     # 청크만 만들어 보고 (기본)
    python scripts/build_index.py --store chroma

설계 근거: docs/02 §3 · docs/06 D-14 · D-20 · D-38

**문장화는 코드가 한다.** 여기서 LLM 을 부르지 않는다.
표가 맞으면 문장도 맞는다 — 그래서 검증은 표 단계에서 끝난다 (01e).
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from pettriage import paths  # noqa: E402
from pettriage.config import get_config  # noqa: E402
from pettriage.ingest.facts_io import build_chunks, load_all, summarize  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser(description="사실 표를 벡터DB에 적재한다")
    ap.add_argument("--facts-dir", type=Path, default=None)
    ap.add_argument("--store", choices=["dry-run", "memory", "chroma"], default="dry-run")
    ap.add_argument("--out", type=Path, help="청크를 JSONL 로 저장 (검수용)")
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

    cfg = get_config()
    if args.store == "memory":
        store = InMemoryStore(embedder=HashEmbedder())
    else:
        store = ChromaStore(
            embedder=get_embedder(cfg.retrieval.embedding_model),
            persist_dir=str(root / ".chroma"),
        )
    n = store.add(chunks)
    print(f"\n적재 {n}건 → {store.name} (총 {store.count()}건)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
