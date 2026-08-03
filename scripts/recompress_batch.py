#!/usr/bin/env python
"""이미 검색해둔 raw_context로 압축만 다시 돌린다 (max_tokens 조정 후 재실행용).

검색은 비용·시간이 크고 이미 됐다 — 재검색하지 않는다. `target`만 다시 뽑는다.

실행: python scripts/recompress_batch.py --in data/train/compress_batch.jsonl \
    --out data/train/compress_batch.jsonl
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

ap = argparse.ArgumentParser()
ap.add_argument("--in", dest="in_path", type=Path, required=True)
ap.add_argument("--out", type=Path, required=True)
args = ap.parse_args()

from pettriage.models.serving.factory import get_client  # noqa: E402
from pettriage.models.tasks import Task  # noqa: E402

client = get_client()
if client is None:
    raise RuntimeError("LLM 클라이언트가 없다")

rows = [json.loads(line) for line in args.in_path.read_text(encoding="utf-8").splitlines() if line]


def _numbers_in(text: str) -> set[str]:
    text = re.sub(r"(?m)^\s*\d+\.\s", "", text)
    return set(re.findall(r"\d+(?:\.\d+)?", text))


for i, r in enumerate(rows, start=1):
    target = client.run(Task.COMPRESS, r["raw_context"], max_tokens=700).strip()
    r["target"] = target
    r["hallucinated_numbers"] = bool(_numbers_in(target) - _numbers_in(r["raw_context"]))
    if i % 50 == 0:
        print(f"  {i}/{len(rows)}")

args.out.parent.mkdir(parents=True, exist_ok=True)
with args.out.open("w", encoding="utf-8") as f:
    for r in rows:
        f.write(json.dumps(r, ensure_ascii=False) + "\n")

print(f"\n재압축 {len(rows)}건 → {args.out}")
