#!/usr/bin/env python
"""이미 생성된 검증 배치를 고친 프롬프트로 재검증한다 (문장 재생성 없이).

실행: python scripts/relabel_verify_batch.py --in data/train/verify_batch.jsonl \
    --out data/train/verify_batch.jsonl
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

ap = argparse.ArgumentParser()
ap.add_argument("--in", dest="in_path", type=Path, required=True)
ap.add_argument("--out", type=Path, required=True)
args = ap.parse_args()

from pettriage.graph.nodes.verify import _llm_judge_sentence  # noqa: E402

rows = [json.loads(line) for line in args.in_path.read_text(encoding="utf-8").splitlines() if line]

mismatches = 0
for i, r in enumerate(rows, start=1):
    for attempt in range(3):
        try:
            verdict = _llm_judge_sentence(r["sentence"], r["context"])
            break
        except Exception as e:  # noqa: BLE001
            if attempt == 2:
                raise
            print(f"    ⚠ 재시도 ({type(e).__name__})")
            time.sleep(2 * (2**attempt))
    r["teacher_verdict"] = verdict
    r["agree"] = verdict == r["intended_label"]
    if not r["agree"]:
        mismatches += 1
    if i % 100 == 0:
        print(f"  재검증 {i}/{len(rows)} · 불일치 {mismatches}")

args.out.parent.mkdir(parents=True, exist_ok=True)
with args.out.open("w", encoding="utf-8") as f:
    for r in rows:
        f.write(json.dumps(r, ensure_ascii=False) + "\n")

print(f"\n재검증 {len(rows)}건 → {args.out}")
print(f"불일치: {mismatches}/{len(rows)}")
