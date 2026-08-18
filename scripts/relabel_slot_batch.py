#!/usr/bin/env python
"""이미 생성된 슬롯 배치를 고친 프롬프트로 재검증한다.

문장 생성은 비용이 크고 이미 검증됐다(패턴별로 잘 만들어짐) — 재생성하지 않는다.
`teacher_slots`(프로덕션 슬롯 프롬프트 결과)만 다시 뽑아 `agree`를 갱신한다.

실행: python scripts/relabel_slot_batch.py --in data/train/slot_batch.jsonl \
    --out data/train/slot_batch.jsonl
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

ap = argparse.ArgumentParser()
ap.add_argument("--in", dest="in_path", type=Path, required=True)
ap.add_argument("--out", type=Path, required=True)
args = ap.parse_args()

from pettriage.graph.nodes.slots import _llm_slots  # noqa: E402

rows = [json.loads(line) for line in args.in_path.read_text(encoding="utf-8").splitlines() if line]


def _slots_equal(a: dict, b: dict | None) -> bool:
    if b is None:
        return False
    keys = ("species", "weight_kg", "amount_g", "substance")
    return all(a.get(k) == b.get(k) for k in keys)


mismatches = 0
for i, r in enumerate(rows, start=1):
    teacher_slots = _llm_slots(r["question"])
    r["teacher_slots"] = teacher_slots
    r["agree"] = _slots_equal(r["gen_slots"], teacher_slots)
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
print("패턴별 불일치:", Counter(r["pattern"] for r in rows if not r["agree"]))
