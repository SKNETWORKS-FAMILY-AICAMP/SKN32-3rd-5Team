#!/usr/bin/env python
"""분류(①) 태스크 — 베이스 Qwen3-4B vs LoRA 어댑터 출력 비교.

`data/train/samples.jsonl`의 dev 분할(학습에 안 쓰인 held-out)로 확인한다 —
train 분할로 확인하면 "외운 것"과 "일반화한 것"을 구분할 수 없다.

실행: PETTRIAGE_PROFILE=train-local python scripts/compare_classify_adapter.py \
    --adapter artifacts/adapters/classify-3k
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from pettriage.config import get_config  # noqa: E402
from pettriage.models.serving.client import LocalQwenClient  # noqa: E402
from pettriage.models.tasks import Task  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--adapter", type=Path, default=ROOT / "artifacts" / "adapters" / "classify-3k")
ap.add_argument("--limit", type=int, default=0, help="0이면 dev 전체")
args = ap.parse_args()

cfg = get_config()

rows = [
    json.loads(line)
    for line in (ROOT / "data" / "train" / "samples.jsonl").read_text(encoding="utf-8").splitlines()
    if line
]
dev_rows = [r for r in rows if r["split"] == "dev"]
if args.limit:
    dev_rows = dev_rows[: args.limit]

print(f"dev {len(dev_rows)}건으로 비교 (학습에 안 쓰인 held-out) · adapter={args.adapter}\n")

print("① 베이스 모델 로드 중...")
base = LocalQwenClient(
    cfg.model.base_id,
    adapter_path=None,
    revision=cfg.model.revision,
    dtype=cfg.model.dtype,
    load_in_4bit=cfg.model.load_in_4bit,
)

print("② 어댑터 모델 로드 중...")
tuned = LocalQwenClient(
    cfg.model.base_id,
    adapter_path=str(args.adapter),
    revision=cfg.model.revision,
    dtype=cfg.model.dtype,
    load_in_4bit=cfg.model.load_in_4bit,
)

correct_base = 0
correct_tuned = 0
base_confusion: Counter = Counter()
tuned_confusion: Counter = Counter()
mismatch_examples: list[tuple[str, str, str, str]] = []

n = len(dev_rows)
for i, r in enumerate(dev_rows, start=1):
    q, gold = r["input"], r["target"]
    b = base.run(Task.CLASSIFY, q, max_tokens=16).strip()
    t = tuned.run(Task.CLASSIFY, q, max_tokens=16).strip()
    ok_b, ok_t = b == gold, t == gold
    correct_base += ok_b
    correct_tuned += ok_t
    if not ok_b:
        base_confusion[(gold, b)] += 1
    if not ok_t:
        tuned_confusion[(gold, t)] += 1
        mismatch_examples.append((q, gold, b, t))
    if i % 50 == 0:
        print(f"  {i}/{n} · 베이스 {correct_base}/{i} · 어댑터 {correct_tuned}/{i}")

print("-" * 100)
print(f"베이스 정확도: {correct_base}/{n} ({correct_base / n:.1%})")
print(f"어댑터 정확도: {correct_tuned}/{n} ({correct_tuned / n:.1%})")

print("\n어댑터가 여전히 틀린 케이스 (최대 20건):")
for q, gold, b, t in mismatch_examples[:20]:
    print(f"  [{gold}->{t}] (베이스는 {b}) {q}")

print("\n어댑터 오답 혼동 패턴 (정답->예측):")
for (gold, pred), cnt in tuned_confusion.most_common(10):
    print(f"  {gold} -> {pred}: {cnt}건")
