#!/usr/bin/env python
"""분류(①) 태스크 — 베이스 Qwen3-4B vs LoRA 어댑터 출력 비교.

`data/train/samples.jsonl`의 dev 분할(학습에 안 쓰인 5건)로 확인한다 —
train 분할로 확인하면 "외운 것"과 "일반화한 것"을 구분할 수 없다.

실행: PETTRIAGE_PROFILE=train-local python scripts/compare_classify_adapter.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from pettriage.config import get_config  # noqa: E402
from pettriage.models.serving.client import LocalQwenClient  # noqa: E402
from pettriage.models.tasks import Task  # noqa: E402

cfg = get_config()

rows = [
    json.loads(line)
    for line in (ROOT / "data" / "train" / "samples.jsonl").read_text(encoding="utf-8").splitlines()
    if line
]
dev_rows = [r for r in rows if r["split"] == "dev"]

print(f"dev {len(dev_rows)}건으로 비교 (학습에 안 쓰인 held-out)\n")

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
    adapter_path=str(ROOT / "artifacts" / "adapters" / "classify-pilot"),
    revision=cfg.model.revision,
    dtype=cfg.model.dtype,
    load_in_4bit=cfg.model.load_in_4bit,
)

correct_base = 0
correct_tuned = 0
print(f"\n{'질문':<40} {'정답':<14} {'베이스':<20} {'어댑터':<20}")
print("-" * 100)
for r in dev_rows:
    q, gold = r["input"], r["target"]
    b = base.run(Task.CLASSIFY, q, max_tokens=16).strip()
    t = tuned.run(Task.CLASSIFY, q, max_tokens=16).strip()
    correct_base += b == gold
    correct_tuned += t == gold
    print(f"{q:<40} {gold:<14} {b:<20} {t:<20}")

print("-" * 100)
n = len(dev_rows)
print(f"베이스 정확도: {correct_base}/{n} · 어댑터 정확도: {correct_tuned}/{n}")
