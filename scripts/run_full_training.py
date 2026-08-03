#!/usr/bin/env python
"""5태스크 전체 데이터로 멀티태스크 QLoRA 학습.

`configs/default.yaml`의 `train.task_mix`(분류20%·슬롯20%·압축20%·
검증30%·평이화10%)를 그대로 쓴다 — 태스크별 수집량이 고르지 않아
(2026-08-03 기준: 분류 3,027 vs 평이화 185) `strict=False`로 부른다.
부족한 태스크는 있는 만큼만 쓰고 로그에 남는다.

실행: PETTRIAGE_PROFILE=train-local python scripts/run_full_training.py \
    --out artifacts/adapters/multitask-v1
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from pettriage.config import get_config  # noqa: E402
from pettriage.models.training.qlora import run_training  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--data", type=Path, default=ROOT / "data" / "train" / "samples.jsonl")
ap.add_argument("--out", type=Path, default=ROOT / "artifacts" / "adapters" / "multitask-v1")
args = ap.parse_args()

cfg = get_config()
args.out.mkdir(parents=True, exist_ok=True)

print(f"task_mix={cfg.train.task_mix}")
print(
    f"max_seq_len={cfg.model.max_seq_len} batch_size={cfg.train.batch_size} "
    f"grad_accum={cfg.train.grad_accum} lora_r={cfg.train.lora.r} epochs={cfg.train.epochs}"
)

result = run_training(args.data, args.out, cfg, strict=False)
print("완료 —", result)
