#!/usr/bin/env python
"""① 분류 태스크만으로 학습 — 파일럿 검증용/본학습 겸용.

설계 근거: docs/03a_파인튜닝-구현기획.md §3 "① 분류 태스크만으로 1차 학습 →
어댑터 산출 (03 §8: 분류만 붙어도 시스템은 동작)"

    `cfg.train.task_mix` 기본값은 5태스크 비율을 전부 요구한다 (default.yaml).
    `data/train/samples.jsonl` 에 classify 밖에 없는 동안은, task_mix를
    이 실행에서만 classify 100%로 덮는다 — **커밋된 configs/*.yaml 은 건드리지
    않는다.** 다른 태스크가 채워지면 이 스크립트는 버린다.

    실행: PETTRIAGE_PROFILE=train-local python scripts/run_classify_pilot_training.py \
        --out artifacts/adapters/classify-3k
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
ap.add_argument("--out", type=Path, default=ROOT / "artifacts" / "adapters" / "classify-pilot")
args = ap.parse_args()

cfg = get_config()
cfg.train.task_mix = {"classify": 1.0}

args.out.mkdir(parents=True, exist_ok=True)

print(f"profile={cfg.serve.engine!r} base_id={cfg.model.base_id!r} revision={cfg.model.revision!r}")
print(
    f"max_seq_len={cfg.model.max_seq_len} batch_size={cfg.train.batch_size} "
    f"grad_accum={cfg.train.grad_accum} lora_r={cfg.train.lora.r}"
)

result = run_training(args.data, args.out, cfg)
print("완료 —", result)
