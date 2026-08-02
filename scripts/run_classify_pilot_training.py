#!/usr/bin/env python
"""① 분류 태스크만으로 1차 학습 — 파이프라인이 실제로 도는지 확인용.

설계 근거: docs/03a_파인튜닝-구현기획.md §3 "① 분류 태스크만으로 1차 학습 →
어댑터 산출 (03 §8: 분류만 붙어도 시스템은 동작)"

    `cfg.train.task_mix` 기본값은 5태스크 비율을 전부 요구한다 (default.yaml).
    지금 `data/train/samples.jsonl` 에는 classify 밖에 없으므로, task_mix를
    이 실행에서만 classify 100%로 덮는다 — **커밋된 configs/*.yaml 은 건드리지
    않는다.** 다른 태스크가 채워지면 이 스크립트는 버린다.

    실행: PETTRIAGE_PROFILE=train-local python scripts/run_classify_pilot_training.py
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from pettriage.config import get_config  # noqa: E402
from pettriage.models.training.qlora import run_training  # noqa: E402

cfg = get_config()
cfg.train.task_mix = {"classify": 1.0}

out_dir = ROOT / "artifacts" / "adapters" / "classify-pilot"
out_dir.mkdir(parents=True, exist_ok=True)
data_path = ROOT / "data" / "train" / "samples.jsonl"

print(f"profile={cfg.serve.engine!r} base_id={cfg.model.base_id!r} revision={cfg.model.revision!r}")
print(
    f"max_seq_len={cfg.model.max_seq_len} batch_size={cfg.train.batch_size} "
    f"grad_accum={cfg.train.grad_accum} lora_r={cfg.train.lora.r}"
)

result = run_training(data_path, out_dir, cfg)
print("완료 —", result)
