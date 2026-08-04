#!/usr/bin/env python
"""분류·슬롯·검증(정답이 명확한 태스크) — 베이스 vs 어댑터 held-out 비교.

압축·평이화는 생성형이라 정확도로 안 잰다(눈으로 확인 대상).

실행: PETTRIAGE_PROFILE=train-local python scripts/compare_multitask_adapter.py \
    --adapter artifacts/adapters/multitask-v1/checkpoint-254 --per-task 20
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
ap.add_argument("--adapter", type=Path, required=True)
ap.add_argument("--per-task", type=int, default=20, help="태스크당 dev 표본 수")
args = ap.parse_args()

cfg = get_config()
_TASK_ENUM = {"classify": Task.CLASSIFY, "slot": Task.SLOT, "verify": Task.VERIFY}

rows = [
    json.loads(line)
    for line in (ROOT / "data" / "train" / "samples.jsonl").read_text(encoding="utf-8").splitlines()
    if line
]
dev = [r for r in rows if r["split"] == "dev" and r["task"] in _TASK_ENUM]

by_task: dict[str, list[dict]] = {}
for r in dev:
    by_task.setdefault(r["task"], []).append(r)
sample: list[dict] = []
for items in by_task.values():
    sample.extend(items[: args.per_task])

print(f"평가 대상: {len(sample)}건 ({dict(Counter(r['task'] for r in sample))})\n")

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

base_correct: Counter = Counter()
tuned_correct: Counter = Counter()
task_n: Counter = Counter()

for i, r in enumerate(sample, start=1):
    task_enum = _TASK_ENUM[r["task"]]
    gold = r["target"]
    task_n[r["task"]] += 1

    b = base.run(task_enum, r["input"], max_tokens=64).strip()
    t = tuned.run(task_enum, r["input"], max_tokens=64).strip()

    # classify/verify: 정답 라벨이 문자열에 그대로 들어있는지로 판정(관대하게).
    # slot: JSON 파싱 후 딕셔너리 비교.
    if r["task"] == "slot":
        try:
            b_ok = json.loads(b[b.find("{") : b.rfind("}") + 1]) == json.loads(gold)
        except Exception:  # noqa: BLE001
            b_ok = False
        try:
            t_ok = json.loads(t[t.find("{") : t.rfind("}") + 1]) == json.loads(gold)
        except Exception:  # noqa: BLE001
            t_ok = False
    else:
        b_ok = gold in b
        t_ok = gold in t

    base_correct[r["task"]] += b_ok
    tuned_correct[r["task"]] += t_ok
    if i % 10 == 0:
        print(f"  {i}/{len(sample)}")

print("\n" + "-" * 60)
for task in task_n:
    n = task_n[task]
    print(f"{task:10s} n={n:3d}  베이스 {base_correct[task]}/{n}  어댑터 {tuned_correct[task]}/{n}")
