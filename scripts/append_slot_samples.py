#!/usr/bin/env python
"""② 슬롯 — `slot_batch.jsonl`의 특정 패턴만 골라 `samples.jsonl`에 **추가**한다.

기존 1,907건은 건드리지 않는다(교체 아님, 추가). 배경: D-89로 SLOT 출력
스키마에 `elapsed_hours`가 명시됐는데(D-86) 그 축의 학습 데이터가 전혀
없었다 — `check_slot_samples.py`로 기존 데이터엔 null만 채워 넣었고,
이 스크립트로 실제 값이 있는 케이스를 보탠다.

타깃 규칙(기존 관행과 동일 — 실측으로 확인함): 생성값(gen_slots)과 교사
판정(teacher_slots)이 다르면 **교사 쪽을 쓴다.** 사람이 만든 문장의
"의도"보다 프로덕션이 실제로 뽑아내는 값이 진짜 학습 대상이다(반대로
④검증은 의도가 안전 방향의 정답이라 반대 규칙을 쓴다 — 태스크 성격이
다르다).

    python scripts/append_slot_samples.py --pattern elapsed_hours_mentioned
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

_KEYS = ("species", "substance", "weight_kg", "amount_g", "elapsed_hours")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--batch", type=Path, default=ROOT / "data" / "train" / "slot_batch.jsonl")
    ap.add_argument("--samples", type=Path, default=ROOT / "data" / "train" / "samples.jsonl")
    ap.add_argument("--pattern", required=True, help="slot_batch.jsonl 안의 pattern 필드로 필터")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    batch_rows = [
        json.loads(line)
        for line in args.batch.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    picked = [r for r in batch_rows if r.get("pattern") == args.pattern]
    if not picked:
        raise SystemExit(f"패턴 {args.pattern!r} 에 해당하는 행이 없다.")

    existing = [
        json.loads(line)
        for line in args.samples.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    existing_slot_nums = [
        int(r["sample_id"].split("-")[-1]) for r in existing if r["task"] == "slot"
    ]
    next_id = (max(existing_slot_nums) if existing_slot_nums else 0) + 1

    seen_questions = {r["input"] for r in existing if r["task"] == "slot"}

    new_rows: list[dict] = []
    teacher_used = 0
    gen_used = 0
    skipped_dupe = 0
    for r in picked:
        if r["question"] in seen_questions:
            skipped_dupe += 1
            continue
        seen_questions.add(r["question"])

        if r["teacher_slots"] is not None:
            target_slots = {k: r["teacher_slots"].get(k) for k in _KEYS}
            teacher_used += 1
        else:
            target_slots = {k: r["gen_slots"].get(k) for k in _KEYS}
            gen_used += 1

        new_rows.append(
            {
                "sample_id": f"slot-batch-{next_id:04d}",
                "task": "slot",
                "input": r["question"],
                "target": json.dumps(target_slots, ensure_ascii=False),
                "origin": "distilled",
                "teacher": r["teacher"],
                "source_ids": [],
                "species": target_slots.get("species"),
                "reviewed_by": "lse",
                "split": "dev" if next_id % 10 == 0 else "train",
            }
        )
        next_id += 1

    print(
        f"패턴 {args.pattern!r}: 후보 {len(picked)}건 · "
        f"중복 스킵 {skipped_dupe}건 · 추가 {len(new_rows)}건"
    )
    print(f"  target = 교사 판정 {teacher_used}건 · 생성값(교사 실패) {gen_used}건")

    if args.dry_run:
        print("\n--dry-run 이라 저장하지 않았다.")
        return 0

    with args.samples.open("a", encoding="utf-8") as f:
        for row in new_rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
    print(f"\n{args.samples} 에 {len(new_rows)}건 추가 완료.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
