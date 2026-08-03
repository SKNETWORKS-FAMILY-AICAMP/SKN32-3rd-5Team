#!/usr/bin/env python
"""④ 근거 검증 — `verify_batch_raw.jsonl` → `samples.jsonl` 최종 반영.

배경: D-83으로 ③압축이 질의 경로에서 빠지면서, ③의 LLM 산출물을 ④의
"정답 근거 문서"로 쓰던 것이 순환 오염이 됐다. `generate_verify_samples.py`를
raw_context(③ 만들 때 캐시된 원본 검색 결과, LLM 미개입) 기반으로 고쳐
`verify_batch_raw.jsonl`을 새로 만들었고, 이 스크립트는 그것을 최종
학습 데이터 형태로 변환해 `samples.jsonl`의 기존 task=verify 행을 통째로
교체한다.

타깃 규칙(기존 관행 유지): **target = intended_label 항상.** 문장은
"이런 판정이 나오게" 의도적으로 만든 것이라 그 의도가 정답이다.
teacher_verdict(프로덕션 VERIFY로 교차검증한 값)는 QA 지표로만 쓴다 —
교사가 위험한 방향(근거없음을 근거있음으로)으로 틀렸을 때 그 오판을
그대로 정답으로 삼지 않기 위해서다.

split 규칙(기존 관행 유지): 10건마다 1건 dev (위치 10n → dev, 나머지 train).

    python scripts/rebuild_verify_samples.py --dry-run   # 반영 안 하고 통계만
    python scripts/rebuild_verify_samples.py              # samples.jsonl 반영
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--batch", type=Path, default=ROOT / "data" / "train" / "verify_batch_raw.jsonl")
    ap.add_argument("--samples", type=Path, default=ROOT / "data" / "train" / "samples.jsonl")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    batch_text = args.batch.read_text(encoding="utf-8")
    batch_rows = [json.loads(line) for line in batch_text.splitlines() if line.strip()]

    seen: set[tuple[str, str]] = set()
    new_rows: list[dict] = []
    for row in batch_rows:
        key = (row["sentence"], row["context"])
        if key in seen:
            continue
        seen.add(key)
        new_rows.append(row)

    dropped_dupes = len(batch_rows) - len(new_rows)

    dangerous = sum(
        1 for r in new_rows if r["intended_label"] == "근거없음" and r["teacher_verdict"] == "근거있음"
    )
    mismatches = sum(1 for r in new_rows if not r["agree"])

    out_rows: list[dict] = []
    for i, row in enumerate(new_rows):
        split = "dev" if (i + 1) % 10 == 0 else "train"
        out_rows.append(
            {
                "sample_id": f"verify-raw-{i + 1:04d}",
                "task": "verify",
                "input": f"문장: {row['sentence']}\n\n근거 문서:\n{row['context']}",
                "target": row["intended_label"],
                "origin": "distilled",
                "teacher": row["teacher"],
                "source_ids": [],
                "species": None,
                "reviewed_by": "lse",
                "split": split,
            }
        )

    samples_text = args.samples.read_text(encoding="utf-8")
    old_rows = [json.loads(line) for line in samples_text.splitlines() if line.strip()]
    kept = [r for r in old_rows if r["task"] != "verify"]
    old_verify_count = len(old_rows) - len(kept)

    print(f"raw 배치: {len(batch_rows)}건 (중복 {dropped_dupes}건 제거 → {len(new_rows)}건)")
    print(f"  위험 방향 불일치(근거없음→근거있음): {dangerous}건 — target은 intended_label(근거없음)로 고정해 안전 방향 유지")
    print(f"  전체 불일치(교사와 의도 라벨이 다른 것): {mismatches}/{len(new_rows)}")
    print(f"samples.jsonl 기존 verify {old_verify_count}건 → 새 verify {len(out_rows)}건으로 교체")

    if args.dry_run:
        print("\n--dry-run 이라 저장하지 않았다.")
        return 0

    final_rows = kept + out_rows
    with args.samples.open("w", encoding="utf-8") as f:
        for row in final_rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
    print(f"\n{args.samples} 저장 완료 — 총 {len(final_rows)}건.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
