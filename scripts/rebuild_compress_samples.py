#!/usr/bin/env python
"""③ 압축 — `report_batch.jsonl` → `samples.jsonl` 최종 반영.

D-83로 ③이 질의 경로에서 기간 리포트로 옮겨가면서, 기존 압축 학습 데이터
(407건 — 검색 결과를 압축하던 옛 형태)는 이제 존재하지 않는 노드를 향한다.
`generate_report_samples.py`로 새로 만든 `report_batch.jsonl`(다이어리 기록
집계 → Task.COMPRESS)로 통째로 교체한다.

target = LLM 출력 그대로. `fabricated` 플래그가 선 행은 **거른다** — 숫자
환각 의심을 학습 정답으로 삼지 않는다(사람이 다시 볼 여지는 report_batch.jsonl에
그대로 남아 있다).

split 규칙(기존 관행 유지): 10건마다 1건 dev.

    python scripts/rebuild_compress_samples.py --dry-run
    python scripts/rebuild_compress_samples.py
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--batch", type=Path, default=ROOT / "data" / "train" / "report_batch.jsonl")
    ap.add_argument("--samples", type=Path, default=ROOT / "data" / "train" / "samples.jsonl")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    batch_rows = [
        json.loads(line)
        for line in args.batch.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    kept_batch = [r for r in batch_rows if not r["fabricated"]]
    dropped_fabricated = len(batch_rows) - len(kept_batch)

    out_rows: list[dict] = []
    for i, row in enumerate(kept_batch):
        split = "dev" if (i + 1) % 10 == 0 else "train"
        out_rows.append(
            {
                "sample_id": f"report-batch-{i + 1:04d}",
                "task": "compress",
                "input": row["digest"],
                "target": row["target"],
                "origin": "distilled",
                "teacher": row["teacher"],
                "source_ids": [],
                "species": row["species"],
                "reviewed_by": "lse",
                "split": split,
            }
        )

    samples_text = args.samples.read_text(encoding="utf-8")
    old_rows = [json.loads(line) for line in samples_text.splitlines() if line.strip()]
    kept = [r for r in old_rows if r["task"] != "compress"]
    old_compress_count = len(old_rows) - len(kept)

    print(f"report 배치: {len(batch_rows)}건 (숫자 환각 의심 {dropped_fabricated}건 제외 → {len(kept_batch)}건)")
    print(f"samples.jsonl 기존 compress {old_compress_count}건(질의 경로용, 무효) → 새 compress {len(out_rows)}건(기간 리포트용)으로 교체")

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
