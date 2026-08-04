#!/usr/bin/env python
"""③ 기간 리포트 — distillation (합성 다이어리 기록 → 진짜 _digest() → Task.COMPRESS).

배경(D-83): ③이 질의 경로에서 빠지고 기간 리포트(`app/report.py::summarize_period`)로
옮겨갔다. 입력 형태가 완전히 바뀌었다 — "검색된 문서"가 아니라 "다이어리 기록
여러 건을 코드가 집계한 텍스트"다.

학습 데이터의 입력은 **진짜 프로덕션 집계 함수(`report._digest`)의 출력**을 그대로
쓴다. 손으로 비슷하게 흉내 내면 실제 입력과 미묘하게 달라질 위험이 있다 —
`_digest`를 다시 구현하지 않고 그대로 import 해서 부른다(단일 출처).

    1. 합성 반려동물 다이어리 기록(증상·급여·배변·체중)을 만든다.
       구조화 데이터라 자연어 생성이 필요 없다 — LLM 없이 코드로 만든다.
    2. `report._digest(rows, ..., include_notes=False)`로 집계 텍스트를 만든다.
       (`include_notes=False`인 이유 — `_notes_may_leave_the_machine()`이 지금은
       항상 False다(D-36, privacy 필터 미구현). 프로덕션이 LLM에게 실제로 주는
       입력과 다르면 학습 데이터가 또 다른 경로를 향하게 된다.)
    3. 그 집계 텍스트로 production Task.COMPRESS를 불러 target을 받는다.
    4. 집계에 없는 숫자가 target에 나왔는지 거칠게 검사한다(D-83이 걱정한 문제) —
       문자열 비교라 "5.30" vs "5.3" 같은 표기 차이도 걸릴 수 있다. **자동 판정이
       아니라 사람이 볼 후보 표시일 뿐이다.**

    python scripts/generate_report_samples.py --target 200 --out data/train/report_batch.jsonl
"""

from __future__ import annotations

import argparse
import json
import random
import re
import sys
from datetime import date, timedelta
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

_SPECIES = ("dog", "cat", "bird")

_SYMPTOMS = [
    "구토",
    "설사",
    "기침",
    "재채기",
    "가려움",
    "무기력",
    "식욕부진",
    "콧물",
    "눈곱",
    "딸꾹질",
    "탈모",
    "헐떡임",
]
_MEALS = ["사료", "간식", "물", "닭가슴살", "고구마", "황태", "채소"]
_DROPPINGS = ["정상", "묽음", "양 감소", "색 변화", "혈액 섞임"]

#: 종별 현실적인 체중 범위(kg). 조류는 g 단위가 흔하지만 스키마가 kg 고정이라 맞춘다.
_WEIGHT_RANGE = {"dog": (3.0, 30.0), "cat": (2.5, 6.5), "bird": (0.02, 1.0)}


def _rand_weight(species: str) -> float:
    lo, hi = _WEIGHT_RANGE[species]
    return round(random.uniform(lo, hi), 2)


def _gen_period(period_days: int, n_records: int, species: str) -> tuple[list[dict], str, str]:
    start = date(2026, random.randint(1, 6), random.randint(1, 28))
    end = start + timedelta(days=period_days)
    dates = sorted(start + timedelta(days=random.randint(0, period_days)) for _ in range(n_records))

    base_weight = _rand_weight(species)
    drift = random.uniform(-0.15, 0.15) * base_weight  # 기간 동안 체중이 서서히 변한다

    rows: list[dict] = []
    for i, d in enumerate(dates):
        row: dict = {"recorded_at": d.isoformat(), "note": ""}
        if random.random() < 0.6:
            row["symptoms"] = random.sample(_SYMPTOMS, k=random.randint(1, 2))
        if random.random() < 0.7:
            row["meals"] = random.sample(_MEALS, k=random.randint(1, 2))
        if species == "bird" and random.random() < 0.5:
            row["droppings"] = random.choice(_DROPPINGS)
        if random.random() < 0.5:
            frac = i / max(1, len(dates) - 1)
            row["weight_kg"] = round(base_weight + drift * frac, 2)
        rows.append(row)

    return rows, start.isoformat(), end.isoformat()


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--target", type=int, default=200)
    ap.add_argument("--out", type=Path, default=ROOT / "data" / "train" / "report_batch.jsonl")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    random.seed(args.seed)

    from pettriage.app.report import _digest
    from pettriage.models.serving.factory import client_name, get_client
    from pettriage.models.tasks import Task

    client = get_client()
    if client is None:
        raise RuntimeError("LLM 클라이언트가 없다 — OPENAI_API_KEY 확인할 것")
    teacher = client_name()

    args.out.parent.mkdir(parents=True, exist_ok=True)
    total = 0
    fabricated = 0
    with args.out.open("w", encoding="utf-8") as f:
        for i in range(args.target):
            species = random.choice(_SPECIES)
            period_days = random.choice([7, 14, 30])
            n_records = random.randint(3, 20)
            rows, period_from, period_to = _gen_period(period_days, n_records, species)

            digest = _digest(rows, period_from, period_to, include_notes=False)

            try:
                target = client.run(Task.COMPRESS, digest, max_tokens=700).strip()
            except Exception as e:  # noqa: BLE001
                print(f"  ⚠ [{i}] 호출 실패: {type(e).__name__}")
                continue

            digest_nums = set(re.findall(r"\d+\.?\d*", digest))
            target_nums = set(re.findall(r"\d+\.?\d*", target))
            extra_nums = sorted(target_nums - digest_nums)
            is_fabricated = bool(extra_nums)
            if is_fabricated:
                fabricated += 1

            f.write(
                json.dumps(
                    {
                        "species": species,
                        "period_from": period_from,
                        "period_to": period_to,
                        "n_records": len(rows),
                        "digest": digest,
                        "target": target,
                        "extra_nums": extra_nums,
                        "fabricated": is_fabricated,
                        "teacher": teacher,
                    },
                    ensure_ascii=False,
                )
                + "\n"
            )
            f.flush()
            total += 1
            if (i + 1) % 20 == 0:
                print(f"  {i + 1}/{args.target} · 누적 {total}건 · 숫자 의심 {fabricated}")

    print(f"\n생성 완료: {total}/{args.target} · 숫자 환각 의심 {fabricated}건 → {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
