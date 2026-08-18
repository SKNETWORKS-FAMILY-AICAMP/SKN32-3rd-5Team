#!/usr/bin/env python
"""② 슬롯 학습 데이터 정합성 검사기 — 재실행 가능, --fix 는 세 가지만 고친다.

배경: D-89(2026-08-03)가 ② SLOT 프롬프트에 출력 스키마를 명시하면서
(D-86) `species·substance·weight_kg·amount_g·elapsed_hours` 5키가
**항상 다 있어야** 하는 것으로 계약이 바뀌었다. 그런데 그 전에 만든
`data/train/samples.jsonl`의 slot 타깃(1,907건)은:

  - 값이 없는 슬롯은 키 자체를 뺐다 (새 계약은 `null`로 채워야 한다)
  - `elapsed_hours`가 애초에 생성 대상에 없어 단 한 건도 없다
  - 초기 데이터 일부가 `species`를 한국어(개·고양이)로 냈을 수 있다

`output_keys`는 `SPECS[Task.SLOT]`에서 읽는다 — 여기 손으로 다시 적지 않는다
(D-73 · D-86, 이 파일도 같은 원칙을 따른다).

    python scripts/check_slot_samples.py                # 리포트만
    python scripts/check_slot_samples.py --fix           # 위 세 가지만 고쳐 저장

⚠️ `amount_g 개수 의심`은 --fix 로도 절대 안 고친다. "3g 먹었다"의 3과
"3알 먹었다"의 3을 스크립트가 못 가린다 — 입력 문장에 개·알·줌·조금 같은
개수 표현이 있는데 amount_g가 차 있으면 표시만 하고 사람이 판정한다.
안 그러면 D-38(코드가 판단을 지어내지 않는다)을 데이터 쪽에서 어기는 것이다.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from pettriage.models.tasks import SPECS, Task  # noqa: E402

#: 한국어 종 표기 → 코드가 쓰는 영문 값. `compute.vocabulary.SPECIES_WORDS`와
#: 어휘 자체는 겹치지만, 여기는 "타깃에 이미 박힌 오류값을 고치는" 좁은 용도라
#: 별도로 둔다 — 질문 문장을 훑는 추출기가 아니다.
_SPECIES_KO_TO_EN = {
    "개": "dog",
    "강아지": "dog",
    "고양이": "cat",
    "냥이": "cat",
    "앵무새": "bird",
    "새": "bird",
}

#: amount_g가 실은 "개수"였을 가능성을 표시하는 어휘 — D-89 key_hints의
#: "하나·한 알·한 줌·조금" 예시를 그대로 따른다(단일 출처를 tasks.py에 두되,
#: 이건 자유 텍스트 안 문자열 매칭이라 이 파일에 둔다).
_COUNT_WORDS = ("개", "알", "줌", "조금")


def _load(path: Path) -> list[dict]:
    text = path.read_text(encoding="utf-8")
    return [json.loads(line) for line in text.splitlines() if line.strip()]


def main() -> int:
    ap = argparse.ArgumentParser(description="② 슬롯 학습 데이터 정합성 검사")
    ap.add_argument("--path", type=Path, default=ROOT / "data" / "train" / "samples.jsonl")
    ap.add_argument(
        "--fix", action="store_true", help="키 누락·스키마 밖 키·species 한글만 고쳐 저장"
    )
    args = ap.parse_args()

    output_keys = SPECS[Task.SLOT].output_keys
    rows = _load(args.path)

    missing_key_rows = 0
    off_schema_rows = 0
    off_schema_names: set[str] = set()
    species_ko_rows = 0
    amount_suspect: list[tuple[str, str, object]] = []

    out_rows: list[dict] = []
    for row in rows:
        if row.get("task") != "slot":
            out_rows.append(row)
            continue

        raw_target = row["target"]
        target = json.loads(raw_target) if isinstance(raw_target, str) else dict(raw_target)

        off = [k for k in target if k not in output_keys]
        if off:
            off_schema_rows += 1
            off_schema_names.update(off)
            if args.fix:
                for k in off:
                    del target[k]

        missing = [k for k in output_keys if k not in target]
        if missing:
            missing_key_rows += 1
            if args.fix:
                for k in missing:
                    target[k] = None

        species = target.get("species")
        if isinstance(species, str) and species in _SPECIES_KO_TO_EN:
            species_ko_rows += 1
            if args.fix:
                target["species"] = _SPECIES_KO_TO_EN[species]

        amount = target.get("amount_g")
        question = row.get("input", "")
        if amount is not None and any(w in question for w in _COUNT_WORDS):
            amount_suspect.append((row.get("sample_id", "?"), question, amount))
            # 🔴 자동 수정 안 함 — 표시만.

        row = dict(row)
        row["target"] = (
            json.dumps(target, ensure_ascii=False) if isinstance(raw_target, str) else target
        )
        out_rows.append(row)

    total_slot = sum(1 for r in rows if r.get("task") == "slot")

    print(f"슬롯 데이터 {total_slot}건 검사 (output_keys = {output_keys})\n")
    tag = " → 수정함" if args.fix else ""
    print(f"  키 누락             {missing_key_rows:>5}건  → null 채움{tag}")
    print(f"  스키마 밖 키         {off_schema_rows:>5}건  → 버림{tag}  {sorted(off_schema_names)}")
    print(f"  species 한글         {species_ko_rows:>5}건  → dog/cat/bird 로{tag}")
    print(
        f"  amount_g 개수 의심    {len(amount_suspect):>5}건  → 🔴 사람이 봐야 함 (자동 수정 금지)"
    )
    if amount_suspect:
        for sid, q, amt in amount_suspect[:30]:
            print(f"    {sid}: {q!r}  amount_g={amt}")
        if len(amount_suspect) > 30:
            print(f"    ... 외 {len(amount_suspect) - 30}건")

    if args.fix:
        with args.path.open("w", encoding="utf-8") as f:
            for row in out_rows:
                f.write(json.dumps(row, ensure_ascii=False) + "\n")
        print(f"\n{args.path} 저장 완료. amount_g 의심 건은 그대로 두었다 — 사람이 확인할 것.")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
