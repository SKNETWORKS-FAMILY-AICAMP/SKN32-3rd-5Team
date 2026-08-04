#!/usr/bin/env python
"""⑤ 평이화 — distillation (④ 검증 산출물의 '근거있음' 문장 재사용).

설계 근거: docs/03_모델-멀티태스크학습.md §4

    ④에서 만든 문장 중 `intended_label == 근거있음`(실제 근거를 정확히
    반영하는, 진짜 수의학 용어가 섞인 문장) 407건을 입력 풀로 재사용한다
    — 새로 생성하지 않는다. 실제 프로덕션 SIMPLIFY 프롬프트로 평이화하고,
    **수치·단위가 보존됐는지**만 자동 검사한다(compress와 같은 방식).

    python scripts/generate_simplify_samples.py --out data/train/simplify_batch.jsonl
"""

from __future__ import annotations

import argparse
import json
import re
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))


def _with_retry(fn, tries: int = 3, base_delay: float = 2.0):
    for attempt in range(tries):
        try:
            return fn()
        except Exception as e:  # noqa: BLE001
            if attempt == tries - 1:
                raise
            wait = base_delay * (2**attempt)
            print(f"    ⚠ 호출 실패({type(e).__name__}) — {wait:.0f}초 후 재시도")
            time.sleep(wait)
    raise AssertionError("unreachable")


def _numbers_in(text: str) -> set[str]:
    return set(re.findall(r"\d+(?:\.\d+)?", text))


def _load_jargon_sentences() -> list[str]:
    """④ 검증 산출물 중 '근거있음'으로 의도된 문장 — 실제 근거를 담은 문장."""
    text = (ROOT / "data" / "train" / "verify_batch.jsonl").read_text(encoding="utf-8")
    rows = [json.loads(line) for line in text.splitlines() if line]
    seen: set[str] = set()
    out = []
    for r in rows:
        if r["intended_label"] == "근거있음" and r["sentence"] not in seen:
            seen.add(r["sentence"])
            out.append(r["sentence"])
    return out


def _already_done(out_path: Path) -> set[str]:
    if not out_path.exists():
        return set()
    done = set()
    for line in out_path.read_text(encoding="utf-8").splitlines():
        if line:
            done.add(json.loads(line)["source"])
    return done


def main() -> int:
    ap = argparse.ArgumentParser(description="⑤ 평이화 distillation")
    ap.add_argument("--out", type=Path, default=ROOT / "data" / "train" / "simplify_batch.jsonl")
    args = ap.parse_args()

    from pettriage.models.serving.factory import client_name, get_client
    from pettriage.models.tasks import Task

    client = get_client()
    if client is None:
        raise RuntimeError("LLM 클라이언트가 없다 — OPENAI_API_KEY 확인할 것")
    teacher = client_name()

    sentences = _load_jargon_sentences()
    done = _already_done(args.out)
    if done:
        print(f"이미 처리된 문장 {len(done)}건 — 건너뛰고 이어서 진행")
    print(f"용어 문장 {len(sentences)}건 (④ 검증 산출물 재사용) — 평이화 시작")

    args.out.parent.mkdir(parents=True, exist_ok=True)
    total = 0
    number_dropped = 0
    with args.out.open("a", encoding="utf-8") as f:
        for i, sentence in enumerate(sentences, start=1):
            if sentence in done:
                continue
            target = _with_retry(
                lambda s=sentence: client.run(Task.SIMPLIFY, s, max_tokens=300).strip()
            )
            missing_numbers = bool(_numbers_in(sentence) - _numbers_in(target))
            if missing_numbers:
                number_dropped += 1
            row = {
                "source": sentence,
                "target": target,
                "number_dropped": missing_numbers,
                "teacher": teacher,
            }
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
            f.flush()
            total += 1
            if i % 50 == 0:
                print(f"  {i}/{len(sentences)} · 누적 {total}건 · 수치 유실 {number_dropped}")

    print(f"\n평이화 {total}건 → {args.out}")
    print(f"수치·단위 유실 의심: {number_dropped}/{total}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
