#!/usr/bin/env python
"""④ 근거 검증 — distillation (실제 근거 문서 + 프로덕션 VERIFY 프롬프트로 교차검증).

설계 근거: docs/03_모델-멀티태스크학습.md §4

    "정답 답변(positive) + 의도적 왜곡 답변(negative) 생성" — ③에서 이미
    실제 검색으로 만든 압축 결과(407건, 전부 진짜 코퍼스 근거)를 "근거
    문서" 풀로 재사용한다. 새로 검색하지 않는다.

    각 근거 문서마다 LLM에게 세 문장을 동시에 만들게 한다 — 문장과 정답을
    같이 받는 것은 ②슬롯과 같은 방식이다(만든 사람이 정답을 제일 잘 안다):
      · 근거있음: 근거 문서 내용을 정확히 반영하는 문장
      · 모순:     근거 문서의 특정 주장을 뒤집는 문장 (예: 위험↔안전, 종 교체)
      · 근거없음: 그럴듯하지만 이 근거 문서엔 없는 내용의 문장

    그 정답을 실제 프로덕션 VERIFY 경로(`graph.nodes.verify._llm_judge_sentence`)
    로 다시 판정해 교차검증한다 — 불일치는 사람 검수 대상.

    ⚠️ 실측(2026-08-03): 끝에 한 번에 파일을 쓰는 구조였다가 407건 중
    150여 건 처리한 시점에 네트워크 오류로 죽으면서 전부 날아갔다.
    그래서 **근거 문서 1건 처리할 때마다 즉시 append + flush**한다 —
    죽어도 그때까지 만든 건 남는다. 재실행하면 이미 처리한 근거 문서는
    건너뛴다(같은 --out 이어서 돌리면 이어서 진행된다).

    python scripts/generate_verify_samples.py --out data/train/verify_batch.jsonl
"""

from __future__ import annotations

import argparse
import json
import re
import sys
import time
from collections.abc import Callable
from pathlib import Path
from typing import TypeVar

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

_GEN_SYSTEM = (
    "너는 반려동물 헬스케어 서비스의 근거 검증 학습 데이터를 만드는 도구다.\n"
    "주어진 '근거 문서'를 보고 그것과의 관계가 다른 문장 3개를 만든다.\n"
    "출력은 JSON 객체 하나만:\n"
    '  {"grounded": str, "contradicted": str, "unsupported": str}\n'
    "- grounded: 근거 문서의 내용을 **정확히** 반영하는 한국어 문장 1개. "
    "근거에 없는 수치·종·단정을 추가하지 않는다.\n"
    "- contradicted: 근거 문서의 특정 주장을 **뒤집는** 문장 1개 "
    "(예: '위험하다'→'안전하다', 위험 종을 안전하다고 서술하는 종으로 바꿔치기 등). "
    "근거 문서에 있는 대상·주제는 그대로 두고 판단만 반대로 뒤집는다.\n"
    "- unsupported: 근거 문서의 주제와는 관련 있어 보이지만, "
    "**이 근거 문서에는 실제로 없는** 구체적 내용(다른 수치·다른 증상·다른 권고)을 "
    "말하는 문장 1개. 그럴듯해야 한다 — 티 나게 엉뚱한 얘기를 하지 않는다.\n"
    "설명·코드블록 없이 JSON 객체만 출력한다."
)

_T = TypeVar("_T")


def _with_retry(fn: Callable[[], _T], tries: int = 3, base_delay: float = 2.0) -> _T:
    """일시적 네트워크 오류를 재시도한다. 마지막 시도도 실패하면 그대로 던진다."""
    for attempt in range(tries):
        try:
            return fn()
        except Exception as e:  # noqa: BLE001 — 어떤 실패든 재시도 대상
            if attempt == tries - 1:
                raise
            wait = base_delay * (2**attempt)
            print(f"    ⚠ 호출 실패({type(e).__name__}) — {wait:.0f}초 후 재시도")
            time.sleep(wait)
    raise AssertionError("unreachable")


def _parse(raw: str) -> dict | None:
    m = re.search(r"\{.*\}", raw.strip(), re.DOTALL)
    if not m:
        return None
    try:
        data = json.loads(m.group(0))
    except json.JSONDecodeError:
        return None
    if not isinstance(data, dict):
        return None
    keys = ("grounded", "contradicted", "unsupported")
    if not all(isinstance(data.get(k), str) and data[k].strip() for k in keys):
        return None
    return {k: data[k].strip() for k in keys}


def _load_context_pool() -> list[str]:
    """③ 압축 산출물(407건, 전부 실제 코퍼스 근거)을 근거 문서 풀로 재사용한다."""
    text = (ROOT / "data" / "train" / "samples.jsonl").read_text(encoding="utf-8")
    rows = [json.loads(line) for line in text.splitlines() if line]
    return [r["target"] for r in rows if r["task"] == "compress"]


def _already_done(out_path: Path) -> set[str]:
    """이미 처리된 근거 문서 집합 — 재실행 시 건너뛴다."""
    if not out_path.exists():
        return set()
    done = set()
    for line in out_path.read_text(encoding="utf-8").splitlines():
        if line:
            done.add(json.loads(line)["context"])
    return done


def main() -> int:
    ap = argparse.ArgumentParser(description="④ 근거 검증 distillation")
    ap.add_argument("--limit", type=int, default=0, help="0이면 근거 문서 풀 전체")
    ap.add_argument("--out", type=Path, default=ROOT / "data" / "train" / "verify_batch.jsonl")
    args = ap.parse_args()

    from pettriage.graph.nodes.verify import _llm_judge_sentence
    from pettriage.models.serving.factory import client_name, get_client

    client = get_client()
    if client is None:
        raise RuntimeError("LLM 클라이언트가 없다 — OPENAI_API_KEY 확인할 것")
    teacher = client_name()

    contexts = _load_context_pool()
    if args.limit:
        contexts = contexts[: args.limit]

    done = _already_done(args.out)
    if done:
        print(f"이미 처리된 근거 문서 {len(done)}건 — 건너뛰고 이어서 진행")

    print(f"근거 문서 {len(contexts)}건 (③ 압축 산출물 재사용) — 생성·교차검증 시작")

    args.out.parent.mkdir(parents=True, exist_ok=True)
    total_rows = 0
    mismatches = 0
    parse_fail = 0
    processed = 0
    with args.out.open("a", encoding="utf-8") as f:
        for i, ctx in enumerate(contexts, start=1):
            if ctx in done:
                continue
            processed += 1

            raw = _with_retry(
                lambda ctx=ctx: client.run_raw(_GEN_SYSTEM, f"근거 문서:\n{ctx}", max_tokens=500)
            )
            parsed = _parse(raw)
            if not parsed:
                parse_fail += 1
                continue

            for label, sentence in (
                ("근거있음", parsed["grounded"]),
                ("모순", parsed["contradicted"]),
                ("근거없음", parsed["unsupported"]),
            ):
                teacher_verdict = _with_retry(lambda s=sentence, c=ctx: _llm_judge_sentence(s, c))
                agree = teacher_verdict == label
                if not agree:
                    mismatches += 1
                row = {
                    "sentence": sentence,
                    "context": ctx,
                    "intended_label": label,
                    "teacher_verdict": teacher_verdict,
                    "agree": agree,
                    "teacher": teacher,
                }
                f.write(json.dumps(row, ensure_ascii=False) + "\n")
                f.flush()
                total_rows += 1

            if i % 50 == 0:
                print(f"  {i}/{len(contexts)} · 누적 {total_rows}건 · 불일치 {mismatches}")

    print(f"\n이번 실행에서 처리: {processed}건 · 생성 {total_rows}건 → {args.out}")
    print(f"생성 파싱 실패(근거 문서 스킵): {parse_fail}")
    print(f"의도 라벨 vs 교사(LLM) 불일치: {mismatches}/{total_rows}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
