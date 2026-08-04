#!/usr/bin/env python
"""③ 압축(요약) 태스크 — distillation (실제 검색 결과 + 프로덕션 COMPRESS 프롬프트).

설계 근거: docs/03_모델-멀티태스크학습.md §4

    ①②와 다른 점 — 입력이 "질문 문장"이 아니라 **"검색된 문서 여러 개"**다.
    지어내지 않는다: 새 질문을 만드는 대신, 이미 만든 ①분류 학습 데이터의
    intoxication 질문(753건, 전부 실제 코퍼스 물질을 언급한다)을 재사용해
    **진짜 로컬 벡터DB로 진짜 검색**을 돌린다 — `graph.nodes.retrieve.retrieve`
    그대로. 가짜 컨텍스트를 만들지 않는다 (D-16 원칙과 같은 이유).

    압축이 필요한 경우만 학습시킨다 — `compress_context`가 실제로 LLM을
    부르는 조건(길이 ≥ 800자)과 같다. 짧은 컨텍스트는 애초에 압축을 안 하므로
    그런 샘플을 학습시키면 실제 분포와 어긋난다.

    자동 품질 검사: target에 있는 숫자가 raw 컨텍스트에 없으면 **환각 의심**으로
    표시한다 — compress_context 자신의 규칙("원문에 없는 수치를 추가하지 않는다")
    을 데이터 쪽에서도 검증한다.

    python scripts/generate_compress_samples.py --limit 400 \
        --out data/train/compress_batch.jsonl
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

_COMPRESS_LEN_THRESHOLD = 800  # graph.nodes.generate._COMPRESS_LEN_THRESHOLD 와 동일


def _load_candidate_questions() -> list[dict]:
    """이미 만든 ①②(분류+슬롯) 샘플 전체 — 실제 종·물질을 언급하는 질문 풀.

    intoxication 라벨로만 좁히면 후보가 753건뿐이라, 실측 수율(~10%)로는
    목표 건수를 못 채운다. ①②를 합친 4,934건 전체를 후보로 넓힌다 —
    새로 생성하지 않고 이미 검수된 질문을 재사용한다(비용 없음).
    """
    text = (ROOT / "data" / "train" / "samples.jsonl").read_text(encoding="utf-8")
    rows = [json.loads(line) for line in text.splitlines() if line]
    seen: set[str] = set()
    out = []
    for r in rows:
        if r["input"] in seen:
            continue
        seen.add(r["input"])
        out.append({"input": r["input"], "species": r.get("species")})
    return out


def _numbers_in(text: str) -> set[str]:
    return set(re.findall(r"\d+(?:\.\d+)?", text))


def _numbers_in(text: str) -> set[str]:
    return set(re.findall(r"\d+(?:\.\d+)?", text))


def main() -> int:
    ap = argparse.ArgumentParser(description="③ 압축 distillation")
    ap.add_argument("--limit", type=int, default=400, help="시도할 질문 개수 상한")
    ap.add_argument("--out", type=Path, default=ROOT / "data" / "train" / "compress_batch.jsonl")
    args = ap.parse_args()

    from pettriage.graph.nodes.retrieve import build_filter, retrieve
    from pettriage.graph.state import initial_state
    from pettriage.models.serving.factory import client_name, get_client
    from pettriage.models.tasks import Task

    client = get_client()
    if client is None:
        raise RuntimeError("LLM 클라이언트가 없다 — OPENAI_API_KEY 확인할 것")
    teacher = client_name()

    import random

    pool = _load_candidate_questions()
    random.Random(42).shuffle(pool)
    candidates = pool[: args.limit]
    print(f"후보 질문 {len(candidates)}건 (①②전체 재사용, 총 {len(pool)}건 중) — 검색·압축 시작")

    rows_out = []
    too_short = 0
    empty_hits = 0
    for i, c in enumerate(candidates, start=1):
        state = initial_state(question=c["input"], session_id=f"gen-{i}")
        if c.get("species"):
            state["slots"] = {"species": c["species"]}
        state.update(build_filter(state))

        hits = (retrieve(state) or {}).get("hits") or []
        if not hits:
            empty_hits += 1
            continue

        texts = []
        for h in hits:
            chunk = getattr(h, "chunk", None) or getattr(h, "text", None)
            text = getattr(chunk, "text", str(chunk)) if chunk else ""
            if text:
                texts.append(text)
        raw = "\n\n".join(texts)

        if len(raw) < _COMPRESS_LEN_THRESHOLD:
            too_short += 1
            continue  # 실제로도 압축을 안 타는 케이스 — 학습에 안 넣는다

        # graph.nodes.generate.compress_context 와 같은 값을 쓴다 (400→700,
        # 2026-08-03: 400에서 응답이 문장 중간에 잘리는 사례가 다수 발견됨).
        target = client.run(Task.COMPRESS, raw, max_tokens=700).strip()
        hallucinated = bool(_numbers_in(target) - _numbers_in(raw))

        rows_out.append(
            {
                "question": c["input"],
                "species": c.get("species"),
                "raw_context": raw,
                "target": target,
                "hallucinated_numbers": hallucinated,
                "teacher": teacher,
            }
        )
        if i % 50 == 0:
            print(f"  {i}/{len(candidates)} · 생성 {len(rows_out)}건")

    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w", encoding="utf-8") as f:
        for r in rows_out:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    print(f"\n압축 샘플 {len(rows_out)}건 → {args.out}")
    print(f"검색 결과 없음: {empty_hits} · 압축 불필요(800자 미만): {too_short}")
    print(f"수치 환각 의심: {sum(r['hallucinated_numbers'] for r in rows_out)}건")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
