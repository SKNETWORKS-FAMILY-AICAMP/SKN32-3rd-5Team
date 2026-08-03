#!/usr/bin/env python3
"""④ 근거 검증이 **무엇을 걸렀는지 눈으로 본다** (현재상태 §5 ①).

    python scripts/probe_grounding.py --arm A --type dose -n 5
    python scripts/probe_grounding.py --arm A --only G-001,G-047 --json eval/reports/근거표본.json

⚠️ **`--arm` 을 반드시 준다.** 안 주면 셸에 남은 `PETTRIAGE__MODEL__*` 를 그대로
   타고, 그렇게 뜬 표본은 46.7% 를 잰 것과 **같은 조건이라고 말할 수 없다**
   (`run_eval.py` 와 같은 이유 — `arms.py` 머리말).

## 왜 필요한가

    2026-08-03 60건 측정 — 검증된 155문장 중 **54문장(35%)이 `근거없음`** 판정을
    받았고, **그대로 답변에 실려 나갔다.** `verify.py` 머리말은
    *"판정에 따른 조치는 코드가 한다 — 문장 제거·재검색 1회·거절"* 이라고 적었는데
    **문장 제거가 구현돼 있지 않다.**

    그런데 그 54문장을 **아무도 본 적이 없다.** 리포트 JSON 에는 개수만 있다.
    그래서 다음 두 갈래를 고를 수 없다 —

      · **오탐이면** 2-gram 임계값(0.3)을 손본다. 문장 제거는 필요 없다
      · **진짜면** 문장 제거를 구현한다. ⚠️ 35%를 지우면 답변이 비어
        계약이 거부할 수 있다 — 그때는 설계 판단이 한 번 더 필요하다

    **표본을 보기 전에는 어느 쪽도 고를 수 없다.** 이 스크립트는 그 표본만 낸다.

## 무엇을 하지 않는가

    **고치지 않는다.** 임계값도 안 건드리고 문장도 안 지운다.
    채점도 안 한다 — 통과/실패는 `run_eval.py` 의 일이다.

## 🔴 연락처를 가린다

    `verify` 는 그래프에서 **`simplify`·`answered` 앞**에 있다.

        generate → judge → decide → verify → simplify → answered(연락처 제거)

    즉 판정 대상 문장은 **D-47 필터를 통과하기 전의 초안**이고, 지워질 전화번호가
    그 안에 그대로 있다. 화면에 찍어 붙여넣는 순간 필터가 무의미해진다.
    그래서 `has_contact()` 인 문장은 **내용을 안 찍고 표시만 남긴다.**
    (`removed_contacts` 를 목록으로 돌려주려다 같은 함정에 걸린 적이 있다 — D-47.)

## 읽는 법

    각 문장 옆의 `일치율` 이 `_judge_sentence` 가 쓰는 2-gram 문자 일치율이고,
    **0.3 미만이면 `근거없음`** 이다. 임계값이 문제인지 문장이 문제인지는
    이 숫자와 문장을 나란히 놓고 사람이 판단한다 —

      · 근거에 **있는 말인데** 일치율이 0.25 쯤이면 → **임계값이 빡빡하다**
      · 근거에 **없는 말이고** 일치율도 낮으면 → **검증기가 맞다. 지워야 한다**
      · 근거에 **없는 말인데** 일치율이 높으면 → 🔴 **2-gram 이 통과시키고 있다.**
        가장 나쁜 경우다 — 환각이 검증을 뚫는다
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "eval" / "harness"))

GOLDEN_DIR = ROOT / "eval" / "goldenset"


def _ratio(sentence: str, context: str) -> float:
    """`_judge_sentence` 가 쓰는 값을 **그대로** 다시 계산한다.

    ⚠️ 판정 로직을 여기 베껴 적지 않는다 — 임계값을 옮기면 두 곳이 어긋난다.
       `verify` 의 내부 함수를 그대로 부른다 (D-22).
    """
    from pettriage.graph.nodes.verify import _char_ngrams

    if not context or not sentence:
        return 0.0
    grams = _char_ngrams(sentence)
    if not grams:
        return 0.0
    return sum(1 for g in grams if g in context) / len(grams)


def _safe(sentence: str) -> tuple[str, bool]:
    """연락처가 든 문장은 **내용을 지운다.** (문장, 가렸나)"""
    from pettriage.safety.contacts import has_contact

    if has_contact(sentence):
        return ("〔연락처가 든 문장 — 가림 (D-47)〕", True)
    return (sentence, False)


def probe(rows: list[dict[str, str]]) -> list[dict]:
    """각 건을 태우고 `verdicts` 를 그대로 꺼낸다.

    ⚠️ **엔진 내부(`_build_state`·`_run_pipeline`)를 부른다.** `ask()` 는
       `AskResponse` 로 조립하면서 `verdicts` 를 떨어뜨리기 때문이다.
       진단 도구라서 감수한다 — **제품 경로가 아니다.**

    🔴 **내부를 부르면 `ask()` 가 그 앞에서 하는 일도 같이 옮겨야 한다.**

           progressed = session.merge(req)      ← 이것을 빼먹었다
           state = self._build_state(req, session)

       `_build_state` 는 `req` 가 아니라 **`session`** 에서 슬롯을 읽는다
       (`species`·`weight_kg`·`amount_g`). `merge` 가 없으면 골든셋 행의
       `species` 가 한 번도 전달되지 않아 **전 건이 종을 되묻고 끝난다** —
       `verify` 를 아무도 타지 않으므로 표본이 0개다. 2026-08-03 실측:
       `--type dose` 5건이 전부 `clarify` 로 나왔고, 유형 탓으로 오해했다.
    """
    from run_eval import build_request  # type: ignore[import-not-found]

    from pettriage.app.session import SessionStore
    from pettriage.graph.engine import GraphEngine
    from pettriage.graph.nodes.verify import _GROUND_THRESHOLD

    engine = GraphEngine()
    store = SessionStore()
    out: list[dict] = []

    for row in rows:
        cid = row.get("case_id", "?")
        # 세션을 공유하지 않는다 — 앞 건의 종·체중이 새면 경로가 달라진다.
        session = store.get_or_create(None)
        req = build_request(row)
        # 🔴 `ask()` 와 **같은 순서**로 부른다 (위 머리말). 요청의 슬롯을 세션에
        #    옮기는 것이 `merge` 이고, `_build_state` 는 세션만 읽는다.
        session.merge(req)
        try:
            state = engine._run_pipeline(engine._build_state(req, session))
        except Exception as e:  # noqa: BLE001
            out.append({"case_id": cid, "error": f"{type(e).__name__}: {e}"})
            print(f"  ✗ {cid:8} {type(e).__name__}: {e}")
            continue

        context = state.get("context", "") or ""
        verdicts = state.get("verdicts") or []
        sentences = []
        for v in verdicts:
            text, redacted = _safe(v.get("sentence", ""))
            sentences.append(
                {
                    "verdict": v.get("verdict", ""),
                    "ratio": round(_ratio(v.get("sentence", ""), context), 3),
                    "sentence": text,
                    "redacted": redacted,
                }
            )

        rec = {
            "case_id": cid,
            "case_type": row.get("case_type", ""),
            "species": row.get("species", ""),
            "question": row.get("question", ""),
            "status": state.get("status", ""),
            "context_chars": len(context),
            "checked": len(sentences),
            "unsupported": sum(1 for s in sentences if s["verdict"] == "근거없음"),
            "sentences": sentences,
        }
        out.append(rec)

        # ── 화면 출력 ─────────────────────────────────────────
        print(f"\n{'─' * 78}")
        print(f"  {cid}  [{rec['case_type']}·{rec['species']}]  status={rec['status']}")
        print(f"  질문: {rec['question']}")
        if not sentences:
            # **0건과 "안 돌았다"는 다르다.** 되묻기·거절이면 verify 를 안 탄다.
            print("  ⚠️  검증이 돌지 않았다 (되묻기·거절 경로) — 표본 없음")
            continue
        print(
            f"  근거 {rec['context_chars']}자 · 문장 {rec['checked']}개"
            f" · 근거없음 {rec['unsupported']}개"
        )
        print()
        marks = {"근거없음": "🔴", "모순": "⚠️ "}
        for i, s in enumerate(sentences, 1):
            mark = marks.get(s["verdict"], "  ")
            near = "  ← 임계값 부근" if abs(s["ratio"] - _GROUND_THRESHOLD) <= 0.05 else ""
            print(f"  {mark} {i:2}. 일치율 {s['ratio']:.3f}  {s['verdict']}{near}")
            print(f"        {s['sentence']}")
    return out


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--arm",
        choices=["none", "A", "A-LC", "C", "D"],
        help=(
            "04 §3 비교군을 이름으로 고른다 (models/serving/arms.py). "
            "**주지 않으면 셸에 남은 `PETTRIAGE__MODEL__*` 를 그대로 탄다** — "
            "그 표본은 어떤 조건에서 나온 것인지 말할 수 없다"
        ),
    )
    ap.add_argument("--type", help="case_type 으로 거른다 (dose·symptom·prevent·none …)")
    ap.add_argument("--species", help="종으로 거른다 (dog·cat·bird)")
    ap.add_argument("--only", help="case_id 를 쉼표로 (예: G-001,G-047)")
    ap.add_argument("-n", type=int, default=5, help="최대 건수 (기본 5)")
    ap.add_argument("--json", type=Path, help="표본을 파일로 저장")
    a = ap.parse_args(argv)

    # ⚠️ **설정을 읽기 전에** 세운다 — `get_config` 는 `lru_cache` 라 한 번 읽히면
    #    굳는다. `run_eval.py` 가 인자 파싱 직후에 이것을 부르는 것과 같은 이유다.
    if a.arm:
        from pettriage.models.serving.arms import apply_arm

        print(f"비교군 {a.arm} — {apply_arm(a.arm)}")
    else:
        print("⚠️  --arm 이 없다 — 현재 셸·.env 의 설정을 그대로 탄다 (조건을 못 적는다)")

    from run_eval import load_goldenset  # type: ignore[import-not-found]

    rows = load_goldenset(sorted(GOLDEN_DIR.glob("golden_*.csv")))
    if a.only:
        want = {s.strip() for s in a.only.split(",") if s.strip()}
        rows = [r for r in rows if r.get("case_id") in want]
    if a.type:
        rows = [r for r in rows if r.get("case_type") == a.type]
    if a.species:
        rows = [r for r in rows if r.get("species") == a.species]
    rows = rows[: a.n]

    if not rows:
        print("해당하는 케이스가 없다. --type/--species/--only 를 확인한다.")
        return 1

    print("=" * 78)
    print(f"  ④ 근거 검증 표본 — {len(rows)}건")
    print("  🔴 근거없음   ⚠️  모순   일치율 0.3 미만이 근거없음이다")
    print("=" * 78)

    records = probe(rows)

    checked = sum(r.get("checked", 0) for r in records)
    unsup = sum(r.get("unsupported", 0) for r in records)
    ran = sum(1 for r in records if r.get("checked"))
    print(f"\n{'=' * 78}")
    print(f"  검증이 돈 건 {ran}/{len(records)} · 문장 {checked}개 · 근거없음 {unsup}개", end="")
    print(f" ({unsup / checked:.0%})" if checked else "")
    print("=" * 78)
    if not checked:
        # ⚠️ **유형 탓으로 먼저 짚지 않는다.** 2026-08-03 에 그렇게 안내했다가
        #    `--type dose` 를 다시 돌리게 만들었는데, 진짜 원인은 `session.merge`
        #    누락이었다 — 유형과 무관하게 전 건이 되물었다. 흔한 순서로 적는다.
        print("\n⚠️  한 건도 검증을 타지 않았다 — 되묻기·거절만 나왔다.")
        print("    위 status 가 **전부** clarify 면 유형 문제가 아니다. 순서대로 본다 —")
        print("      1. 종·체중이 전달됐나 (전 건 clarify 면 슬롯이 안 들어간 것이다)")
        print("      2. python scripts/doctor.py   — 키·인덱스가 있나")
        print("      3. --arm A 를 줬나            — 없으면 LLM 없이 돈 것일 수 있다")
        print("    일부만 clarify 면 그때가 유형 문제다. `--type dose` 로 좁힌다.")

    if a.json:
        a.json.parent.mkdir(parents=True, exist_ok=True)
        a.json.write_text(json.dumps(records, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"\n저장 — {a.json}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
