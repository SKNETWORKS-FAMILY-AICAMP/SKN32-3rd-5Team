"""골든셋 → 엔진 → 채점 → 리포트.

설계 근거: docs/04_테스트-평가계획.md §1.2 · §4 · §8 · docs/02 §12 (QAEngine)

사용법
    python eval/harness/run_eval.py                    # 설정의 엔진 (기본 stub)
    python eval/harness/run_eval.py --engine stub
    python eval/harness/run_eval.py --only G-028 G-029
    python eval/harness/run_eval.py --json eval/reports/run.json

엔진을 어떻게 잡는가
    `QAEngine` 프로토콜(`ask(req, session) -> AskResponse`)에만 의존한다 (D-40).
    WS2가 `GraphEngine` 을 완성하면 `--engine graph` 한 마디로 갈아끼운다.
    **이 파일은 그때 손대지 않는다.**

⚠️ 지금 기본 엔진은 `stub` 이다
    `StubEngine` 은 물질 3종(초콜릿·포도·아보카도)만 아는 고정 지식 엔진이다.
    따라서 이 하네스를 지금 돌리면 대부분이 `근거없음` 으로 거절된다.
    **그 숫자는 시스템 성능이 아니라 베이스라인**이며, 리포트 머리에 그렇게 찍는다.
    엔진 이름을 리포트에 박는 이유가 이것이다 — 나중에 숫자만 떼어 인용하면
    "초록불이 곧 거짓 근거" 가 된다 (04 §2.5.6).
"""

from __future__ import annotations

import argparse
import contextlib
import csv
import json
import re
import sys
import time
from collections.abc import Iterable, Sequence
from dataclasses import asdict
from functools import lru_cache
from pathlib import Path

_HERE = Path(__file__).resolve().parent
ROOT = _HERE.parents[1]
for _p in (str(_HERE), str(ROOT / "src")):  # 설치 없이도, 어디서 불러도 돈다
    if _p not in sys.path:
        sys.path.insert(0, _p)

from metrics import (  # noqa: E402  (sys.path 조작 뒤에 와야 한다)
    CaseResult,
    Summary,
    fmt,
    fmt_ms,
    group_by,
    score_case,
    summarize,
)

GOLDEN_DIR = ROOT / "eval" / "goldenset"
REPORT_DIR = ROOT / "eval" / "reports"

#: 골든셋 `species` → `AskRequest.species`. 비면 종 미확인(되묻기 기대)이다.
SPECIES_OK = {"dog", "cat", "bird"}


# ─────────────────────────────────────────────────────────────
# 입력
# ─────────────────────────────────────────────────────────────
def load_goldenset(paths: Sequence[Path]) -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    seen: set[str] = set()
    for p in paths:
        with p.open(encoding="utf-8-sig", newline="") as f:
            for row in csv.DictReader(f):
                cid = (row.get("case_id") or "").strip()
                if not cid:
                    continue
                if cid in seen:
                    raise SystemExit(f"case_id 중복: {cid} ({p.name}). 구간이 겹쳤다 (04a §7).")
                seen.add(cid)
                rows.append({k: (v or "").strip() for k, v in row.items()})
    if not rows:
        raise SystemExit(f"골든셋이 비었다. {GOLDEN_DIR} 에 golden_*.csv 를 둘 것.")
    return rows


def build_request(row: dict[str, str]):
    """골든셋 행 → `AskRequest`.

    **체중·섭취량을 여기서 채우지 않는다.** 골든셋에 그 칸이 없고,
    질문 문장 안에 자연어로 들어 있다 — 그것을 뽑는 것이 ② 슬롯 추출 노드의 일이다.
    하네스가 대신 파싱해 넣으면 **슬롯 추출을 채점에서 빼는 셈**이 되고,
    `dose` 13건이 검증하려던 경로가 통째로 사라진다 (04 §2.2).
    """
    from pettriage.app.contracts import AskRequest

    sp = row.get("species") or None
    return AskRequest(question=row["question"], species=sp if sp in SPECIES_OK else None)


# ─────────────────────────────────────────────────────────────
# 실행
# ─────────────────────────────────────────────────────────────
def make_engine(kind: str | None):
    """엔진 1개를 만든다. 이름은 리포트에 그대로 박힌다."""
    from pettriage.config import get_config

    kind = kind or get_config().serve.engine
    if kind == "stub":
        from pettriage.app.engine import StubEngine

        return StubEngine()
    if kind == "graph":
        from pettriage.graph.engine import GraphEngine  # type: ignore[attr-defined]

        return GraphEngine()
    raise SystemExit(f"알 수 없는 엔진: {kind!r} (stub | graph)")


def _disclaimer_pattern() -> re.Pattern[str]:
    """고지 문구를 **공백 차이에 둔감한** 정규식으로 만든다.

    문구 자체는 `contracts.DISCLAIMER` 한 곳에서 온다 — 여기에 다시 적지 않는다.
    """
    from pettriage.app.contracts import DISCLAIMER

    # 마지막 마침표는 있어도 없어도 걸리게 한다 (엔진이 붙여 쓰는 경우가 있다)
    body = re.escape(DISCLAIMER.rstrip("."))
    body = re.sub(r"(\\ )+", r"\\s+", body)  # 이스케이프된 공백 → 임의 공백
    return re.compile(body + r"\.?", re.MULTILINE)


@lru_cache(maxsize=1)
def _disclaimer_re() -> re.Pattern[str]:
    return _disclaimer_pattern()


def scored_text(resp) -> str:
    """채점 대상 문장. **고지 문구는 뺀다.**

    처음엔 `full_text` 를 그대로 넣었다가 채점이 통째로 망가졌다.

        DISCLAIMER = "본 안내는 참고용이며 수의학적 **진단**이 아닙니다.
                      이상이 의심되면 **수의사**와 상담하세요."

    이 문장이 모든 응답에 무조건 붙는다 (02 §9). 그래서

      · `must_contain: 수의사`      → 거절 응답도 **거저 통과**한다
      · `must_not_contain: 진단`    → 어떤 응답도 **통과 불가**하다 (G-004가 실제로 걸렸다)

    둘 다 채점기가 틀린 것이지 시스템이 틀린 게 아니다.
    **고정 상용구를 채점하면 지표가 상용구를 측정한다.**

    빼되 **상승 조건은 남긴다** — 조건 누락은 이 도메인에서 과소평가와 같다 (D-39).
    """
    parts: list[str] = []
    if resp.answer:
        parts.append(resp.answer)
    elif resp.clarify:
        parts.append(resp.clarify.question)
    elif resp.refusal:
        parts.append(f"{resp.refusal.message} {resp.refusal.advice}")
    if resp.triage and resp.triage.escalation_conditions:
        parts.append(", ".join(resp.triage.escalation_conditions))
    text = " ".join(parts)
    # 엔진이 본문에 고지를 한 번 더 넣었더라도 채점에서는 지운다.
    #
    # ⚠️ **정확 일치로 지우면 새어 나간다.** `text.replace(DISCLAIMER, " ")` 만 쓰던 때는
    # 줄바꿈 하나, 공백 하나만 달라도 고지가 그대로 남아 `must_contain: 수의사` 가
    # 거저 통과하고 `must_not_contain: 진단` 이 영원히 실패했다 (G-004 실제 사례).
    # 공백을 정규화한 패턴으로 지운다 — 문구가 조금 흐트러져도 걸린다.
    return _disclaimer_re().sub(" ", text)


def node_timings(resp) -> dict[str, float]:
    """엔진이 노드별 시각을 실어 보내면 꺼낸다. **없으면 빈 dict — 지어내지 않는다.**

    `AskResponse` 계약에는 없는 선택 필드다. `GraphEngine` 이 붙을 때
    `resp.timings` 로 실어 보내면 여기서 자동으로 잡힌다.
    **없다고 전체 지연을 노드에 배분하지 않는다** — 그건 측정이 아니라 추정이다.
    """
    t = getattr(resp, "timings", None)
    if not isinstance(t, dict):
        return {}
    # `(int, float)` 튜플이 아니라 `int | float` 를 쓴다. `store.py` 와 같은 표기이고,
    # 핀된 ruff(<0.9)의 UP038 이 튜플 형태를 잡는다. 동작은 같다.
    return {str(k): float(v) for k, v in t.items() if isinstance(v, int | float)}


def warm_up(engine, rows: Sequence[dict[str, str]]) -> None:
    """**측정 전에 한 번 버린다** (D-53).

    첫 호출에는 임베딩 모델 로딩이 섞인다. 그대로 재면

      · `--only` 로 1건만 돌릴 때 **그 한 건이 로딩 시간으로 찍힌다**
      · `--fail-over` 게이트가 **로딩 때문에 실패**한다

    측정 도구가 자기 측정을 오염시키는 셈이다. 그래서 첫 건을 한 번 태우고 버린다.

    **결과를 쓰지 않는다.** 채점에도, 지연 집계에도 안 들어간다.
    실패해도 조용히 넘어간다 — 워밍업이 안 되는 것과 평가가 안 되는 것은 다르고,
    진짜 실패라면 본 측정에서 같은 예외로 다시 잡힌다.
    """
    if not rows:
        return
    from pettriage.app.session import SessionStore

    # 워밍업 실패는 평가 실패가 아니다. 진짜 문제라면 본 측정에서 같은 예외로 다시 잡힌다.
    with contextlib.suppress(Exception):
        engine.ask(build_request(rows[0]), SessionStore().get_or_create(None))


def run(rows: Iterable[dict[str, str]], engine) -> list[CaseResult]:
    """각 건을 **새 세션**으로 한 번씩 태운다.

    세션을 공유하면 앞 건의 슬롯(체중·종)이 뒤 건에 새어 들어가
    되묻기가 안 나온다 — `slot` 9건이 통째로 오염된다.

    **첫 응답만 채점한다.** 되묻기에 답을 대신 채워 주면 `clarify` 를 기대한
    케이스가 `answered` 로 바뀌고, 무엇을 되물었는지도 못 본다.
    """
    from pettriage.app.session import SessionStore

    store = SessionStore()
    results: list[CaseResult] = []
    for row in rows:
        # perf_counter 를 쓴다 — time.time() 은 시스템 시계 조정에 흔들린다.
        t0 = time.perf_counter()
        try:
            resp = engine.ask(build_request(row), store.get_or_create(None))
            elapsed = (time.perf_counter() - t0) * 1000
            results.append(
                score_case(
                    row,
                    status=resp.status,
                    level=resp.triage.level if resp.triage else None,
                    answer_text=scored_text(resp),  # 상승 조건 포함 · 고지 문구 제외
                    citations=[c.source_id for c in resp.citations],
                    latency_ms=elapsed,
                    node_ms=node_timings(resp),
                )
            )
        except Exception as e:  # 계약 위반(ValidationError)도 여기 잡힌다 — 결과다
            results.append(
                score_case(
                    row,
                    status=None,
                    level=None,
                    answer_text="",
                    citations=[],
                    latency_ms=(time.perf_counter() - t0) * 1000,
                    error=f"{type(e).__name__}: {e}",
                )
            )
    return results


# ─────────────────────────────────────────────────────────────
# 출력
# ─────────────────────────────────────────────────────────────
def _table(title: str, groups: dict[str, Summary]) -> list[str]:
    out = [
        f"\n■ {title}",
        f"  {'':12} {'n':>4} {'통과':>7} {'등급일치':>8} {'과소':>7} {'과대':>7}",
    ]
    for k, s in groups.items():
        out.append(
            f"  {k:12} {s.n:>4} {fmt(s.pass_rate):>7} {fmt(s.level_accuracy):>8} "
            f"{fmt(s.under_rate):>7} {fmt(s.over_rate):>7}"
        )
    return out


def report(results: list[CaseResult], *, engine_name: str) -> str:
    s = summarize(results)
    L: list[str] = []
    L.append("=" * 66)
    L.append(f"  평가 하네스 — 엔진 `{engine_name}` · 골든셋 {s.n}건")
    L.append("=" * 66)
    if engine_name == "stub":
        L.append("  ⚠️ StubEngine 은 물질 3종만 아는 고정 지식 엔진이다.")
        L.append("     아래 수치는 시스템 성능이 아니라 **베이스라인**이다 (04 §5 구성 A).")

    L.append("\n■ 전체")
    L.append(f"  통과            {fmt(s.pass_rate):>7}   ({s.passed}/{s.n})")
    L.append(f"  상태 일치       {fmt(s.status_accuracy):>7}   ({s.status_correct}/{s.n})")
    if s.errors:
        L.append(f"  ⚠️ 예외          {s.errors}건 — 계약 위반이면 응답 조립이 막힌 것이다")

    L.append(f"\n■ 트리아지 (분모 {s.level_n} — 양쪽 다 등급이 있는 건만)")
    L.append(f"  등급 일치도     {fmt(s.level_accuracy):>7}")
    L.append(f"  인접 허용       {fmt(s.adjacent_accuracy):>7}")
    L.append(f"  🔴 과소평가율    {fmt(s.under_rate):>7}   ({s.under}) ← 최우선 지표")
    L.append(f"  과대평가율      {fmt(s.over_rate):>7}   ({s.over}) ← 의도된 편향 (04 §4.1.x)")
    L.append(f"  🔴 중대 과소     {fmt(s.critical_under_rate):>7}   ({s.critical_under}) ← 목표 0")
    L.append(
        f"\n  ▸ 등급을 못 낸 긴급 건  {fmt(s.missed_urgent_rate):>7}   "
        f"({s.missed_urgent}/{s.urgent_n})"
    )
    L.append("    정답이 CALL_NOW 이상인데 거절·되묻기로 빠진 건이다.")
    L.append("    등급 오류가 아니라 **분모가 다르다** — 과소평가율에 섞지 않는다 (04 §1.2).")

    L.append("\n■ 근거·문구")
    L.append(f"  must_cite 적중(any)  {fmt(s.cite_any_rate):>7}   ({s.cite_any}/{s.cite_n})")
    L.append(f"  must_cite 적중(all)  {fmt(s.cite_all_rate):>7}   ({s.cite_all}/{s.cite_n})")
    L.append(f"  must_contain (any)   {fmt(s.contain_rate):>7}   ({s.contain_ok}/{s.contain_n})")
    L.append(
        f"  must_contain (all)   {fmt(s.contain_all_rate):>7}   ({s.contain_all}/{s.contain_n})"
    )
    L.append(
        f"  must_not_contain     {fmt(s.not_contain_rate):>7}   "
        f"({s.not_contain_ok}/{s.not_contain_n})   ← answered 만"
    )
    L.append(
        f"    전체 기준          {fmt(s.not_contain_rate_all):>7}   "
        f"({s.not_contain_all_ok}/{s.not_contain_all_n})"
    )
    L.append("    거절·되묻기는 금지 문구를 쓸 기회가 없어 **거저 통과**한다.")
    L.append("    분모를 나누지 않으면 '답을 안 했다' 가 만점으로 보고된다 (04 §1.2).")

    L.append("\n■ 지연 (02 §12.4 로 스트리밍을 안 쓰므로 이 값이 그대로 침묵이 된다)")
    L.append(
        f"  전체        p50 {fmt_ms(s.p50_ms):>8}   p95 {fmt_ms(s.p95_ms):>8}"
        f"   (n={len(s.latencies)})"
    )
    L.append(
        f"  answered    p50 {fmt_ms(s.answered_p50_ms):>8}   p95 {fmt_ms(s.answered_p95_ms):>8}"
        f"   (n={len(s.answered_latencies)}) ← 실제 체감"
    )
    L.append("    되묻기·거절은 LLM 을 2번만 돌아 빠르다. 섞으면 평균이 낙관적이다.")
    L.append("    ※ 측정 전 워밍업 1회를 버렸다 — 모델 로딩은 이 숫자에 없다.")
    L.append("      콜드 스타트를 재려면 `--no-warmup`.")
    nodes = s.node_p95()
    if nodes:
        L.append("\n  노드별 p95 (느린 순)")
        for name, ms in nodes[:10]:
            L.append(f"    {name:18} {fmt_ms(ms):>8}")
    else:
        L.append("    ▸ 노드별 분해 없음 — 엔진이 `resp.timings` 를 실어 보내면 자동으로 잡힌다.")

    L += _table("종별 (04 §4.2 — 전체 평균은 조류 저하를 가린다)", group_by(results, "species"))
    L += _table("유형별 (case_type · 04 §2.2)", group_by(results, "case_type"))

    if s.status_confusion:
        L.append("\n■ 상태 혼동 (정답 → 예측)")
        for (exp, act), n in sorted(s.status_confusion.items(), key=lambda kv: -kv[1]):
            mark = "  " if exp == act else "✗ "
            L.append(f"  {mark}{exp or '?':9} → {act or '(예외)':9} {n:>3}")

    fails = [r for r in results if not r.passed]
    if fails:
        L.append(f"\n■ 실패 {len(fails)}건 (04 §7 실패 분석 입력)")
        for r in fails[:40]:
            why = r.error or (
                f"상태 {r.expected_status}→{r.actual_status}"
                if not r.status_ok
                else f"등급 {r.expected_level}→{r.actual_level}"
                if r.expected_level != r.actual_level
                else "근거/문구"
            )
            flag = (
                " 🔴중대과소" if r.critical_under else (" ▸긴급미판정" if r.missed_urgent else "")
            )
            L.append(f"  {r.case_id:8} {r.case_type:8} {r.species:9} {why}{flag}")
        if len(fails) > 40:
            L.append(f"  … 외 {len(fails) - 40}건 (전체는 --json)")

    L.append("")
    return "\n".join(L)


def main(argv: Sequence[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="골든셋 평가 하네스 (04 §4)")
    ap.add_argument("--engine", choices=["stub", "graph"], help="기본값은 configs 의 serve.engine")
    ap.add_argument(
        "--goldenset", nargs="*", type=Path, help="기본값은 eval/goldenset/golden_*.csv"
    )
    ap.add_argument("--only", nargs="*", metavar="CASE_ID", help="특정 케이스만")
    ap.add_argument("--json", type=Path, help="건별 결과를 JSON 으로 기록")
    ap.add_argument(
        "--fail-under",
        type=float,
        metavar="RATE",
        help="과소평가율이 이 값을 넘으면 종료코드 1 (CI 게이트)",
    )
    ap.add_argument(
        "--no-warmup",
        action="store_true",
        help="워밍업을 건너뛴다. **콜드 스타트 지연을 재고 싶을 때만** 쓴다",
    )
    ap.add_argument(
        "--fail-over",
        type=float,
        metavar="MS",
        help="answered p95 지연이 이 값(ms)을 넘으면 종료코드 1 (CI 게이트)",
    )
    ap.add_argument(
        "--min-graded",
        type=int,
        default=10,
        metavar="N",
        help=(
            "과소평가율 게이트가 요구하는 **최소 분모**. 분모가 작으면 비율이 무의미하다 "
            "(기본 10). `--fail-under` 와 함께만 쓰인다"
        ),
    )
    ap.add_argument(
        "--fail-missed",
        type=float,
        default=0.30,
        metavar="RATE",
        help=(
            "정답이 CALL_NOW 이상인데 **등급을 아예 못 낸** 비율의 상한 (기본 0.30). "
            "이 값은 과소평가율 분모 밖이라 별도 게이트가 필요하다 (04 §1.2)"
        ),
    )
    a = ap.parse_args(argv)

    paths = a.goldenset or sorted(GOLDEN_DIR.glob("golden_*.csv"))
    rows = load_goldenset(paths)
    if a.only:
        keep = set(a.only)
        rows = [r for r in rows if r["case_id"] in keep]
        if not rows:
            raise SystemExit(f"해당 case_id 가 없다: {sorted(keep)}")

    engine = make_engine(a.engine)
    if not a.no_warmup:
        warm_up(engine, rows)  # 첫 건에 모델 로딩이 섞이지 않게 (D-53)
    results = run(rows, engine)
    print(report(results, engine_name=engine.name))

    if a.json:
        a.json.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "engine": engine.name,
            "goldenset": [p.name for p in paths],
            "n": len(results),
            "latency": {
                "p50_ms": summarize(results).p50_ms,
                "p95_ms": summarize(results).p95_ms,
                "answered_p50_ms": summarize(results).answered_p50_ms,
                "answered_p95_ms": summarize(results).answered_p95_ms,
                "node_p95_ms": dict(summarize(results).node_p95()),
            },
            "cases": [
                asdict(r) | {"passed": r.passed, "level_delta": r.level_delta} for r in results
            ],
        }
        a.json.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"→ {a.json}")

    if a.fail_under is not None:
        s = summarize(results)
        # 측정 대상이 **너무 적으면** 통과가 아니다.
        #
        # 예전에는 `level_n == 0` 만 막았다. 그런데 분모가 1이어도 통과했다 —
        # 52건 중 등급을 낸 건이 1건뿐이고 그 1건이 맞으면 과소평가율 0.0% 가 되어,
        # `통과 21.2%` · `등급 못 낸 긴급 건 96%` 인 실행이 **최엄격 게이트를 초록으로
        # 통과했다** (2026-08-02 재현). 04 §2.5.6 이 겪은 "초록불이 곧 거짓 근거" 그대로다.
        if s.level_n < a.min_graded:
            print(
                f"✗ 등급을 낸 건이 {s.level_n}건뿐이다 (최소 {a.min_graded}) — "
                "과소평가율을 신뢰할 수 없다. 통과로 치지 않는다."
            )
            return 1
        if s.under_rate is not None and s.under_rate > a.fail_under:
            print(f"✗ 과소평가율 {fmt(s.under_rate)} > 상한 {fmt(a.fail_under)}")
            return 1
        # **등급을 아예 못 낸 긴급 건**은 과소평가율의 분모에 없다 (04 §1.2).
        # 그래서 별도 게이트가 필요하다 — 없으면 "전부 거절" 이 0.0% 로 통과한다.
        if s.missed_urgent_rate is not None and s.missed_urgent_rate > a.fail_missed:
            print(
                f"✗ 등급을 못 낸 긴급 건 {fmt(s.missed_urgent_rate)} > "
                f"상한 {fmt(a.fail_missed)}"
            )
            return 1

    if a.fail_over is not None:
        s = summarize(results)
        # 과소평가율 게이트와 같은 규칙 — **측정 0건은 통과가 아니다.**
        # 전부 거절되면 answered 가 없고, 그 상태로 초록을 주면
        # "빠른 게 아니라 답을 안 한 것" 이 통과로 읽힌다.
        if not s.answered_latencies:
            print("✗ answered 응답이 0건이다 — 지연을 측정하지 못했다. 통과로 치지 않는다.")
            return 1
        p95 = s.answered_p95_ms
        if p95 is not None and p95 > a.fail_over:
            print(f"✗ answered p95 {fmt_ms(p95)} > 상한 {fmt_ms(a.fail_over)}")
            return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
