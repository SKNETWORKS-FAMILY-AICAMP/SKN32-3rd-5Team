"""골든셋 → 엔진 → 채점 → 리포트.

설계 근거: docs/04_테스트-평가계획.md §1.2 · §4 · §8 · docs/02 §12 (QAEngine)

사용법
    python eval/harness/run_eval.py                    # 설정의 엔진 (기본 stub)
    python eval/harness/run_eval.py --engine stub
    python eval/harness/run_eval.py --only G-028 G-029
    python eval/harness/run_eval.py --json eval/reports/run.json

엔진을 어떻게 잡는가
    `QAEngine` 프로토콜(`ask(req, session) -> AskResponse`)에만 의존한다 (D-40).
    `--engine graph` 로 갈아끼운다. 기본값은 `configs/*.yaml` 의 `serve.engine`
    (`eval` 프로파일은 이미 `graph` 다).
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
import os
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
    NAME_TO_LEVEL,
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


def _provenance() -> dict[str, object]:
    """**이 숫자가 무엇으로 나온 것인가** (04 §8 · `eval/reports/README.md`).

    `eval/reports/README.md` 는 처음부터 리포트 머리말에 코퍼스·설정·의존성 해시를
    적으라고 요구했다. 그런데 하네스는 **엔진 이름만 남겼다** — 사람이 손으로 적게
    두면 안 적히고, 안 적힌 리포트는 나중에 해석할 수 없다.

    ⚠️ **`dirty` 가 참이면 그 결과는 재현할 수 없다.** 커밋되지 않은 변경 위에서
    잰 숫자이고, 그 상태를 남에게 줄 방법이 없다. 숨기지 않고 적는다 —
    *"측정하지 않은 것을 0 으로 적지 않는다"* 와 같은 이유다.
    """
    import subprocess

    def git(*args: str) -> str:
        try:
            # ⚠️ **`encoding` 을 반드시 준다.** 윈도우의 기본은 cp949 이고,
            # 한글 경로·커밋메시지가 섞이면 읽는 스레드가 `UnicodeDecodeError` 로
            # 죽는다 (2026-08-02 실측). 그러면 `dirty` 가 조용히 거짓이 되고,
            # **재현 불가인 결과가 재현 가능한 것처럼 기록된다** (04 §8).
            return subprocess.run(
                ["git", *args],
                capture_output=True,
                text=True,
                encoding="utf-8",
                errors="replace",
                timeout=5,
                check=False,
            ).stdout.strip()
        except Exception:  # noqa: BLE001 — git 이 없거나 저장소가 아니면 그냥 비운다
            return ""

    from pettriage.config import get_config

    cfg = get_config()
    head = git("rev-parse", "HEAD")
    out: dict[str, object] = {
        "repo_commit": head or "(git 없음)",
        # 커밋 안 된 변경 위에서 쟀는가. 참이면 **재현 불가**다.
        "dirty": bool(git("status", "--porcelain")),
        "profile": os.getenv("PETTRIAGE_PROFILE", "default"),
        "engine_configured": cfg.serve.engine,
        "model_provider": cfg.model.provider,
    }

    # ── 🔴 **쓰는 값만 적는다** ─────────────────────────────────
    #
    # 예전에는 provider 와 무관하게 `base_id`·`revision` 을 늘 적었다. 그래서
    # `--arm A`(gpt-4o-mini) 실행의 provenance 에 이렇게 남았다 —
    #
    #     model_provider: api
    #     model_base_id:  Qwen/Qwen3-4B                    ← 이 실행과 무관하다
    #     model_revision: 1cfa9a72...                      ← 게다가 무효한 핀이다
    #
    # 정작 그 실행을 정의하는 `api_model` 은 provenance 에 **없었다.**
    # 나중에 이 JSON 만 보면 **Qwen 으로 잰 것으로 읽는다.** 04 §8 이
    # *"무엇으로 잰 건지 모르는 숫자를 남기지 않는다"* 라고 한 것의 반대다.
    #
    # **안 쓰는 값을 쓰는 값처럼 적는 것은, 안 적는 것보다 나쁘다.**
    if cfg.model.provider in ("api", "langchain"):
        out["model_api_model"] = cfg.model.api_model
        out["model_api_base_url"] = cfg.model.api_base_url or "(OpenAI 본가)"
    elif cfg.model.provider == "qwen":
        out["model_base_id"] = cfg.model.base_id
        out["model_revision"] = cfg.model.revision
        out["model_dtype"] = cfg.model.dtype
        out["model_load_in_4bit"] = cfg.model.load_in_4bit
        # **C 와 D 를 가르는 유일한 값이다.** 없으면 어느 쪽을 잰 건지 모른다.
        out["model_adapter_path"] = cfg.model.adapter_path
    # provider == "none" 이면 모델 관련 값을 아무것도 적지 않는다 — 쓴 것이 없다.

    out |= {
        "embedding_model": cfg.retrieval.embedding_model,
        "top_k": cfg.retrieval.top_k,
        "score_threshold": cfg.retrieval.score_threshold,
        "max_clarify_turns": cfg.triage.max_clarify_turns,
    }
    return out


def make_engine(kind: str | None):
    """엔진 1개를 만든다. 이름은 리포트에 그대로 박힌다."""
    from pettriage.config import get_config

    kind = kind or get_config().serve.engine
    if kind == "stub":
        from pettriage.app.engine import StubEngine

        return StubEngine()
    if kind == "graph":
        from pettriage.app.safety_engine import SafetyEngine
        from pettriage.graph.engine import GraphEngine

        # ⚠️ **반드시 감싼다** (D-47 · D-54). 서비스 경로는 `deps.get_engine()` 이
        #    감싸는데 하네스는 엔진을 직접 만든다 — 안 감싸면 **평가만 다른 코드를 잰다.**
        #    코퍼스 응급 자료가 전부 미국 것이라 답변에 톨프리 번호가 실릴 수 있고,
        #    그러면 계약 `_no_foreign_contacts` 가 터져 그 케이스가 **오류로 집계된다.**
        #    실제로는 스크러빙되어 정상 응답이 나가는 건이다 — 지표가 틀어진다.
        return SafetyEngine(GraphEngine())
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
                    # **조건 없는 MONITOR 는 MONITOR 로 채점한다** (D-39 · 04 §4.1.0).
                    #
                    # 시스템은 실제로 `MONITOR` 로 판정했고, 상승 조건을 못 내서
                    # 출력을 막았을 뿐이다. 04 §4.1.0 은 *"조건 없는 관찰은 그 자체가
                    # 과소평가"* 라고 정했는데, 2026-08-03 까지는 이 건이 `triage=None`
                    # 인 거절로 들어와 **등급 분모에서 통째로 빠졌다.** 규칙이 채점에
                    # 한 줄도 반영돼 있지 않았고, 그 결과가 `과소평가율 0.0%` 였다.
                    level=(
                        NAME_TO_LEVEL["MONITOR"]
                        if getattr(resp, "monitor_without_conditions", False)
                        else (resp.triage.level if resp.triage else None)
                    ),
                    refusal_reason=resp.refusal.reason if resp.refusal else "",
                    answer_text=scored_text(resp),  # 상승 조건 포함 · 고지 문구 제외
                    citations=[c.source_id for c in resp.citations],
                    latency_ms=elapsed,
                    node_ms=node_timings(resp),
                    llm_fallbacks=resp.llm_fallbacks,
                    rule_level=resp.triage.rule_level if resp.triage else None,
                    llm_level=resp.triage.llm_level if resp.triage else None,
                    gate_overridden=bool(resp.triage and resp.triage.overridden),
                    gate_capped=bool(resp.triage and resp.triage.llm_capped),
                    grounding=getattr(resp, "grounding", None),
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


def report(results: list[CaseResult], *, engine_name: str, model_name: str = "?") -> str:
    s = summarize(results)
    L: list[str] = []
    L.append("=" * 66)
    L.append(f"  평가 하네스 — 엔진 `{engine_name}` · 골든셋 {s.n}건")
    # **무엇으로 잰 건지 모르는 숫자를 남기지 않는다** (04 §8).
    #   엔진 이름만 박혀 있어서 같은 `graph` 라도 gpt-4o-mini 로 잰 건지
    #   Qwen 베이스로 잰 건지 리포트에서 구분할 수 없었다 (D-65).
    L.append(f"  모델 `{model_name}`")
    L.append("=" * 66)
    if model_name.startswith("none"):
        L.append("  ⚠️ **모델 없이 돌았다** — 05 §4 의 5태스크가 전부 폴백이다.")
        L.append("     이 수치는 LLM 성능이 아니라 **코드·규칙만의 기준선**이다 (04 §3).")
        L.append("     특히 ④ 근거 검증은 draft=context 라 판정이 항상 `근거있음` 이다 —")
        L.append("     근거없음 탐지 재현율은 0인 채 초록으로 보인다. 그렇게 읽지 않는다.")
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

    # ── LLM 이 실제로 불렸나 ──────────────────────────────────────
    #
    # 🔴 **다른 모든 수치보다 먼저 읽어야 하는 칸이다.** 이것이 없으면 성적이 낮을 때
    #    *모델이 못한 것*인지 *모델이 안 불린 것*인지 구분이 안 된다.
    #    이서은 팀원이 잡은 D-73(라벨 누락으로 LLM 이 키워드 폴백보다 나빴다)이
    #    오래 안 보인 이유가 정확히 그 구분이 없었기 때문이다.
    # ── 게이트가 무엇을 했나 ────────────────────────────────────
    #
    # 산출물 ④ 가 요구하는 *"하향 금지 게이트가 실제로 작동했다"* 는 이 칸으로 보인다.
    # 그리고 등급 오류의 **원인**이 여기서 갈린다 — 규칙이 틀렸나, LLM 이 덮었나.
    if s.gate_n:
        L.append(f"\n■ 하향 금지 게이트 (D-09 · 분모 {s.gate_n} — 규칙·LLM 둘 다 등급을 낸 건)")
        L.append(f"  일치            {s.gate_agreed:>4}")
        L.append(
            f"  LLM 이 올렸다    {s.gate_raised:>4}   (그중 과대로 끝난 것 {s.gate_raised_wrong})"
        )
        L.append(f"  🔒 하향 차단     {s.gate_blocked:>4}   ← 게이트가 실제로 막은 횟수")
        if s.gate_capped:
            L.append(
                f"  🔒 상승 차단     {s.gate_capped:>4}   "
                f"← 잰 자리라 막았다 (D-80. 그중 여전히 어긋난 것 {s.gate_capped_wrong})"
            )
        if s.gate_raised_wrong:
            L.append("    ▸ 올림이 언제나 옳지는 않다. 규칙이 정량 계산으로 낸 등급을")
            L.append("      LLM 이 덮었다면 **근거 없는 상승**이다 (D-79 트레이드오프).")

    L.append("\n■ LLM 실행 (05 §6 — 폴백은 끄지 않고 표시한다)")
    # 🔴 예전에는 *"5태스크 전부 모델"* 이라고 적었다. **거짓이었다** —
    #    `Task.VERIFY` 는 호출부가 없어 부르지 않고, 부르지 않으니 폴백도 안 남는다.
    #    세는 대상을 이름으로 밝힌다 (`fallbacks.WIRED`).
    from pettriage.graph.fallbacks import UNWIRED, WIRED

    L.append(
        f"  {len(WIRED)}태스크 전부 모델   {fmt(s.fully_llm_rate):>7}   ({s.fully_llm}/{s.n})"
        f"   [{' · '.join(WIRED)}]"
    )
    L.append(
        f"  ⚠️ 집계에 없는 태스크: {' · '.join(UNWIRED)} — 프롬프트는 있으나 "
        "호출부가 없다 (05 §4 와 어긋남)"
    )
    if s.fallback_counts:
        L.append("  폴백으로 처리된 태스크 (분모 = 전체 건수)")
        for task, cnt in s.fallback_counts.most_common():
            flag = "  ← 한 번도 모델을 타지 않았다" if cnt == s.n else ""
            L.append(f"    {task:12} {cnt:>4}/{s.n}{flag}")
    else:
        L.append("  폴백 없음 — 모든 태스크가 모델을 탔다.")
    if s.fully_llm == 0 and not model_name.startswith("none"):
        L.append("  🔴 **모델을 붙였는데 전 건이 어딘가에서 폴백을 탔다.**")
        L.append("     아래 지표는 비교군 성능이 아니다. 키·한도·프롬프트를 먼저 확인한다.")

    # ── ④ 근거 검증 ────────────────────────────────────────────
    #
    # 🔴 **이것은 탐지율이지 재현율이 아니다.** 04 는 *"근거없음 탐지 재현율"* 을
    #    요구했는데, 재현율을 재려면 *"실제로 근거 없는 문장"* 의 정답 라벨이 있어야
    #    한다. 우리에게는 없다. 없는 것을 있는 것처럼 부르지 않는다.
    L.append("\n■ ④ 근거 검증 (02 §2 — 이 프로젝트의 핵심)")
    if s.ground_cases:
        L.append(f"  검증이 돈 건수    {s.ground_cases:>4}/{s.n}   문장 {s.ground_sentences}개")
        L.append(f"  근거없음 판정     {s.ground_unsupported:>4}문장")
        L.append(f"  모순 판정         {s.ground_contradicted:>4}문장")
        L.append(f"  재검색으로 감     {s.ground_retried:>4}건")
        if s.ground_unsupported == 0:
            L.append("  ⚠️ **한 문장도 못 걸렀다.** 환각이 없었다는 뜻이 아니다 —")
            L.append("     검증기가 약해서 못 잡은 것과 구별되지 않는다.")
    else:
        L.append("  검증이 한 번도 돌지 않았다 (전건이 거절·되묻기로 끝났다).")
    L.append("  ⚠️ 검증기는 **LLM 이 아니라 2-gram 문자 일치율**이다 —")
    L.append("     `Task.VERIFY` 프롬프트가 있으나 `verify_grounding` 이 부르지 않는다.")
    L.append("     05 §4 는 ④를 LLM 태스크로 적어 두었다. **문서와 코드가 어긋나 있다.**")
    L.append("  ⚠️ 위 숫자는 **탐지율**이다. 재현율은 정답 라벨이 없어 잴 수 없다 (04 §8).")

    L.append("\n■ 근거·문구")
    L.append(f"  must_cite 적중(any)  {fmt(s.cite_any_rate):>7}   ({s.cite_any}/{s.cite_n})")
    L.append(f"  must_cite 적중(all)  {fmt(s.cite_all_rate):>7}   ({s.cite_all}/{s.cite_n})")
    L.append(f"  must_contain (any)   {fmt(s.contain_rate):>7}   ({s.contain_ok}/{s.contain_n})")
    L.append(
        f"  must_contain (all)   {fmt(s.contain_all_rate):>7}   ({s.contain_all}/{s.contain_n})"
    )
    # 🔴 **되묻기 문구 채점을 답변 채점과 섞지 않는다.**
    #    `clarify` 기대 건에서 `must_contain` 은 *우리가 정한 되묻기 문장의 표현*이
    #    골든셋 표기와 같은지를 본다 — `무엇을 먹었나요?` vs `무엇을 먹었는지`(G-014).
    #    행동은 옳은데 어미가 달라 실패한다. 섞으면 **문구 문제가 성능 문제로 보인다.**
    L.append(
        f"    ▸ answered 만        {fmt(s.contain_answered_rate):>7}   "
        f"({s.contain_answered_ok}/{s.contain_answered_n})   ← 답에 필요한 말이 들어갔나"
    )
    L.append(
        f"    ▸ clarify 만         {fmt(s.contain_clarify_rate):>7}   "
        f"({s.contain_clarify_ok}/{s.contain_clarify_n})   ← **되묻기 문구 표기 일치**"
    )
    if s.missed_terms:
        L.append("  못 채운 문구 (빈도순) — 같은 것이 여러 건이면 골든셋 쪽을 본다")
        L.append("    " + " · ".join(f"{t}×{n}" for t, n in s.missed_terms.most_common(8)))
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
                # **거절했는데 이유가 다르다** — 상태만 보면 통과로 보이는 자리다.
                else f"거절이유 {r.expected_refusal_reason}→{r.actual_refusal_reason or '(없음)'}"
                if r.reason_ok is False
                else (
                    f"등급 {r.expected_level}→{r.actual_level}"
                    # **원인을 한 줄에 같이 낸다** — 규칙이 틀렸나, LLM 이 덮었나.
                    + (
                        f" (rule={r.rule_level} llm={r.llm_level}"
                        + (" ↑LLM" if r.gate_raised else "")
                        + (" 🔒차단" if r.gate_overridden else "")
                        + (" 🔒상승차단" if r.gate_capped else "")
                        + ")"
                        if r.rule_level is not None or r.llm_level is not None
                        else ""
                    )
                )
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
    ap.add_argument(
        "--arm",
        choices=["none", "A", "A-LC", "C", "D"],
        help=(
            "04 §3 비교군을 이름으로 고른다 (models/serving/arms.py). "
            "`PETTRIAGE__MODEL__*` 를 손으로 맞추다 하나 빠뜨리면 "
            "**다른 조건으로 재고도 모른다** — D 를 잰다고 생각하며 C 를 잰다"
        ),
    )
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

    # ⚠️ **설정을 읽기 전에** 세운다 — `get_config` 는 `lru_cache` 라 한 번 읽히면 굳는다.
    if a.arm:
        from pettriage.models.serving.arms import apply_arm

        print(f"비교군 {a.arm} — {apply_arm(a.arm)}")

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
    from pettriage.models.serving.factory import client_name

    model_name = client_name()
    print(report(results, engine_name=engine.name, model_name=model_name))

    if a.json:
        a.json.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "engine": engine.name,
            "model": model_name,
            # **엔진 이름만으로는 무엇을 잰 건지 모른다** (04 §8).
            "provenance": _provenance(),
            "goldenset": [p.name for p in paths],
            "n": len(results),
            # **이 실행이 LLM 을 잰 것인가** (D-76). 리포트에서만 보이면 JSON 을 나중에
            # 비교할 때 그 사실이 사라진다 — 04 §8 이 요구하는 것은 *숫자와 조건이 같이*다.
            "grounding": {
                "cases": summarize(results).ground_cases,
                "sentences": summarize(results).ground_sentences,
                "unsupported": summarize(results).ground_unsupported,
                "contradicted": summarize(results).ground_contradicted,
                "retried": summarize(results).ground_retried,
                "note": "탐지율이지 재현율이 아니다. 검증기는 LLM 이 아니라 2-gram 이다.",
            },
            "llm": {
                "fully_llm": summarize(results).fully_llm,
                "fallback_counts": dict(summarize(results).fallback_counts),
            },
            # 산출물 ④ — **게이트가 작동했다는 증거**를 숫자로 남긴다 (D-09).
            "gate": {
                "n": summarize(results).gate_n,
                "agreed": summarize(results).gate_agreed,
                "raised": summarize(results).gate_raised,
                "raised_wrong": summarize(results).gate_raised_wrong,
                "blocked": summarize(results).gate_blocked,
                "capped": summarize(results).gate_capped,
                "capped_wrong": summarize(results).gate_capped_wrong,
            },
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

    # 🔴 **중대 과소평가는 상한이 없다. 0이 아니면 실패다** (04 §4.1.0).
    #
    #    2026-08-03 까지 이 지표에는 게이트가 아예 없었다. *"0을 목표로 한다"* 고
    #    문서가 못박은 유일한 지표인데, 1건이 나와도 `--fail-under 0.05` 는
    #    `1/29 = 3.4% ≤ 5%` 로 초록이었다. 분모를 고친 지금은 `1/8 = 12.5%` 지만,
    #    애초에 **비율로 볼 것이 아니라 건수로 볼 것**이다.
    #
    #    `--fail-under` 블록 **밖**에 둔다. 안에 두면 그 인자를 안 준 실행에서
    #    조용히 통과한다 (`--fail-missed` 가 정확히 그 상태였다).
    s = summarize(results)
    if s.critical_under:
        print(
            f"✗ 🔴 중대 과소평가 {s.critical_under}건 "
            f"(정답 EMERGENCY 이면서 등급을 낸 {s.critical_n}건 중) — "
            "04 §4.1.0 은 0을 목표로 한다. 상한을 두지 않는다."
        )
        return 1

    # **등급을 아예 못 낸 긴급 건**은 과소평가율의 분모에 없다 (04 §1.2).
    # 그래서 별도 게이트가 필요하다 — 없으면 "전부 거절" 이 0.0% 로 통과한다.
    #
    # ⚠️ 이 검사도 `--fail-under` 블록 밖으로 꺼냈다. 안에 있던 동안에는
    #    `--fail-missed` 만 주고 돌리면 **아무것도 검사하지 않고 종료코드 0** 이었다.
    if (
        a.fail_missed is not None
        and s.missed_urgent_rate is not None
        and s.missed_urgent_rate > a.fail_missed
    ):
        print(f"✗ 등급을 못 낸 긴급 건 {fmt(s.missed_urgent_rate)} > 상한 {fmt(a.fail_missed)}")
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
