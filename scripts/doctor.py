#!/usr/bin/env python3
"""새 기계에서 **돌 준비가 됐는지** 한 번에 본다.

    python scripts/doctor.py

## 왜 필요한가

    저장소를 새로 받으면 세 가지가 없다 — **파이썬 환경 · API 키 · 벡터 인덱스.**
    셋 다 없어도 서버는 뜨고 평가도 돈다. **그리고 전부 거절로 집계된다.**

    2026-08-02 실측 — `langgraph` 가 없는 환경에서 서버가 **정상 기동**했고 모든
    질의가 `판정불가` 로 나갔다. HTTP 200 이었다. 팀원이 `git pull` 만 하고 재설치를
    안 하면 정확히 이 상태가 되고, *"시스템이 다 거절해요"* 만 보인다 (D-64).

    **조용히 반쯤 도는 것이 가장 나쁘다.** 이 스크립트는 그 반쪽을 이름으로 부른다.

## 무엇을 하지 않는가

    **고치지 않는다.** 무엇이 없고 무엇을 하면 되는지만 적는다 —
    설치·적재는 시간과 디스크를 쓰는 일이라 사람이 정할 일이다.
    **가중치를 받지 않는다** (모델 호출은 `smoke_llm.py --call`).
"""

from __future__ import annotations

import importlib.util
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

OK, WARN, BAD = "✅", "⚠️ ", "❌"
_bad = 0


def line(mark: str, what: str, detail: str = "", fix: str = "") -> None:
    global _bad
    if mark is BAD:
        _bad += 1
    print(f"  {mark} {what:26} {detail}")
    if fix:
        print(f"       → {fix}")


def _has(mod: str) -> bool:
    try:
        return importlib.util.find_spec(mod) is not None
    except (ImportError, ValueError):
        return False


def section(title: str) -> None:
    print(f"\n── {title} " + "─" * max(0, 48 - len(title)))


def check_python() -> None:
    section("파이썬")
    v = sys.version_info
    ok = (v.major, v.minor) == (3, 11)
    line(
        OK if ok else WARN,
        "버전",
        f"{v.major}.{v.minor}.{v.micro}",
        "" if ok else "3.11 로 맞추는 것을 권한다 (pyproject target-version)",
    )
    line(
        OK if sys.prefix != sys.base_prefix else WARN,
        "가상환경",
        Path(sys.prefix).name,
        "" if sys.prefix != sys.base_prefix else "python -m venv .venv 후 활성화",
    )


def check_packages() -> None:
    section("패키지")
    #: (모듈, 이름, 없으면 무엇이 안 되나, 필수인가)
    need = [
        ("pettriage", "pettriage(본체)", "아무것도 안 된다", True),
        ("langgraph", "langgraph", "질의 그래프를 못 만든다 (D-64)", True),
        ("pydantic", "pydantic", "계약이 안 선다", True),
        ("chromadb", "chromadb", "벡터 검색이 안 된다 (D-44)", True),
        ("sentence_transformers", "sentence-transformers", "임베딩을 못 만든다", True),
        ("openai", "openai", "--arm A 를 못 돌린다", False),
        ("langchain_openai", "langchain-openai", "--arm A-LC 를 못 돌린다 (D-71)", False),
        ("fastapi", "fastapi", "서버를 못 띄운다", False),
        # 🔴 **`[db]` 가 없으면 조용히 검증이 빠진다.** `tests/test_auth_api.py` 는
        #    `importorskip` 으로 **모듈째** 건너뛰어지고, 요약줄에는 `1 skipped` 로만
        #    보인다 — 실제로는 회원가입·로그인·프로필 **25건이 안 돈다.**
        #    2026-08-03 실측: `518 passed, 1 skipped` 를 머지 안전 신호로 읽었는데
        #    그 머지가 바꾼 것이 `routes/pets.py` 였다. **필수로 둔다.**
        ("sqlalchemy", "SQLAlchemy [db]", "인증·프로필 25건이 안 돈다", True),
        ("bcrypt", "bcrypt [db]", "인증·프로필 25건이 안 돈다", True),
        ("jwt", "PyJWT [db]", "인증·프로필 25건이 안 돈다", True),
        ("torch", "torch", "--arm C/D (로컬 Qwen) 를 못 돌린다", False),
    ]
    for mod, name, why, required in need:
        if _has(mod):
            line(OK, name)
        else:
            line(BAD if required else WARN, name, f"없음 — {why}")
    if not all(_has(m) for m, _, _, req in need if req):
        print("       → pip install -e '.[api,rag,ingest,db,dev]' -c constraints.txt")


def check_config() -> None:
    section("설정")
    profile = os.getenv("PETTRIAGE_PROFILE")
    line(
        OK if profile else WARN,
        "PETTRIAGE_PROFILE",
        profile or "(없음 → default)",
        "" if profile else '평가는 eval 로 돈다: $env:PETTRIAGE_PROFILE="eval"',
    )
    try:
        from pettriage.config import get_config

        cfg = get_config()
    except Exception as e:  # noqa: BLE001
        line(BAD, "설정 읽기", f"{type(e).__name__}: {e}")
        return
    line(OK, "engine", cfg.serve.engine)
    line(OK, "provider", cfg.model.provider)
    if cfg.model.provider in ("api", "langchain"):
        line(OK, "api_model", cfg.model.api_model)
        line(OK, "api_base_url", cfg.model.api_base_url or "(OpenAI 본가)")


def check_secrets() -> None:
    section("비밀")
    env = ROOT / ".env"
    line(
        OK if env.exists() else WARN,
        ".env",
        "있음" if env.exists() else "없음",
        "" if env.exists() else "cp .env.example .env  (PowerShell: Copy-Item)",
    )
    # 🔴 **`.env` 를 직접 읽는다.** 예전에는 `os.getenv` 만 봤는데, 앱은 키를
    #    pydantic-settings 로 `.env` 에서 읽으므로 **키가 멀쩡히 있어도 "없음"** 이
    #    찍혔다 (2026-08-03). 없는 문제를 쫓게 만드는 점검은 없느니만 못하다.
    #    ⚠️ **값을 찍지 않는다** — 길이와 접두사만 본다. 화면 공유·로그에 키가 남는다.
    key = os.getenv("OPENAI_API_KEY", "").strip()
    where = "셸 환경변수"
    if not key and env.exists():
        for raw in env.read_text(encoding="utf-8", errors="replace").splitlines():
            k, sep, v = raw.partition("=")
            if sep and k.strip() == "OPENAI_API_KEY":
                key = v.strip().strip("'\"")
                where = ".env"
                break
    if key:
        line(OK, "OPENAI_API_KEY", f"{key[:7]}… ({len(key)}자 · {where})")
    else:
        line(
            WARN,
            "OPENAI_API_KEY",
            "없음 — --arm none 기준선만 돈다",
            ".env 에 넣는다. **셸에 넣지 않는다** — --arm 이 지우지는 않지만 이력에 남는다",
        )

    # `.env` 에 살아 있으면 OpenAI 키로 Gemini 엔드포인트를 때려 401 이 난다.
    lines = env.read_text(encoding="utf-8", errors="replace").splitlines() if env.exists() else []
    for raw in lines:
        s = raw.strip()
        if s.startswith("PETTRIAGE__MODEL__API_BASE_URL") and s.partition("=")[2].strip():
            line(
                WARN,
                "API_BASE_URL",
                ".env 에 살아 있다",
                "다른 사업자를 가리키면 401 이다. OpenAI 를 쓰면 **주석 처리한다**",
            )
            break


def check_tables() -> None:
    section("표 (커밋되어 있다)")
    tables = ROOT / "src" / "pettriage" / "compute" / "tables"
    for name in ("물질어휘.csv", "별칭.csv", "정량임계치.csv", "정성등급.csv", "성분함량.csv"):
        p = tables / name
        n = sum(1 for _ in p.open(encoding="utf-8-sig")) - 1 if p.exists() else 0
        line(
            OK if p.exists() else BAD,
            name,
            f"{n}행" if p.exists() else "없음",
            "" if p.exists() else "make rules",
        )

    facts = sorted((ROOT / "data" / "facts").glob("facts_*.csv"))
    rows = sum(sum(1 for _ in p.open(encoding="utf-8-sig")) - 1 for p in facts)
    line(
        OK if facts else BAD,
        "사실 표",
        f"{len(facts)}개 파일 · {rows}행" if facts else "없음",
        "" if facts else "data/facts/ 가 비었다. D-37 대로 커밋되어야 한다 — git pull 확인",
    )

    golden = sorted((ROOT / "eval" / "goldenset").glob("golden_*.csv"))
    g = sum(sum(1 for _ in p.open(encoding="utf-8-sig")) - 1 for p in golden)
    line(OK if g else BAD, "골든셋", f"{len(golden)}개 파일 · {g}건" if golden else "없음")


def check_index() -> None:
    section("벡터 인덱스 (커밋되지 않는다 — 각자 만든다)")
    try:
        from pettriage.config import get_config

        persist = ROOT / get_config().retrieval.persist_dir
    except Exception:  # noqa: BLE001
        persist = ROOT / ".chroma"
    if not persist.exists():
        line(
            BAD,
            "인덱스",
            f"없음 ({persist.name})",
            "python scripts/build_index.py --store chroma   ← **처음이면 임베딩 모델을 받는다**",
        )
        return
    try:
        # **검색 노드가 쓰는 팩토리를 그대로 쓴다** — 다른 경로로 열면
        # *"점검은 통과했는데 실행은 안 되는"* 상태가 생긴다 (D-22).
        from pettriage.graph.nodes.retrieve import _default_store

        n = _default_store().count()
        line(
            OK if n else BAD,
            "인덱스",
            f"{n}개 청크" if n else "비었다",
            "" if n else "build_index.py --store chroma",
        )
    except Exception as e:  # noqa: BLE001
        line(
            WARN,
            "인덱스",
            f"폴더는 있으나 열지 못했다 — {type(e).__name__}",
            "지우고 다시 적재한다",
        )


def check_model_cache() -> None:
    section("임베딩 모델 캐시")
    try:
        from pettriage.config import get_config

        name = get_config().retrieval.embedding_model
    except Exception:  # noqa: BLE001
        name = "BAAI/bge-m3"
    home = Path(os.getenv("HF_HOME", Path.home() / ".cache" / "huggingface"))
    hub = home / "hub" / ("models--" + name.replace("/", "--"))
    if hub.exists():
        size = sum(f.stat().st_size for f in hub.rglob("*") if f.is_file()) / 1e9
        line(OK, name, f"{size:.1f}GB 캐시됨")
    else:
        line(
            WARN,
            name,
            "캐시 없음",
            "첫 실행에서 자동으로 받는다 (약 4.3GB). 네트워크·디스크를 미리 확보한다",
        )


def check_git() -> None:
    section("저장소")
    import subprocess

    def git(*a: str) -> str:
        try:
            return subprocess.run(
                ["git", *a],
                capture_output=True,
                text=True,
                encoding="utf-8",
                errors="replace",
                timeout=5,
                check=False,
                cwd=ROOT,
            ).stdout.strip()
        except Exception:  # noqa: BLE001
            return ""

    head = git("rev-parse", "--short", "HEAD")
    branch = git("rev-parse", "--abbrev-ref", "HEAD")
    line(OK if head else WARN, "HEAD", f"{head} ({branch})" if head else "git 저장소가 아니다")
    if not head:
        # ⚠️ git 이 없으면 `status` 도 빈 문자열이라 **"깨끗함"으로 보인다.**
        #    모르는 것을 좋은 쪽으로 읽지 않는다 (D-79 와 같은 성질의 함정).
        line(WARN, "작업 트리", "확인 불가 (git 이 없다)")
        return
    dirty = bool(git("status", "--porcelain"))
    # 🔴 커밋 안 된 변경 위에서 잰 숫자는 **남에게 줄 수 없다** (04 §8).
    line(
        OK if not dirty else WARN,
        "작업 트리",
        "깨끗함" if not dirty else "커밋 안 된 변경이 있다",
        "" if not dirty else "이 상태에서 잰 결과는 재현할 수 없다 — 측정 전에 커밋한다",
    )


def main() -> int:
    print("=" * 60)
    print("  환경 점검 — 새 기계에서 돌 준비가 됐나")
    print("=" * 60)
    check_python()
    check_packages()
    check_config()
    check_secrets()
    check_tables()
    check_index()
    check_model_cache()
    check_git()

    print("\n" + "=" * 60)
    if _bad:
        print(f"  ❌ 막히는 것 {_bad}개 — 위 → 를 따라간다")
        print("=" * 60)
        return 1
    print("  ✅ 막히는 것 없음")
    print("=" * 60)
    print("\n다음 —")
    # ⚠️ `-q` 를 붙이지 않는다. `addopts` 에 이미 있어 `-qq` 가 되면
    #    **`X passed` 요약줄이 사라진다** (2026-08-03).
    print("  pytest                                      # 546 passed, 22 deselected")
    print("  python scripts/smoke_llm.py --arm A")
    print("  python eval/harness/run_eval.py --arm none --json eval/reports/확인.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
