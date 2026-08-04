# -*- coding: utf-8 -*-
"""지금 무엇이 안 돼 있는지 한 줄로 보여 준다.

**런처의 절반은 이 파일이다.** 2026-08-04 에 겪은 문제 대부분이
*"지금 무엇이 안 돼 있는지 몰라서"* 생겼다 — 키가 빈 줄인 것도, 프로필이 없어
`stub` 이 답한 것도, 인덱스가 없어 검색이 0건인 것도 **돌려 보고 나서야** 알았다.

여기서 재는 것은 **빠른 것만**이다. 무거운 검사는 `scripts/doctor.py` 가 한다.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

OK, WARN, BAD = "OK", "!!", "XX"


def venv_python() -> Path:
    """이 저장소의 가상환경 파이썬. 있든 없든 경로는 계산된다."""
    if os.name == "nt":
        return ROOT / ".venv" / "Scripts" / "python.exe"
    return ROOT / ".venv" / "bin" / "python"


def in_venv() -> bool:
    return Path(sys.prefix).resolve() == (ROOT / ".venv").resolve()


def check_venv() -> tuple[str, str]:
    if not venv_python().exists():
        return BAD, "없음 - [1] 부터"
    return (OK, "활성") if in_venv() else (WARN, "있으나 비활성")


def check_packages() -> tuple[str, str]:
    """**임포트로 확인한다.** 폴더가 있다고 깔린 것이 아니다."""
    if not in_venv():
        return WARN, "venv 밖이라 확인 불가"
    missing = []
    for mod, label in (
        ("pettriage", "본체"),
        ("langgraph", "langgraph"),
        ("chromadb", "chromadb"),
        ("sqlalchemy", "[db]"),
        ("jwt", "[db]"),
    ):
        try:
            __import__(mod)
        except ImportError:
            missing.append(label)
    if missing:
        return BAD, f"빠짐: {' '.join(missing)} - [1]"
    return OK, "설치됨"


def check_env() -> tuple[str, str]:
    """`.env` 는 있고 **키가 쓸 수 있는가.** 있음/없음만 보면 자리표시자가 통과한다."""
    env = ROOT / ".env"
    if not env.exists():
        return BAD, "없음 - [2]"
    try:
        from dotenv import dotenv_values
    except ImportError:
        return WARN, "확인 불가 (패키지 없음)"
    v = dotenv_values(str(env))
    bad = []
    if len((v.get("OPENAI_API_KEY") or "").strip()) < 20:
        bad.append("OPENAI_API_KEY")
    if len((v.get("JWT_SECRET_KEY") or "").strip()) < 16:
        bad.append("JWT_SECRET_KEY")
    if not (v.get("DATABASE_URL") or "").strip():
        bad.append("DATABASE_URL")
    if bad:
        return WARN, f"{' '.join(bad)} 비었음 - [2]"
    return OK, "키 3종"


def check_index() -> tuple[str, str]:
    """벡터 인덱스. **없으면 검색이 0건이고 시스템이 아무것도 못 한다.**"""
    d = ROOT / ".chroma"
    if not d.exists():
        return BAD, "없음 - [3]"
    n = sum(1 for _ in d.rglob("*") if _.is_file())
    if n < 3:
        return WARN, "비어 있음 - [3]"
    return OK, "있음"


def check_db() -> tuple[str, str]:
    """시연용 SQLite. 0바이트면 `initdb` 를 아직 안 돌린 것이다."""
    try:
        from dotenv import dotenv_values

        url = (dotenv_values(str(ROOT / ".env")).get("DATABASE_URL") or "").strip()
    except Exception:
        url = ""
    if url and not url.startswith("sqlite"):
        return OK, "외부 DB"
    f = ROOT / "pettriage.sqlite3"
    if not f.exists() or f.stat().st_size == 0:
        return WARN, "테이블 없음 - [8]"
    return OK, f"{f.stat().st_size // 1024}KB"


def status_line() -> str:
    parts = []
    for label, fn in (
        ("venv", check_venv),
        ("패키지", check_packages),
        (".env", check_env),
        ("인덱스", check_index),
        ("DB", check_db),
    ):
        mark, note = fn()
        parts.append(f"{label} {mark}" + (f"({note})" if mark != OK else ""))
    return "  ".join(parts)


def blocking(step: str) -> str | None:
    """이 항목을 누르기 전에 먼저 해야 할 것. 없으면 None."""
    needs = {
        "2": [], "3": ["pkg"], "4": ["pkg"], "5": ["pkg", "env", "idx"],
        "6": ["pkg", "env", "idx"], "7": ["pkg"], "8": ["pkg", "env"], "9": ["pkg", "env", "idx"],
    }.get(step, [])
    if "pkg" in needs and check_packages()[0] == BAD:
        return "패키지가 없습니다. 먼저 [1] 처음 설치."
    if "env" in needs and check_env()[0] == BAD:
        return ".env 가 없습니다. 먼저 [2] .env 설정."
    if "idx" in needs and check_index()[0] == BAD:
        return "벡터 인덱스가 없습니다. 먼저 [3] 벡터 인덱스 만들기."
    return None
