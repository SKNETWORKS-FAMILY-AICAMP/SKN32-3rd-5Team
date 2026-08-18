"""`.env` 를 만들고 **쓸 수 있는지까지 검증한다.**

    python scripts/setup_env.py              # 대화형 — 없는 값만 묻는다
    python scripts/setup_env.py --check      # 고치지 않고 검사만 한다

`doctor.py` 와 나누는 이유 — `doctor.py` 는 *"있나"* 를 보고, 이것은 *"쓸 수 있나"* 를 본다.
2026-08-04 에 겪은 것들이 그 차이다.

  · `OPENAI_API_KEY=...` 세 글자 자리표시자가 ✅ 로 찍혔다 — 있긴 있었으니까
  · 편집기에서 고쳤는데 저장이 안 돼 값이 계속 비어 있었다
  · `JWT_SECRET_KEY` 가 두 줄로 중복돼 뒤엣것이 이겼다
  · 셸의 빈 환경변수가 `.env` 를 가렸다 (환경변수가 dotenv 보다 우선한다)
  · 메모장 저장으로 인코딩이 바뀌면 한글 주석에서 파싱이 깨진다

**값을 화면에 찍지 않는다.** 길이와 판정만 낸다 — 콘솔 기록에 키가 남으면 그것도 유출이다.
"""

from __future__ import annotations

import argparse
import os
import re
import secrets
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
ENV = ROOT / ".env"
EXAMPLE = ROOT / ".env.example"

OK, WARN, BAD = "✅", "⚠️ ", "❌"

#: 자리표시자로 자주 남는 값들. **있음/없음만 보면 이것들이 통과한다.**
PLACEHOLDERS = {"", "...", "sk-...", "sk-xxx", "changeme", "your-key-here", "<키>"}


def _line_ending(raw: bytes) -> str:
    return "\r\n" if b"\r\n" in raw else "\n"


def _read() -> tuple[list[str], str]:
    raw = ENV.read_bytes()
    if raw.startswith(b"\xef\xbb\xbf"):
        raw = raw[3:]  # BOM 은 첫 변수명을 망가뜨린다. 조용히 걷어낸다
    return raw.decode("utf-8").replace("\r\n", "\n").split("\n"), _line_ending(ENV.read_bytes())


def _write(lines: list[str], eol: str) -> None:
    """**UTF-8, BOM 없이** 쓴다. 줄끝은 원본을 따른다."""
    ENV.write_bytes(eol.join(lines).encode("utf-8"))


def _find(lines: list[str], key: str) -> list[int]:
    pat = re.compile(rf"^\s*{re.escape(key)}\s*=")
    return [i for i, ln in enumerate(lines) if pat.match(ln)]


def _value(lines: list[str], key: str) -> str:
    idx = _find(lines, key)
    if not idx:
        return ""
    return lines[idx[-1]].split("=", 1)[1].strip().strip('"').strip("'")


def _set(lines: list[str], key: str, value: str) -> list[str]:
    """**마지막 정의를 고치고 나머지 중복은 지운다.** 뒤엣것이 이기므로 그것이 실효값이다."""
    idx = _find(lines, key)
    if not idx:
        return [*lines, f"{key}={value}"]
    lines[idx[-1]] = f"{key}={value}"
    for i in reversed(idx[:-1]):
        del lines[i]
    return lines


def _mask(v: str) -> str:
    """**길이만 낸다.** 값의 일부도 보여 주지 않는다.

    앞뒤 몇 글자를 보여 주는 흔한 방식을 쓰지 않는 이유 — 콘솔 기록·화면 공유·
    스크린샷에 남는다. 여기서 확인해야 하는 것은 *"들어갔나"* 이지 *"무엇이 들어갔나"*
    가 아니다. 길이만으로 자리표시자와 실물이 구분된다.
    """
    return f"{len(v)}자" if v else "(비어 있음)"


# ── 검사 ─────────────────────────────────────────────────────


def _check(lines: list[str]) -> list[str]:
    """문제 목록을 낸다. 비어 있으면 통과."""
    problems: list[str] = []

    for key, need, hint in (
        ("OPENAI_API_KEY", False, "없으면 --arm none 기준선만 돈다 (LLM 태스크 전부 폴백)"),
        ("JWT_SECRET_KEY", True, "없으면 로그인이 실패한다 (JWTKeyMissingError)"),
        ("DATABASE_URL", True, "없으면 회원가입·프로필·다이어리가 통째로 빠진다"),
    ):
        v = _value(lines, key)
        dup = len(_find(lines, key))
        if v in PLACEHOLDERS:
            mark = BAD if need else WARN
            problems.append(f"{mark} {key:16} 비었거나 자리표시자다 — {hint}")
        elif key == "OPENAI_API_KEY" and len(v) < 20:
            problems.append(f"{BAD} {key:16} {len(v)}자 — 너무 짧다. 자리표시자가 남아 있다")
        else:
            print(f"  {OK} {key:16} {_mask(v)}")
        if dup > 1:
            problems.append(
                f"{WARN} {key:16} {dup}줄 중복 — 뒤엣것이 실효값이다. "
                f"고칠 때 앞엣것을 지운다(값은 보존)"
            )

    # 셸이 .env 를 가리는가 — **환경변수가 dotenv 보다 우선한다**
    for key in ("OPENAI_API_KEY", "JWT_SECRET_KEY", "DATABASE_URL"):
        shell = os.environ.get(key)
        if shell is not None:
            problems.append(
                f"{WARN} {key:16} 셸에 있다({_mask(shell)}) — **.env 를 가린다.** "
                f"지우려면  Remove-Item Env:{key}"
            )

    if os.environ.get("PETTRIAGE_PROFILE") is None:
        print(f"  {WARN} PETTRIAGE_PROFILE  없음 → default 프로필 = engine 이 stub 이다")
        print("       시연·평가는 eval 로 돈다. 런처가 세우거나 직접:")
        print('         $env:PETTRIAGE_PROFILE="eval"')

    return problems


def _verify_loadable() -> list[str]:
    """설정이 **최종적으로 무엇을 보는지** 확인한다. 파일만 보면 우선순위를 못 잡는다."""
    problems: list[str] = []
    try:
        from dotenv import dotenv_values
    except ImportError:
        return [f"{WARN} python-dotenv 가 없다 — 패키지 설치가 먼저다"]

    dv = dotenv_values(str(ENV))
    for key in ("OPENAI_API_KEY", "JWT_SECRET_KEY", "DATABASE_URL"):
        got = dv.get(key)
        if got is None:
            problems.append(f"{BAD} {key} 를 dotenv 가 못 읽는다 — 인코딩이나 문법 문제다")
        elif got.strip() == "":
            problems.append(f"{BAD} {key} 가 빈 값으로 읽힌다")
    return problems


# ── 대화형 ───────────────────────────────────────────────────


def _ask_secret(prompt: str) -> str:
    """입력을 화면에 남기지 않는다."""
    import getpass

    try:
        return getpass.getpass(prompt).strip()
    except Exception:  # pragma: no cover - 일부 터미널
        return input(prompt).strip()


def _interactive(lines: list[str]) -> list[str]:
    print("\n── 없는 값을 채웁니다 (엔터로 건너뜁니다) ──")

    if _value(lines, "OPENAI_API_KEY") in PLACEHOLDERS or len(_value(lines, "OPENAI_API_KEY")) < 20:
        print("\n  OPENAI_API_KEY — 없으면 LLM 태스크가 전부 폴백으로 돕니다.")
        print("  입력은 화면에 보이지 않습니다.")
        v = _ask_secret("  붙여넣기: ")
        if v:
            if len(v) < 20:
                print(f"  {BAD} {len(v)}자 — 너무 짧습니다. 넣지 않았습니다.")
            else:
                lines = _set(lines, "OPENAI_API_KEY", v)
                print(f"  {OK} {len(v)}자 저장")

    if _value(lines, "JWT_SECRET_KEY") in PLACEHOLDERS:
        gen = secrets.token_urlsafe(48)
        lines = _set(lines, "JWT_SECRET_KEY", gen)
        print(f"\n  {OK} JWT_SECRET_KEY  자동 생성했습니다 ({len(gen)}자)")

    if _value(lines, "DATABASE_URL") in PLACEHOLDERS:
        default = "sqlite+pysqlite:///./pettriage.sqlite3"
        lines = _set(lines, "DATABASE_URL", default)
        print(f"  {OK} DATABASE_URL    SQLite 기본값을 넣었습니다 (설치 불필요)")
        print(f"       {default}")

    return lines


def main() -> int:
    ap = argparse.ArgumentParser(description=".env 를 만들고 쓸 수 있는지 검증한다")
    ap.add_argument("--check", action="store_true", help="고치지 않고 검사만 한다")
    a = ap.parse_args()

    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")

    print("=" * 60)
    print("  .env 설정 · 검증")
    print("=" * 60)

    if not ENV.exists():
        if a.check:
            print(f"{BAD} .env 가 없습니다.")
            return 1
        if not EXAMPLE.exists():
            print(f"{BAD} .env 도 .env.example 도 없습니다. 저장소가 온전한지 확인하세요.")
            return 1
        ENV.write_bytes(EXAMPLE.read_bytes())
        print(f"{OK} .env.example 에서 .env 를 만들었습니다.")

    lines, eol = _read()

    if not a.check:
        # ⚠️ **백업 파일을 만들지 않는다.** `.env.bak` 은 평문 키 사본이 디스크에
        #    하나 더 생기는 것이고, 이 스크립트에는 파괴적 동작이 없다 —
        #    이미 값이 있는 항목은 묻지도 건드리지도 않고, 지우는 것은 중복 키의
        #    **앞엣것**(뒤엣것이 이기므로 이미 무효인 값)뿐이다.
        dups = [
            k
            for k in ("OPENAI_API_KEY", "JWT_SECRET_KEY", "DATABASE_URL")
            if len(_find(lines, k)) > 1
        ]
        if dups:
            print(f"\n{WARN} 중복된 키가 있습니다: {' '.join(dups)}")
            print("   마지막 정의(실효값)만 남기고 앞엣것을 지웁니다.")
            if input("   계속할까요? [Y/n] ").strip().lower() == "n":
                return 1
        lines = _interactive(lines)
        _write(lines, eol)
        eol_name = "CRLF" if eol == "\r\n" else "LF"
        print(f"\n{OK} 저장했습니다 — UTF-8, BOM 없음, 줄끝 {eol_name}")
        lines, eol = _read()

    print("\n── 검사 ──")
    problems = _check(lines) + _verify_loadable()

    print()
    if problems:
        for p in problems:
            print(f"  {p}")
        print("\n" + "=" * 60)
        blocking = [p for p in problems if p.startswith(BAD)]
        if blocking:
            print(f"  {BAD} 막히는 것 {len(blocking)}개 — 위를 따라가세요")
            print("=" * 60)
            return 1
        print(f"  {OK} 돌릴 수 있습니다 (경고 {len(problems)}개)")
        print("=" * 60)
        return 0

    print("=" * 60)
    print(f"  {OK} 막히는 것 없음")
    print("=" * 60)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
