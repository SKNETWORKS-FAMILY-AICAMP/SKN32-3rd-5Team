#!/usr/bin/env python
"""층 0 — 데이터 파이프라인 검증.

설계 근거: docs/04_테스트-평가계획.md §2.5 · docs/06 D-38

    대부분의 자료가 원문이 아니라 우리가 문장화한 형태로 적재되므로,
    문장화 자체가 검증 대상이 된다.
    검증 대상은 문장이 아니라 **필드**다 — 그래서 자동화할 수 있다.

지표 3종
    추출 정확도       숫자·단위·종이 원문과 일치하는 비율
    문장화 충실도     원문에 없는 내용이 포함된 비율        목표 0
    역추적 가능성     source_id + 원문 위치 보유 비율       목표 100%

실행:  make verify   또는   python scripts/verify_corpus.py
"""

from __future__ import annotations

import csv
import re
import sys
from dataclasses import dataclass
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
MANIFESTS = ROOT / "data" / "manifests"
SNAPSHOTS = ROOT / "data" / "snapshot"

# 단위 변환 검사쌍 — S-074에서 실제로 오류가 나왔다
#   "530° F (280° Celsius)"  →  280°C = 536°F 이므로 불일치
F_C_PATTERN = re.compile(r"(\d{2,4})\s*°?\s*F[^0-9]{0,12}?(\d{2,4})\s*°?\s*C", re.I)

# 1,000배 오식 탐지 — S-034 본문 "2.8 mg/kg" vs 같은 논문 Table "2.8-36.4 g/kg"
DOSE_PATTERN = re.compile(r"(\d[\d.,]*)\s*(mg|g)\s*/\s*kg", re.I)


@dataclass
class Finding:
    level: str  # ERROR | WARN | INFO
    where: str
    message: str

    def __str__(self) -> str:
        icon = {"ERROR": "✗", "WARN": "⚠", "INFO": "·"}[self.level]
        return f"  {icon} [{self.where}] {self.message}"


def check_manifest_vs_disk() -> list[Finding]:
    """대장 ↔ 실파일 정합성. 자료 파일은 커밋되지 않으므로 로컬에서만 의미가 있다."""
    out: list[Finding] = []
    path = MANIFESTS / "SNAPSHOT_MANIFEST.csv"
    if not path.exists():
        return [Finding("ERROR", "manifest", "SNAPSHOT_MANIFEST.csv 가 없다")]

    rows = list(csv.DictReader(path.open(encoding="utf-8")))
    on_disk = (
        {str(p.relative_to(ROOT / "data")) for p in SNAPSHOTS.rglob("*.md")}
        if SNAPSHOTS.exists()
        else set()
    )
    if not on_disk:
        # 자료는 커밋되지 않는다 (D-29 · 공개 저장소). 대장만으로 검증한다.
        out.append(
            Finding(
                "INFO",
                "snapshot",
                f"자료 파일이 로컬에 없다 (대장 {len(rows)}행). "
                "이 저장소는 자료를 커밋하지 않는다 — 정상이다. "
                "전체 검증은 자료를 내려받은 뒤 다시 실행한다.",
            )
        )
        return out

    in_manifest = {r["file"] for r in rows}
    for missing in sorted(in_manifest - on_disk):
        out.append(Finding("ERROR", "manifest", f"대장에 있으나 파일 없음: {missing}"))
    for extra in sorted(on_disk - in_manifest):
        out.append(Finding("ERROR", "manifest", f"파일은 있으나 대장에 없음: {extra}"))
    return out


def check_deleted_not_present() -> list[Finding]:
    """삭제 판정 자료가 되살아나지 않았는지. D-33 · D-37."""
    out: list[Finding] = []
    path = MANIFESTS / "DELETION_LOG.csv"
    if not path.exists():
        return out
    deleted = {r["source_id"] for r in csv.DictReader(path.open(encoding="utf-8"))}
    for csv_name in ("SNAPSHOT_MANIFEST.csv", "MANIFEST.csv"):
        p = MANIFESTS / csv_name
        if not p.exists():
            continue
        live = {r["source_id"] for r in csv.DictReader(p.open(encoding="utf-8"))}
        for sid in sorted(deleted & live):
            out.append(Finding("ERROR", "deletion", f"{sid} 는 삭제 판정인데 {csv_name} 에 있다"))
    if not out:
        out.append(Finding("INFO", "deletion", f"삭제 {len(deleted)}건 — 잔존 없음"))
    return out


def check_unit_conversions(text: str, where: str) -> list[Finding]:
    """화씨↔섭씨 재계산. 자동으로 잡히는 오류 유형."""
    out: list[Finding] = []
    for f_val, c_val in F_C_PATTERN.findall(text):
        f, c = float(f_val), float(c_val)
        expected_f = c * 9 / 5 + 32
        if abs(expected_f - f) > 1.0:
            out.append(
                Finding(
                    "WARN",
                    where,
                    f"단위 변환 불일치: {f_val}°F vs {c_val}°C (계산상 {expected_f:.0f}°F)",
                )
            )
    return out


def check_traceability() -> list[Finding]:
    """역추적 가능성 — 모든 행이 source_id 를 갖는가. 목표 100%."""
    out: list[Finding] = []
    path = MANIFESTS / "SNAPSHOT_MANIFEST.csv"
    if not path.exists():
        return out
    rows = list(csv.DictReader(path.open(encoding="utf-8")))
    missing = [r for r in rows if not r.get("source_id") or not r.get("url")]
    ratio = (len(rows) - len(missing)) / len(rows) * 100 if rows else 0
    level = "INFO" if not missing else "ERROR"
    out.append(Finding(level, "역추적", f"{ratio:.1f}% ({len(rows) - len(missing)}/{len(rows)})"))
    for r in missing:
        out.append(Finding("ERROR", "역추적", f"{r.get('source_id', '?')} — source_id/url 결측"))
    return out


def check_quality_grades() -> list[Finding]:
    """`요약` 등급은 인용 근거로 쓸 수 없다 (D-30)."""
    out: list[Finding] = []
    path = MANIFESTS / "SNAPSHOT_MANIFEST.csv"
    if not path.exists():
        return out
    rows = list(csv.DictReader(path.open(encoding="utf-8")))
    from collections import Counter

    counts = Counter(r.get("quality", "?") for r in rows)
    out.append(Finding("INFO", "품질", " · ".join(f"{k} {v}" for k, v in counts.most_common())))
    blocked = counts.get("요약", 0) + counts.get("빈약", 0) + counts.get("부적합", 0)
    if blocked:
        out.append(
            Finding("WARN", "품질", f"인용 근거로 쓸 수 없는 자료 {blocked}건 — 적재 시 제외할 것")
        )
    return out


def main() -> int:
    checks = [
        ("대장 ↔ 실파일", check_manifest_vs_disk),
        ("삭제 자료 잔존", check_deleted_not_present),
        ("역추적 가능성", check_traceability),
        ("스냅샷 품질 등급", check_quality_grades),
    ]
    findings: list[Finding] = []
    print("층 0 — 데이터 파이프라인 검증 (04 §2.5)\n")
    for name, fn in checks:
        result = fn()
        findings.extend(result)
        print(f"[{name}]")
        for f in result:
            print(f)
        print()

    # 스냅샷이 로컬에 있으면 단위 변환까지 검사한다
    if SNAPSHOTS.exists():
        unit_findings: list[Finding] = []
        for p in sorted(SNAPSHOTS.rglob("*.md")):
            unit_findings += check_unit_conversions(
                p.read_text(encoding="utf-8"), p.stem.split("_")[0]
            )
        findings.extend(unit_findings)
        print("[단위 변환]")
        for f in unit_findings or [Finding("INFO", "단위", "불일치 없음")]:
            print(f)
        print()

    errors = sum(1 for f in findings if f.level == "ERROR")
    warns = sum(1 for f in findings if f.level == "WARN")
    print(f"→ ERROR {errors} · WARN {warns}")
    return 1 if errors else 0


if __name__ == "__main__":
    sys.exit(main())
