"""경로① 자격 판정 — 화이트리스트가 원장과 어긋나지 않는지 (D-45).

이전 판은 자격 목록을 코드에 박아두었고 **8건 중 7건이 원장과 어긋나 있었다.**
삭제된 자료가 남아 있었고, 복제 금지 자료가 허용 목록에 있었다.
그 사고를 다시 내지 않기 위한 테스트다.
"""

from __future__ import annotations

import csv
import importlib.util
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
MANIFESTS = ROOT / "data" / "manifests"


def _load_checker():
    """`scripts/` 는 패키지가 아니라 파일 경로로 읽는다.

    `sys.modules` 에 먼저 등록해야 한다 — `@dataclass` 가 애노테이션을 풀 때
    자기 모듈을 `sys.modules` 에서 찾기 때문이다.
    """
    path = ROOT / "scripts" / "check_facts.py"
    spec = importlib.util.spec_from_file_location("check_facts", path)
    assert spec is not None and spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    sys.modules["check_facts"] = mod
    spec.loader.exec_module(mod)
    return mod


@pytest.fixture(scope="module")
def checker():
    return _load_checker()


@pytest.fixture(scope="module")
def eligible(checker) -> set[str]:
    return checker.route1_eligible()


def _rows(name: str) -> list[dict[str, str]]:
    p = MANIFESTS / name
    if not p.exists():
        return []
    return list(csv.DictReader(p.open(encoding="utf-8-sig")))


class TestRoute1Eligibility:
    def test_switch_is_off(self, checker) -> None:
        """경로①은 이번 산출물 범위에서 꺼져 있다 (D-45)."""
        assert checker.ROUTE1_ENABLED is False

    def test_deleted_sources_are_never_eligible(self, eligible: set[str]) -> None:
        """G1a 로 삭제된 자료가 자격을 얻으면 안 된다 (D-33).

        이전 판은 `S-001`(VCA, AI 학습 금지)을 허용하고 있었다.
        """
        deleted = {(r.get("source_id") or "").strip() for r in _rows("DELETION_LOG.csv")}
        assert deleted, "삭제 이력을 읽지 못했다 — 검사가 축소되었다 (04 §8)"
        assert not (eligible & deleted)

    def test_only_citable_quality(self, eligible: set[str]) -> None:
        """`요약`·`빈약`·`부적합` 은 원문 인용 근거로 쓸 수 없다 (D-30)."""
        bad = {
            (r.get("source_id") or "").strip()
            for r in _rows("SNAPSHOT_MANIFEST.csv")
            if (r.get("quality") or "").strip() not in ("완전", "부분")
        }
        assert not (eligible & bad)

    def test_copy_restricted_sources_are_excluded(self, eligible: set[str]) -> None:
        """`B 가공활용` 은 **복제 금지**다. 원문을 실으면 약관 위반이다 (D-37).

        이전 판은 `S-012`·`S-023`·`S-064` 를 허용하고 있었고,
        이 셋은 실제로 사실 표 행을 갖고 있어 사고 직전이었다.
        """
        restricted = {
            (r.get("source_id") or "").strip()
            for r in _rows("SNAPSHOT_MANIFEST.csv")
            if (r.get("usability") or "").strip().startswith("B")
        }
        assert restricted, "원장을 읽지 못했다"
        assert not (eligible & restricted)

    def test_quote_is_rejected_while_switch_is_off(self, checker) -> None:
        row = {
            "fact_id": "F-029-001",
            "source_id": "S-029",  # 자격이 **있는** 자료라도 스위치가 꺼져 있으면 막힌다
            "publisher": "FDA",
            "doc_type": "toxicity_food",
            "species": "dog",
            "substance": "포도",
            "locator": "본문",
            "quote": "Grapes and raisins can cause kidney failure",
        }
        issues = checker.check_row(row, "t:1", checker.route1_eligible())
        assert any(i.level == "ERROR" and "D-45" in i.message for i in issues)

    def test_missing_ledger_fails_closed(self, checker, monkeypatch, tmp_path) -> None:
        """원장을 못 읽으면 **자격 없음**으로 실패한다. 열어주지 않는다."""
        monkeypatch.setattr(checker, "MANIFEST_DIR", tmp_path)
        assert checker.route1_eligible() == set()
