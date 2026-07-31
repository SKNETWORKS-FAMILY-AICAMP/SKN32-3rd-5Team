"""API 계약 테스트.

여기서 검증하는 것은 "엔드포인트가 200을 준다"가 아니라
**02 §9 정책이 계약 수준에서 깨질 수 없는가**다.
"""

from __future__ import annotations

import pytest
from fastapi.testclient import TestClient
from pydantic import ValidationError

from pettriage.app.contracts import AskResponse, Citation, TriageResult
from pettriage.app.main import create_app


@pytest.fixture
def client() -> TestClient:
    return TestClient(create_app())


# ─────────────────────────────────────────────────────────────
# 계약 불변식 — 스키마가 정책을 강제하는가
# ─────────────────────────────────────────────────────────────
def test_answered_without_citation_is_impossible():
    """근거 없는 답변은 **객체 생성 자체가 안 된다.** 이 프로젝트의 존재 이유."""
    with pytest.raises(ValidationError):
        AskResponse(
            status="answered",
            session_id="s",
            answer="괜찮습니다",
            triage=TriageResult(level=4, name="EMERGENCY", badge="응급", message="가세요"),
            citations=[],
        )


def test_refused_cannot_carry_answer():
    with pytest.raises(ValidationError):
        AskResponse(
            status="refused",
            session_id="s",
            answer="그래도 알려드리자면",
            refusal={"reason": "근거없음", "message": "없습니다"},
        )


def test_clarify_cannot_carry_answer():
    with pytest.raises(ValidationError):
        AskResponse(
            status="clarify",
            session_id="s",
            answer="아마도 응급입니다",
            clarify={"missing": ["species"], "question": "어떤 동물인가요?", "turn": 1},
        )


def test_monitor_without_conditions_rejected_at_contract():
    """게이트(gate.py)와 계약(contracts.py) 양쪽에서 막는다 — D-39."""
    with pytest.raises(ValidationError):
        TriageResult(level=1, name="MONITOR", badge="관찰", message="지켜보세요")


def test_route2_citation_cannot_have_quote():
    """경로 ② 자료에 원문 인용을 실으면 D-37 위반 — 계약에서 막는다."""
    with pytest.raises(ValidationError):
        Citation(source_id="S-034", publisher="X", route="사실추출", quote="원문 문장")

    ok = Citation(source_id="S-042", publisher="AAFCO", route="원문적재", quote="원문 문장")
    assert ok.quote


def test_disclaimer_always_present():
    r = AskResponse(
        status="refused", session_id="s", refusal={"reason": "범위밖", "message": "밖입니다"}
    )
    assert "수의학적 진단이 아닙니다" in r.disclaimer


# ─────────────────────────────────────────────────────────────
# 엔드포인트 — 02 §9 분기가 실제로 도는가
# ─────────────────────────────────────────────────────────────
def test_health(client: TestClient):
    d = client.get("/api/health").json()
    assert d["status"] == "ok" and d["engine"] == "stub"


def test_species_missing_forces_clarify(client: TestClient):
    d = client.post("/api/ask", json={"question": "초콜릿을 먹었어요"}).json()
    assert d["status"] == "clarify"
    assert d["clarify"]["missing"] == ["species"]
    assert d["answer"] is None


def test_no_evidence_is_refusal_not_error(client: TestClient):
    """거절은 200이다. 4xx로 만들면 프론트가 장애 화면으로 그린다."""
    r = client.post("/api/ask", json={"question": "고양이 이름 지어줘", "species": "cat"})
    assert r.status_code == 200
    assert r.json()["status"] == "refused"
    assert r.json()["refusal"]["reason"] == "근거없음"


def test_slot_clarify_then_answer(client: TestClient):
    """되묻기 → 슬롯 충족 → 답변. 세션이 슬롯을 이어받는가."""
    first = client.post("/api/ask", json={"question": "초콜릿을 먹었어요", "species": "dog"}).json()
    assert first["status"] == "clarify"
    assert set(first["clarify"]["missing"]) == {"weight_kg", "amount_g"}

    second = client.post(
        "/api/ask",
        json={
            "question": "초콜릿을 먹었어요",
            "session_id": first["session_id"],
            "weight_kg": 5.0,
            "amount_g": 30,
        },
    ).json()
    # 두 번째 요청에 species 를 안 실었는데도 세션이 기억한다
    assert second["status"] == "answered"
    assert second["triage"]["badge"] == "전화"
    assert second["citations"][0]["source_id"] == "S-034"
    assert second["citations"][0]["quote"] is None  # 경로 ②


def test_clarify_limit_becomes_refusal(client: TestClient):
    """되묻기 상한 2회 초과 → 거절 (02 §9)."""
    sid = None
    statuses = []
    for _ in range(3):
        body = {"question": "초콜릿을 먹었어요", "species": "dog"}
        if sid:
            body["session_id"] = sid
        d = client.post("/api/ask", json=body).json()
        sid = d["session_id"]
        statuses.append(d["status"])
    assert statuses == ["clarify", "clarify", "refused"]


def test_species_mismatch_refuses(client: TestClient):
    """개 자료를 앵무새 질문에 쓰지 않는다."""
    d = client.post("/api/ask", json={"question": "포도를 먹었어요", "species": "bird"}).json()
    assert d["status"] == "refused"


def test_bird_path_answers(client: TestClient):
    d = client.post("/api/ask", json={"question": "아보카도를 먹었어요", "species": "bird"}).json()
    assert d["status"] == "answered"
    assert d["triage"]["level"] == 4


def test_engine_failure_degrades_to_refusal(client: TestClient):
    """엔진이 터져도 단정적인 답을 흘리지 않는다."""
    from pettriage.app.deps import get_engine

    class Boom:
        name = "boom"

        def ask(self, req, session):
            raise RuntimeError("kaboom")

    app = client.app
    app.dependency_overrides[get_engine] = lambda: Boom()
    try:
        r = client.post("/api/ask", json={"question": "초콜릿", "species": "dog"})
        assert r.status_code == 200
        assert r.json()["status"] == "refused"
    finally:
        app.dependency_overrides.clear()


def test_triage_levels_expose_evidence(client: TestClient):
    """등급 표현의 단일 출처. 프론트가 하드코딩하지 않게 한다."""
    d = client.get("/api/triage-levels").json()
    assert [x["level"] for x in d["levels"]] == [4, 3, 2, 1]
    assert all(x["evidence"]["source_id"] for x in d["levels"])
    assert d["bird_feeding_levels"] == [2, 3]  # 조류는 SAFE 미노출 (D-39)


def test_bird_only_field_dropped_for_mammals(client: TestClient):
    """조류 전용 필드는 종이 맞을 때만 보관한다."""
    from pettriage.app.routes.records import _RECORDS

    client.post(
        "/api/records",
        json={
            "pet_id": "p1",
            "species": "dog",
            "recorded_at": "2026-07-31T09:00:00",
            "droppings": "노란색",
        },
    )
    assert "droppings" not in _RECORDS["p1"][0]


def test_frontend_is_served(client: TestClient):
    r = client.get("/")
    assert r.status_code == 200
    assert "PetTriage" in r.text


def test_full_text_carries_escalation_conditions(client: TestClient):
    """`answer` 만 읽는 소비자가 상승 조건을 빠뜨리지 않게 한다."""
    d = client.post(
        "/api/ask",
        json={
            "question": "초콜릿을 먹었어요",
            "species": "dog",
            "weight_kg": 5.0,
            "amount_g": 30,
        },
    ).json()
    assert "발작" not in d["answer"]
    assert "발작" in d["full_text"]
    assert "수의학적 진단이 아닙니다" in d["full_text"]
