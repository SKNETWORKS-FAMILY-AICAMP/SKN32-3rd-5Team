"""API 계약 테스트.

여기서 검증하는 것은 "엔드포인트가 200을 준다"가 아니라
**02 §9 정책이 계약 수준에서 깨질 수 없는가**다.
"""

from __future__ import annotations

import pytest
from fastapi.testclient import TestClient
from pydantic import ValidationError

from pettriage.app.contracts import DISCLAIMER, AskResponse, Citation, TriageResult


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
    # 폴백이 일어났는지 화면·스크립트가 알아챌 수 있어야 한다 (04 §8)
    assert d["engine_configured"] == "stub"
    assert d["profile"] == "default"


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
    """되묻기 → 슬롯 충족 → 답변. 세션이 슬롯을 이어받는가.

    질문이 `다크초콜릿` 인 이유 — 초콜릿 역치는 **테오브로민 기준**이라
    종류를 모르면 환산이 불가능하고, 엔진이 한 번 더 되묻는다
    (`test_초콜릿_종류를_모르면_되묻는다` 참조).
    """
    first = client.post(
        "/api/ask", json={"question": "다크초콜릿을 먹었어요", "species": "dog"}
    ).json()
    assert first["status"] == "clarify"
    assert set(first["clarify"]["missing"]) == {"weight_kg", "amount_g"}

    second = client.post(
        "/api/ask",
        json={
            "question": "다크초콜릿을 먹었어요",
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
    """조류 전용 필드는 종이 맞을 때만 보관한다 (최소 수집 · D-36)."""
    client.post(
        "/api/records",
        json={
            "pet_id": "p1",
            "species": "dog",
            "recorded_at": "2026-07-31T09:00:00",
            "droppings": "노란색",
        },
    )
    rows = client.get("/api/report", params={"pet_id": "p1"}).json()["timeline"]
    assert rows and "droppings" not in rows[0]


def test_bird_field_kept_for_birds(client: TestClient):
    client.post(
        "/api/records",
        json={
            "pet_id": "b1",
            "species": "bird",
            "recorded_at": "2026-07-31T09:00:00",
            "droppings": "녹색",
        },
    )
    rows = client.get("/api/report", params={"pet_id": "b1"}).json()["timeline"]
    assert rows[0]["droppings"] == "녹색"


def test_report_applies_period_filter(client: TestClient):
    """받기만 하고 안 쓰면 화면의 기간 선택이 거짓말이 된다."""
    for day in ("2026-07-01", "2026-07-15", "2026-07-30"):
        client.post(
            "/api/records",
            json={"pet_id": "p2", "species": "dog", "recorded_at": f"{day}T09:00:00"},
        )
    rows = client.get(
        "/api/report",
        params={"pet_id": "p2", "period_from": "2026-07-10", "period_to": "2026-07-20"},
    ).json()["timeline"]
    assert [r["recorded_at"][:10] for r in rows] == ["2026-07-15"]


def test_records_do_not_leak_across_app_instances(client: TestClient):
    """저장소가 모듈 전역이면 앱을 새로 만들어도 남의 기록이 보인다."""
    from pettriage.app.main import create_app

    client.post(
        "/api/records",
        json={"pet_id": "secret", "species": "dog", "recorded_at": "2026-07-31T09:00:00"},
    )
    from pettriage.app import deps

    deps.reset_state()
    other = TestClient(create_app())
    assert other.get("/api/report", params={"pet_id": "secret"}).json()["timeline"] == []


def test_frontend_is_served(client: TestClient):
    r = client.get("/")
    assert r.status_code == 200
    assert "PetTriage" in r.text


def test_full_text_carries_escalation_conditions(client: TestClient):
    """`answer` 만 읽는 소비자가 상승 조건을 빠뜨리지 않게 한다."""
    d = client.post(
        "/api/ask",
        json={
            "question": "다크초콜릿을 먹었어요",
            "species": "dog",
            "weight_kg": 5.0,
            "amount_g": 30,
        },
    ).json()
    assert "발작" not in d["answer"]
    assert "발작" in d["full_text"]
    assert "수의학적 진단이 아닙니다" in d["full_text"]


# ─────────────────────────────────────────────────────────────
# 회귀 — 감사에서 나온 결함들이 다시 들어오지 않게 한다
# ─────────────────────────────────────────────────────────────
def test_invariants_survive_assignment():
    """생성 시점에만 검증하면 대입 한 줄로 불변식이 뚫린다."""
    r = AskResponse(
        status="refused", session_id="s", refusal={"reason": "근거없음", "message": "없음"}
    )
    with pytest.raises(ValidationError):
        r.status = "answered"  # 근거·판정 없이 answered 로 바꿀 수 없다

    c = Citation(source_id="S-042", publisher="AAFCO", route="원문적재", quote="원문")
    with pytest.raises(ValidationError):
        c.route = "사실추출"  # 인용문을 실은 채 경로만 바꿀 수 없다

    t = TriageResult(level=2, name="VISIT_SOON", badge="내원", message="오늘 중")
    with pytest.raises(ValidationError):
        t.level = 1  # 상승 조건 없이 MONITOR 로 낮출 수 없다


def test_clarify_budget_resets_on_progress(client: TestClient):
    """슬롯을 하나씩 채우는 협조적 사용자가 상한에 걸려 거절되면 안 된다."""
    sid = None
    seq = []
    for body in (
        {"question": "다크초콜릿을 먹었어요"},
        {"question": "다크초콜릿을 먹었어요", "species": "dog"},
        {"question": "다크초콜릿을 먹었어요", "weight_kg": 5.0},
        {"question": "다크초콜릿을 먹었어요", "amount_g": 30},
    ):
        if sid:
            body["session_id"] = sid
        d = client.post("/api/ask", json=body).json()
        sid = d["session_id"]
        seq.append(d["status"])
    assert seq == ["clarify", "clarify", "clarify", "answered"], seq


def test_bird_is_not_asked_for_weight(client: TestClient):
    """조류는 체중당 임계치가 0건이라 수치를 요구하지 않는다 (D-09 개정).

    요구하면 근거에 없는 값을 모델이 지어낸다.
    """
    d = client.post("/api/ask", json={"question": "아보카도를 먹었어요", "species": "bird"}).json()
    assert d["status"] == "answered"


def test_eval_profile_disables_clarify(monkeypatch: pytest.MonkeyPatch):
    """평가 중 되묻기가 섞이면 과소평가율 분모가 흔들린다 (04 §4.1)."""
    from pettriage import config as config_mod
    from pettriage.app import deps
    from pettriage.app.main import create_app

    monkeypatch.setenv("PETTRIAGE_PROFILE", "eval")
    monkeypatch.setenv("PETTRIAGE_ALLOW_ENGINE_FALLBACK", "1")
    config_mod.reset_caches()
    deps.reset_state()

    c = TestClient(create_app())
    d = c.post("/api/ask", json={"question": "초콜릿을 먹었어요"}).json()
    assert d["status"] == "refused"
    assert d["refusal"]["reason"] == "되묻기상한"


def test_graph_engine_builds_when_ready(monkeypatch: pytest.MonkeyPatch):
    """GraphEngine 구현 완료 후 engine:graph 로 정상 생성되는지 확인."""
    from pettriage import config as config_mod
    from pettriage.app import deps

    monkeypatch.setenv("PETTRIAGE_PROFILE", "eval")
    monkeypatch.delenv("PETTRIAGE_ALLOW_ENGINE_FALLBACK", raising=False)
    config_mod.reset_caches()
    deps.reset_state()

    engine = deps.get_engine()
    assert engine.name == "graph"


def test_response_contract_violation_becomes_refusal(client: TestClient):
    """계약 위반은 500(장애 화면)이 아니라 거절 화면으로 내려간다."""
    from pettriage.app.deps import get_engine

    class Liar:
        name = "liar"

        def ask(self, req, session):
            # 근거 없는 answered — 계약 위반. 직렬화 단계에서 걸린다.
            return AskResponse.model_construct(
                status="answered", session_id="x", answer="괜찮습니다", citations=[]
            )

    client.app.dependency_overrides[get_engine] = lambda: Liar()
    try:
        r = client.post("/api/ask", json={"question": "초콜릿", "species": "dog"})
        assert r.status_code == 200
        assert r.json()["status"] == "refused"
        assert "수의학적 진단이 아닙니다" in r.json()["disclaimer"]
    finally:
        client.app.dependency_overrides.clear()


# ─────────────────────────────────────────────────────────────
# 2026-08-02 검토 회귀 — 안전 불변식이 "불릴 때만" 지켜지던 것들
# ─────────────────────────────────────────────────────────────
def test_초콜릿_종류를_모르면_되묻는다(client: TestClient):
    """초콜릿 역치 `20 mg/kg` 은 **테오브로민** 기준이지 초콜릿 질량이 아니다.

    함량은 밀크 2 · 다크 5 · 베이킹 14 mg/g 로 **7배** 다르다 (S-034).
    종류를 모르는 채 아무 값이나 고르면 그 순간부터 **우리가 만든 숫자가
    등급을 판정한다** — D-51 이 막은 것과 같은 종류다. 그래서 되묻는다.
    골든셋 G-047 이 기대하는 동작이기도 하다.
    """
    d = client.post(
        "/api/ask",
        json={"question": "초콜릿을 먹었어요", "species": "dog", "weight_kg": 4.0, "amount_g": 20},
    ).json()
    assert d["status"] == "clarify"
    assert "다크" in d["clarify"]["question"] and "밀크" in d["clarify"]["question"]


def test_섭취량이_등급을_바꾼다(client: TestClient):
    """되물어 받은 값을 **실제로 쓴다** (D-50).

    예전에는 `_STUB_RULES` 의 상수를 그대로 게이트에 넣어서, 4kg 개가
    0.1g 을 먹든 500g 을 먹든 답이 `CALL_NOW` 로 같았다. *"체중당 섭취량으로
    판단합니다"* 라고 되물어 놓고 그 값을 버리고 있었다 (2026-08-02 검토).
    """
    levels = []
    for amount in (1, 20, 200):
        d = client.post(
            "/api/ask",
            json={
                "question": "다크초콜릿을 먹었어요",
                "species": "dog",
                "weight_kg": 4.0,
                "amount_g": amount,
            },
        ).json()
        levels.append(d["triage"]["level"])
    # 1g→1.2 · 20g→25 · 200g→250 mg/kg. 역치 20(임상징후)·40-50(중증) 을 차례로 넘는다
    assert levels == [1, 3, 4], levels


def test_답변에_계산_근거가_실린다(client: TestClient):
    """판정에 쓴 수치를 보여주지 않으면 보호자가 검산할 수 없다 (04 §8)."""
    d = client.post(
        "/api/ask",
        json={
            "question": "다크초콜릿을 먹었어요",
            "species": "dog",
            "weight_kg": 4.0,
            "amount_g": 20,
        },
    ).json()
    assert "다크 5mg/g" in d["answer"]
    assert "mg/kg" in d["answer"]


def test_연락처는_엔진을_거치는_모든_응답에서_제거된다(client: TestClient):
    """D-47 이 **주입 지점**에서 강제된다.

    예전에는 `scrub_contacts` 의 유일한 호출부가 `graph.nodes.finalize` 였고,
    `GraphEngine` 은 `NODES_IMPLEMENTED=False` 라 만들어지지도 않았다 —
    **그때까지 `/api/ask` 응답은 어떤 연락처 필터도 통과하지 않았다.**
    """
    from pettriage.app import deps
    from pettriage.app.safety_engine import SafetyEngine

    class Leaky:
        name = "leaky"

        def ask(self, req, session):  # noqa: ANN001
            return AskResponse.model_construct(
                status="answered",
                session_id=session.session_id,
                answer="다크초콜릿은 5 mg/g 입니다.\nASPCA APCC 888-426-4435 로 연락하세요.",
                triage=TriageResult(
                    level=3,
                    message="지금 연락하세요",
                    escalation_conditions=["구토", "Pet Poison Helpline (855)-764-7661 로 연락"],
                ),
                citations=[Citation(source_id="S-034", publisher="Veterinary Sciences")],
                clarify=None,
                refusal=None,
                disclaimer=DISCLAIMER,
            )

    deps.set_engine(Leaky())
    try:
        assert isinstance(deps.get_engine(), SafetyEngine)  # 주입이 강제한다
        d = client.post("/api/ask", json={"question": "다크초콜릿", "species": "dog"}).json()
    finally:
        deps.set_engine(None)

    assert "888-426-4435" not in d["full_text"]
    assert "764-7661" not in d["full_text"]
    assert "5 mg/g" in d["answer"]  # 안전한 정보는 살아남는다
    assert d["triage"]["escalation_conditions"] == ["구토"]  # 항목 단위로만 뺀다


def test_밝히지_않은_추정은_응답을_만들_수_없다():
    """추정 물질로 답하면서 그 가정을 숨길 수 없다 (D-59).

    물질을 말하지 않는 질의(*"앵무새 앞에서 프라이팬을 태웠어요"*)에
    후보 중 최고 등급으로 답하는 것은 D-13(과소평가 최우선)에 따른 선택이다.
    그 선택이 정직하려면 **무엇을 가정했는지가 문장에 있어야** 한다.

    문장 생성에 맡기면 LLM 이 한 줄을 빠뜨리는 순간 추측이 단정이 된다 —
    그것이 곧 환각이다. 그래서 계약이 강제한다 (D-54 와 같은 방식).

    ⚠️ **이 테스트는 원래 `"PTFE(테플론) 과열 흄"` 을 썼다.** 코퍼스에 없는 이름이다 —
    코퍼스는 `PTFE(테플론) 과열 흄` 으로 적는다. 폐쇄 목록 계약(D-59 ①)을 넣자
    여기서 걸렸고, **테스트가 만들어 낸 이름으로 검증하고 있었다는 뜻**이다.
    이름을 코퍼스의 것으로 바꿨다 (D-57: 틀린 동작을 고정한 테스트는 함께 뒤집고
    이유를 남긴다).
    """
    cit = [Citation(source_id="S-071", publisher="AAV 미국조류수의사회")]
    tri = TriageResult(level=4, message="지금 바로 동물병원으로 가세요")

    # 가정을 밝히지 않으면 만들어지지 않는다
    with pytest.raises(ValidationError, match="밝히지 않은 추정은 환각"):
        AskResponse(
            status="answered",
            session_id="s",
            answer="앵무새를 즉시 환기된 곳으로 옮기고 병원으로 가세요.",
            triage=tri,
            citations=cit,
            assumed_substance="PTFE(테플론) 과열 흄",
        )

    # 밝히면 통과한다
    r = AskResponse(
        status="answered",
        session_id="s",
        answer="PTFE(테플론) 과열 흄으로 보고 안내드립니다. 즉시 병원으로 가세요.",
        triage=tri,
        citations=cit,
        assumed_substance="PTFE(테플론) 과열 흄",
    )
    assert r.assumed_substance == "PTFE(테플론) 과열 흄"
    assert "PTFE(테플론) 과열 흄" in r.full_text

    # 추정이 없으면 검사 자체가 돌지 않는다 (기존 응답에 영향 0)
    assert (
        AskResponse(
            status="answered", session_id="s", answer="답변", triage=tri, citations=cit
        ).assumed_substance
        is None
    )
