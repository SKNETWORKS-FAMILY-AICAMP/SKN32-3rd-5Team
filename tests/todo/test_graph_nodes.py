"""WS2 — 그래프 노드 구현 대기 테스트.

각 테스트가 노드 하나의 **완료 기준**이다. 초록이 되면 그 노드는 끝난 것이다.
조건은 대부분 06 의 설계 결정에서 나왔으므로, 조건을 바꾸려면 먼저 확인한다.

    pytest -m todo
"""

from __future__ import annotations

import pytest

from pettriage.graph import initial_state
from pettriage.retrieval import HashEmbedder, InMemoryStore
from pettriage.schemas import Chunk

pytestmark = pytest.mark.todo


@pytest.fixture
def store() -> InMemoryStore:
    s = InMemoryStore(embedder=HashEmbedder())
    s.add(
        [
            Chunk(
                chunk_id="c-dog-choco",
                text=(
                    "개에게 초콜릿은 급여 금지로 분류된다. "
                    "20mg/kg 이상 섭취 시 경증 임상징후가 보고되었다."
                ),
                source_id="S-034",
                species="dog",
                doc_type="toxicity_food",
                substance="초콜릿",
            ),
            Chunk(
                chunk_id="c-bird-avo",
                text="앵무새에게 아보카도는 급여 금지로 분류된다.",
                source_id="S-005",
                species="bird",
                doc_type="toxicity_food",
                substance="아보카도",
            ),
        ]
    )
    return s


class TestClassify:
    def test_returns_allowed_label_only(self):
        """허용목록 밖 라벨은 unknown 으로 떨어진다 (05 §4 ①)."""
        from pettriage.graph.nodes import ALLOWED_INTENTS, classify_intent

        out = classify_intent(initial_state("강아지가 초콜릿을 먹었어요", "s1"))
        assert out["intent"] in (*ALLOWED_INTENTS, "unknown")

    def test_intoxication_is_detected(self):
        from pettriage.graph.nodes import classify_intent

        out = classify_intent(initial_state("강아지가 초콜릿을 먹었어요", "s1"))
        assert out["intent"] == "intoxication"


class TestSlots:
    def test_species_is_required_before_search(self):
        """종이 없으면 검색으로 넘어가지 않는다 (D-10)."""
        from pettriage.graph.nodes import extract_slots

        out = extract_slots(initial_state("초콜릿을 먹었어요", "s1", intent="intoxication"))
        assert "species" in out["missing_slots"]

    def test_absent_value_is_not_invented(self):
        """발화에 없는 값을 채우면 그게 곧 환각이다."""
        from pettriage.graph.nodes import extract_slots

        out = extract_slots(
            initial_state("강아지가 초콜릿을 먹었어요", "s1", intent="intoxication")
        )
        assert "weight_kg" not in out["slots"], "체중은 발화에 없다"

    def test_breed_does_not_imply_species(self):
        """품종명이나 이름에서 종을 추측하지 않는다."""
        from pettriage.graph.nodes import extract_slots

        out = extract_slots(
            initial_state("우리 코코가 초콜릿을 먹었어요", "s1", intent="intoxication")
        )
        assert out["slots"].get("species") is None

    def test_clarify_stops_at_configured_limit(self):
        """되묻기 상한은 설정값이다. 넘으면 거절로 간다 (02 §9)."""
        from pettriage.graph.nodes import ask_clarify

        st = initial_state("초콜릿", "s1", missing_slots=["species"], clarify_turns=2)
        out = ask_clarify(st)
        assert out.get("status") == "refused"
        assert out.get("refusal_reason") == "되묻기상한"


class TestFilter:
    def test_species_filter_is_always_present(self):
        """종 필터가 D-10 의 구현부다. 빠지면 조류에 개 기준이 적용된다."""
        from pettriage.graph.nodes import build_filter

        out = build_filter(initial_state("q", "s1", slots={"species": "dog"}))
        assert "species" in out["where"]

    def test_mammal_and_all_are_included_for_cat(self):
        """고양이 자체 자료는 2단계뿐이라 mammal·all 을 함께 봐야 한다 (D-39)."""
        from pettriage.graph.nodes import build_filter

        out = build_filter(initial_state("q", "s1", slots={"species": "cat"}))
        target = out["where"]["species"]
        assert set(target) >= {"cat", "mammal", "all"}


class TestRetrieve:
    def test_species_filter_excludes_other_species(self, store):
        from pettriage.graph.nodes import retrieve

        st = initial_state("초콜릿", "s1", where={"species": "bird"})
        out = retrieve(st, store=store)
        assert all(h.chunk.species == "bird" for h in out["hits"])

    def test_below_threshold_returns_empty(self, store):
        """임계 미만은 잘라낸다. 빈 결과는 실패가 아니라 거절 신호다 (02 §8.3)."""
        from pettriage.graph.nodes import retrieve

        st = initial_state("전혀 무관한 질문", "s1", where={"species": "dog"})
        out = retrieve(st, store=store)
        assert out["hits"] == []


class TestCompute:
    def test_dose_per_kg_is_computed_by_code(self):
        """수치는 벡터 검색이 아니라 코드가 계산한다 (D-16)."""
        from pettriage.graph.nodes import compute_metrics

        st = initial_state("q", "s1", slots={"species": "dog", "weight_kg": 5.0, "amount_g": 30.0})
        out = compute_metrics(st)
        assert out["computed"]["dose_per_kg"] == pytest.approx(6.0)


class TestTriage:
    def test_llm_cannot_lower_rule_level(self):
        """🔒 하향 금지 — LLM 이 낮춰도 최종은 규칙 등급이다 (D-09)."""
        from pettriage.graph.nodes import decide_triage

        st = initial_state("q", "s1", rule_level=4, llm_level=1)
        out = decide_triage(st)
        assert out["triage_level"] == 4

    def test_upgrade_is_accepted(self):
        from pettriage.graph.nodes import decide_triage

        st = initial_state("q", "s1", rule_level=1, llm_level=4, escalation_conditions=["구토"])
        out = decide_triage(st)
        assert out["triage_level"] == 4

    def test_monitor_without_conditions_is_refused(self):
        """조건 없는 '관찰'은 과소평가로 채점된다 (D-39 · 04 §4.1.0)."""
        from pettriage.graph.nodes import decide_triage

        st = initial_state("q", "s1", rule_level=1, llm_level=None, escalation_conditions=[])
        out = decide_triage(st)
        assert out.get("status") == "refused" or out.get("escalation_conditions")

    def test_bird_is_not_asked_for_dose(self):
        """조류는 체중당 임계치가 0건이다. 요구하면 모델이 지어낸다 (D-09 개정)."""
        from pettriage.graph.nodes import decide_triage

        st = initial_state(
            "q", "s1", slots={"species": "bird", "substance": "아보카도"}, rule_level=4
        )
        out = decide_triage(st)
        assert "weight_kg" not in out.get("missing_slots", [])


class TestGenerate:
    def test_draft_does_not_add_numbers_absent_from_context(self):
        """원문에 없는 수치를 만들지 않는다."""
        from pettriage.graph.nodes import generate_draft

        st = initial_state(
            "초콜릿 얼마나 위험한가요", "s1", context="개에게 초콜릿은 급여 금지로 분류된다."
        )
        out = generate_draft(st)
        assert "mg/kg" not in out["draft"], "근거에 없는 수치를 만들었다"

    def test_simplify_does_not_soften_risk(self):
        """위험도를 낮추는 완곡 표현을 쓰지 않는다."""
        from pettriage.graph.nodes import simplify

        st = initial_state("q", "s1", draft="지금 바로 동물병원으로 가세요.", triage_level=4)
        out = simplify(st)
        assert "괜찮" not in out["answer"] and "지켜보" not in out["answer"]


class TestVerify:
    def test_unsupported_sentence_is_flagged(self):
        """애매하면 근거없음 쪽으로 판정한다 — 놓친 환각보다 낫다."""
        from pettriage.graph.nodes import verify_grounding

        st = initial_state(
            "q",
            "s1",
            draft="초콜릿은 위험하다. 그리고 고양이는 3일이면 회복된다.",
            context="개에게 초콜릿은 급여 금지로 분류된다.",
        )
        out = verify_grounding(st)
        assert any(v["verdict"] in ("근거없음", "모순") for v in out["verdicts"])

    def test_retry_is_capped(self):
        """재검색은 1회까지. 무한 루프를 막는다 (02 §2)."""
        from pettriage.graph.nodes import MAX_RETRY, verify_grounding

        st = initial_state("q", "s1", draft="근거 없는 문장.", context="", retry_count=MAX_RETRY)
        out = verify_grounding(st)
        assert out.get("status") == "refused"
        assert out.get("refusal_reason") == "검증실패"
