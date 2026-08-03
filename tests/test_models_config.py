"""설정·태스크·혼합 테스트 — GPU 없이 도는 것만.

torch 를 임포트하지 않는다. CI가 최소 의존성으로 돌아야 한다.
"""

from __future__ import annotations

import pytest

from pettriage.app.contracts import MAX_CLARIFY_TURNS
from pettriage.config import load_config
from pettriage.models.datasets.mixer import MixTargetUnreachable, check_leakage, mix, task_counts
from pettriage.models.datasets.schema import TrainSample
from pettriage.models.prompts import build_messages, build_sample, system_prompt
from pettriage.models.tasks import DEFAULT_TASKS, SPECS, Task


# ── 설정 ─────────────────────────────────────────────────────
def test_default_profile_loads():
    cfg = load_config("default")
    assert cfg.model.base_id == "Qwen/Qwen3-4B"
    assert cfg.model.load_in_4bit is True
    assert cfg.serve.engine == "stub"


def test_profile_overrides_default():
    assert load_config("eval").serve.engine == "graph"
    # 덮지 않은 값은 default 를 유지한다
    assert load_config("eval").model.base_id == "Qwen/Qwen3-4B"


def test_eval_profile_does_not_disable_clarify():
    """**평가에서도 되묻는다** (D-66 · D-57 로 뒤집은 값).

    예전에는 `0` 이었다. 그러면 되묻기 기대 15건이 **구조적으로 통과 불가**가 되고,
    *"결측을 알아채고 멈췄다"* 는 핵심 안전 동작이 측정에서 사라진다.
    근거였던 *"분모가 흔들린다"* 는 성립하지 않았다 — `clarify` 도 `refused` 도
    `triage` 가 없어 등급 분모에서 **똑같이** 빠진다.
    """
    assert load_config("eval").triage.max_clarify_turns == MAX_CLARIFY_TURNS


def test_env_overrides_yaml(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setenv("PETTRIAGE__RETRIEVAL__TOP_K", "9")
    assert load_config("default").retrieval.top_k == 9


def test_dotenv_overrides_yaml(tmp_path, monkeypatch: pytest.MonkeyPatch):
    """`.env` 의 `PETTRIAGE__…` 도 설정을 덮는다.

    ⚠️ **한 번도 동작한 적이 없었다** (2026-08-02 확인). `.env.example` 은
    처음부터 `PETTRIAGE__SERVE__ENGINE=graph` 를 예시로 안내했는데,
    `_env_overrides` 는 `os.environ` 만 봤고 pydantic-settings 는 `.env` 를
    `os.environ` 에 내보내지 않는다. `extra="ignore"` 라 **경고도 없이 버려졌다.**

    안내가 거짓이면 사람은 설정을 바꿨다고 믿고 **안 바뀐 조건으로 측정한다.**
    04 §8 재현성이 무너지는 방식이 오타(`UnknownConfigKey`)와 같다.
    """
    monkeypatch.chdir(tmp_path)
    (tmp_path / ".env").write_text("PETTRIAGE__MODEL__PROVIDER=none\n", encoding="utf-8")
    assert load_config("default").model.provider == "none"


def test_shell_beats_dotenv(tmp_path, monkeypatch: pytest.MonkeyPatch):
    """`.env` 는 그 컴퓨터의 상시 설정, 셸 변수는 *"이번 한 번"* 이다.

    한 번짜리가 이겨야 04 §3 비교군을 한 줄로 갈아끼울 수 있다.
    """
    monkeypatch.chdir(tmp_path)
    (tmp_path / ".env").write_text("PETTRIAGE__MODEL__PROVIDER=none\n", encoding="utf-8")
    monkeypatch.setenv("PETTRIAGE__MODEL__PROVIDER", "qwen")
    assert load_config("default").model.provider == "qwen"


def test_serving_provider_reaches_the_client(monkeypatch: pytest.MonkeyPatch):
    """설정이 **서빙 클라이언트까지** 닿는다 (D-65).

    2026-08-02 까지 노드 4곳이 `APIClient()` 를 직접 만들어서 `model.*` 절이
    서빙에 안 닿았고, `LocalQwenClient` 는 아무도 생성하지 않았다.
    04 §3 비교표의 C·D 열을 **채울 방법 자체가 없었다.**
    """
    from pettriage.config import reset_caches
    from pettriage.models.serving.factory import get_client, reset_client

    monkeypatch.setenv("PETTRIAGE__MODEL__PROVIDER", "none")
    reset_caches()
    reset_client()
    assert get_client() is None, "provider=none 인데 클라이언트가 만들어졌다"

    monkeypatch.setenv("PETTRIAGE__MODEL__PROVIDER", "qwen")
    reset_caches()
    reset_client()
    c = get_client()
    assert c is not None and c.name.startswith("qwen:"), c
    # revision 핀이 서빙에 걸리는지 — 학습 경로에만 걸려 있었다
    assert c._revision == load_config("default").model.revision  # noqa: SLF001
    # 프로토콜이 요구하지 않아 구현이 빠져 있었다
    assert hasattr(c, "run_raw"), "generate.py 가 부르는 순간 AttributeError 다"

    reset_caches()
    reset_client()


def test_clarify_limit_matches_contract():
    """설정과 계약이 어긋나면 되묻기 상한이 두 값이 된다 (02 §9)."""
    assert load_config("default").triage.max_clarify_turns == MAX_CLARIFY_TURNS


def test_bird_excluded_from_quantitative_species():
    """조류는 체중당 임계치가 0건이다 (D-09 개정). 정량 슬롯을 요구하면 환각이 난다."""
    assert "bird" not in load_config("default").triage.quantitative_species


def test_score_threshold_is_positive():
    """0이면 유사도 무관하게 답을 만들게 된다 (02 §8.3)."""
    assert load_config("default").retrieval.score_threshold > 0


def test_score_threshold_does_not_reject_grounded_queries():
    """근거가 있는 질의를 거절하는 임계값은 **과소평가를 만든다** (D-13 · D-46).

    실측(`scripts/calibrate_threshold.py`)에서 근거 있는 질의의 최저 점수가 **0.547**
    이었다 — "앵무새 앞에서 프라이팬을 태웠어요" → PTFE. 그 아래로 유지해야 한다.

    **임계값을 올려 거절 정확도를 높이려는 시도를 이 테스트가 막는다.**
    거절은 ① 의도 분류와 ④ 근거 검증이 만든다 (D-46).
    """
    lowest_grounded = 0.547
    assert load_config("default").retrieval.score_threshold < lowest_grounded


def test_config_and_module_default_agree_on_threshold():
    """YAML 과 코드 기본값이 어긋나면 설치 형태에 따라 안전 동작이 달라진다 (D-41)."""
    from pettriage.config import RetrievalConfig

    assert load_config("default").retrieval.score_threshold == RetrievalConfig().score_threshold


# ── 태스크·프롬프트 ──────────────────────────────────────────
def test_every_default_task_has_spec():
    for t in DEFAULT_TASKS:
        assert t in SPECS
        assert SPECS[t].verified_by  # 05 §6 — LLM 지점마다 검증이 붙는다


def test_verify_task_is_weighted_highest():
    """④ 근거 검증이 환각 방지의 핵심이다 (D-05)."""
    mix_cfg = load_config("default").train.task_mix
    # 동률에서도 참인 `== max(...)` 로는 "가장 높다"를 보장하지 못한다.
    # **유일한 최댓값**인지 확인한다 (2026-08-02 정정).
    top = max(mix_cfg.values())
    assert mix_cfg[Task.VERIFY] == top
    assert sum(1 for v in mix_cfg.values() if v == top) == 1, mix_cfg


def test_prompt_forbids_invention():
    for t in DEFAULT_TASKS:
        p = system_prompt(t)
        assert "진단하지 않는다" in p
        assert "추측" in p or "만들어내지 않는다" in p


def test_messages_and_sample_share_prefix():
    """학습과 추론이 같은 프롬프트를 써야 한다 (03 §2 템플릿 통일)."""
    msgs = build_messages(Task.SLOT, "우리 개가 초콜릿을 먹었어요")
    sample = build_sample(Task.SLOT, "우리 개가 초콜릿을 먹었어요", '{"species":"dog"}')
    assert sample[: len(msgs)] == msgs
    assert sample[-1]["role"] == "assistant"


# ── 혼합·누수 ────────────────────────────────────────────────
def _samples(task: Task, n: int, split: str = "train") -> list[TrainSample]:
    return [
        TrainSample(
            sample_id=f"{task}-{i}",
            task=task,
            input=f"{task} 입력 {i}",
            target="t",
            origin="synthetic_profile",
            split=split,  # type: ignore[arg-type]
        )
        for i in range(n)
    ]


def test_mix_respects_ratios():
    pool = _samples(Task.VERIFY, 60) + _samples(Task.CLASSIFY, 60)
    out = mix(pool, {"verify": 0.75, "classify": 0.25}, total=40, seed=1)
    counts = task_counts(out)
    assert counts["verify"] == 30 and counts["classify"] == 10


def test_mix_raises_when_pool_too_small():
    """조용히 적게 넣으면 '비율대로 학습했다'는 보고가 거짓이 된다."""
    pool = _samples(Task.VERIFY, 5)
    with pytest.raises(MixTargetUnreachable):
        mix(pool, {"verify": 1.0}, total=50)


def test_mix_is_deterministic_by_seed():
    pool = _samples(Task.VERIFY, 50)
    a = [s.sample_id for s in mix(pool, {"verify": 1.0}, total=20, seed=7)]
    b = [s.sample_id for s in mix(pool, {"verify": 1.0}, total=20, seed=7)]
    assert a == b


def test_leakage_detected():
    """골든셋이 학습에 섞이면 평가 전체가 무의미해진다 (04 §2.3)."""
    train = _samples(Task.VERIFY, 3)
    dev = _samples(Task.VERIFY, 3, split="dev")
    assert check_leakage(train, dev)  # 입력 문자열이 같으므로 전건 적발


def test_no_leakage_when_inputs_differ():
    train = _samples(Task.VERIFY, 3)
    dev = [
        TrainSample(
            sample_id="d1",
            task=Task.VERIFY,
            input="완전히 다른 입력",
            target="t",
            origin="handwritten",
            split="dev",
        )
    ]
    assert check_leakage(train, dev) == []


# ── 프롬프트와 검증기가 같은 목록을 본다 (D-73) ────────────────
class TestLabelsAreShared:
    """**코드는 아는데 모델은 모르는 목록**을 만들지 않는다.

    2026-08-02 이서은 팀원 발견 — ①분류 프롬프트가 *"허용된 라벨 중 하나만
    출력한다"* 라고만 적고 **그 라벨이 무엇인지는 안 적었다.** 모델은
    `'위험성우려'`·`'high_risk'` 를 냈고 코드가 전부 `unknown` 으로 걸러 거절이 됐다.

    결과가 뒤집혔다 — **키워드 폴백(통과 10%)보다 진짜 LLM(3.3%)이 더 나빴다.**
    """

    def test_classify_prompt_lists_every_allowed_label(self):
        from pettriage.graph.nodes.classify import ALLOWED_INTENTS
        from pettriage.models.prompts import system_prompt

        prompt = system_prompt(Task.CLASSIFY)
        missing = [label for label in ALLOWED_INTENTS if label not in prompt]
        assert not missing, f"검증기는 아는데 프롬프트에 없는 라벨: {missing}"

    def test_validator_and_prompt_share_one_source(self):
        """손으로 두 곳에 적으면 하나만 고치는 일이 반드시 생긴다 (D-22).

        `ALLOWED_INTENTS` 에 `" "` 가 들어가 있던 사고(2026-08-02)도
        두 곳에 적혀 있어서 안 드러났다.
        """
        from pettriage.graph.nodes.classify import ALLOWED_INTENTS

        assert ALLOWED_INTENTS is SPECS[Task.CLASSIFY].labels

    def test_verify_prompt_lists_its_labels(self):
        """④ 근거 검증도 같다 — 라벨이 어긋나면 판정이 통째로 버려진다."""
        from pettriage.models.prompts import system_prompt

        prompt = system_prompt(Task.VERIFY)
        for label in SPECS[Task.VERIFY].labels:
            assert label in prompt, label

    def test_intent_type_covers_the_labels(self):
        """`graph.state.Intent` 가 라벨 + `unknown` 을 덮는다.

        타입과 목록이 어긋나면 정상 라벨이 타입 검사에서 죽는다.
        """
        from typing import get_args

        from pettriage.graph.state import Intent

        allowed = set(get_args(Intent))
        assert set(SPECS[Task.CLASSIFY].labels) <= allowed
        assert "unknown" in allowed, "폴백 라벨이 타입에 없다"

    def test_free_output_tasks_have_no_label_block(self):
        """압축·평이화는 자유 출력이다. 없는 라벨을 프롬프트가 지어내면 안 된다."""
        from pettriage.models.prompts import system_prompt

        for task in (Task.COMPRESS, Task.SIMPLIFY):
            assert SPECS[task].labels == ()
            assert "[라벨]" not in system_prompt(task)


# ── 호출 간격 제한 ───────────────────────────────────────────
class Test호출간격:
    """저등급 API 의 429 를 **부딪히기 전에** 막는다.

    2026-08-02 Gemini 실측 — 재시도 8회를 다 쓰고도 실패했고, 실행이 통째로 멈췄다.
    재시도는 실패한 뒤의 대응이고, 간격 제한은 실패를 안 만드는 쪽이다.
    """

    def test_기본은_꺼져있다(self):
        """기본값이 0이 아니면 **아무도 모르게 평가가 느려진다.**"""
        assert load_config("default").model.min_interval_ms == 0

    def test_0이면_기다리지_않는다(self, monkeypatch):
        import time

        from pettriage.models.serving import client as mod

        monkeypatch.setattr(mod, "_last_call_at", 0.0)
        t0 = time.monotonic()
        mod._throttle()
        assert time.monotonic() - t0 < 0.05

    def test_설정한_간격만큼_기다린다(self, monkeypatch):
        """`min_interval_ms` 를 코드가 **실제로 읽는지** 본다 (D-69).

        설정에 키만 두고 아무도 안 읽던 전례가 있다. 여기서는 시간으로 확인한다.
        """
        import time

        from pettriage.config import get_config, reset_caches
        from pettriage.models.serving import client as mod

        # **가짜 설정을 끼우지 않는다.** 환경변수 → 설정 → 코드까지 사슬을 통째로 태운다.
        # 이 사슬의 마지막 칸이 끊겨 있던 것이 D-69 였다.
        monkeypatch.setenv("PETTRIAGE__MODEL__MIN_INTERVAL_MS", "120")
        reset_caches()
        assert get_config().model.min_interval_ms == 120, "환경변수가 설정에 안 닿는다"

        monkeypatch.setattr(mod, "_last_call_at", 0.0)
        mod._throttle()  # 첫 호출은 기준점만 잡는다
        t0 = time.monotonic()
        mod._throttle()  # 두 번째는 간격을 채운다
        waited = time.monotonic() - t0
        assert waited >= 0.10, f"기다리지 않았다: {waited:.3f}s"

    def test_설정을_못_읽어도_호출은_나간다(self, monkeypatch):
        """간격 제한이 **호출을 막는 새 실패 원인이 되면 안 된다.**"""
        import pettriage.config as cfgmod
        from pettriage.models.serving import client as mod

        def boom():
            raise RuntimeError("설정 없음")

        boom.cache_clear = lambda: None  # conftest 의 캐시 비우기가 이것을 부른다
        monkeypatch.setattr(cfgmod, "get_config", boom)
        monkeypatch.setattr(mod, "_last_call_at", 0.0)
        mod._throttle()  # 예외가 새어 나오지 않는다
