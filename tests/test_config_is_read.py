"""**설정에 키를 두면 코드가 읽어야 한다** (D-40 · D-58 · D-69).

2026-08-02 하루에 같은 사고가 **네 번** 났다.

    langgraph          pyproject 에 선언 · docstring 에 명시 · `import` 0곳     (D-64)
    model.base_id 등   configs 에 있으나 서빙은 `APIClient()` 하드코딩            (D-65)
    model.load_in_4bit configs 에 `true` 인데 `LocalQwenClient` 가 안 읽음         (D-69)
    PETTRIAGE__…       `.env.example` 이 안내하는데 `.env` 를 안 봤음              (D-69)

전부 같은 모양이다 — **적혀 있는데 강제되지 않는다.** 그리고 전부
*"설정을 바꿨다고 믿고 안 바뀐 조건으로 측정"* 하게 만든다. 04 §8 재현성이
무너지는 방식이 오타(`UnknownConfigKey`)와 같다.

**문서로 적어 두는 것으로는 다시 안 생긴다는 보장이 없다.** 여기서 막는다.
"""

from __future__ import annotations

import pathlib
import re

import pytest

from pettriage import config as C

ROOT = pathlib.Path(__file__).resolve().parents[1]

#: **읽히지 않는 것이 맞는 키.** 사유 없이 여기 넣지 않는다 (D-58).
#:
#: *"못 하는 것을 한다고 적지 않는다."* 미구현이면 미구현이라고 적고,
#: 구현되면 이 목록에서 뺀다. 목록이 길어지는 것 자체가 신호다.
UNREAD_BY_DESIGN: dict[str, str] = {
    "RetrievalConfig.rerank": "04 E2 비교군 변수. **리랭커 미구현** — 켜도 아무 일도 안 일어난다",
    "RetrievalConfig.chunk_strategy": "04 E1 비교군 변수. 적재 전략은 build_index.py 가 고정",
    "RetrievalConfig.fixed_chunk_size": "04 E1 비교군 변수. 위와 같다",
    "TriageConfig.llm_fallback": "게이트는 `llm_level` 유무로 이미 분기한다 — 이 스위치는 미배선",
}

_CONFIG_MODELS = (
    C.ModelConfig,
    C.TrainConfig,
    C.RetrievalConfig,
    C.TriageConfig,
    C.ServeConfig,
    C.AuthConfig,
)


@pytest.fixture(scope="module")
def source_text() -> str:
    """`config.py` 를 뺀 전체 소스. 자기 자신을 근거로 삼으면 검사가 무의미하다."""
    parts: list[str] = []
    for base in ("src", "scripts", "eval"):
        for f in (ROOT / base).rglob("*.py"):
            if f.name == "config.py":
                continue
            parts.append(f.read_text(encoding="utf-8"))
    return "\n".join(parts)


def test_every_config_key_is_read(source_text: str):
    """설정 키가 코드 어디에서도 안 읽히면 **그 키는 거짓말이다.**"""
    dead = []
    for m in _CONFIG_MODELS:
        for name in m.model_fields:
            key = f"{m.__name__}.{name}"
            if key in UNREAD_BY_DESIGN:
                continue
            if not re.search(rf"\b{re.escape(name)}\b", source_text):
                dead.append(key)
    assert not dead, (
        f"설정에 있는데 아무도 안 읽는 키: {dead}\n"
        "  구현하거나, 미구현이면 UNREAD_BY_DESIGN 에 **사유와 함께** 적는다 (D-58).\n"
        "  조용히 두면 '설정을 바꿨다고 믿고 안 바뀐 조건으로 측정' 하게 된다."
    )


def test_unread_list_has_no_stale_entries(source_text: str):
    """구현했으면 목록에서 빼야 한다. **면제가 남아 있으면 검사가 무뎌진다.**"""
    stale = [
        k
        for k in UNREAD_BY_DESIGN
        if re.search(rf"\b{re.escape(k.split('.', 1)[1])}\b", source_text)
    ]
    assert not stale, f"이제 읽히는데 면제 목록에 남아 있다: {stale}"


def test_serving_model_config_reaches_the_client():
    """`model.*` 이 **서빙**까지 닿는다 (D-65 · D-69).

    `qlora.py`(학습)가 읽는 것만으로는 부족하다. 학습에서만 읽고 서빙엔 안 닿는 것이
    2026-08-02 의 사각지대였다 — `revision` 핀이 *"평가한 모델"* 을 가리키는 줄 알았다.
    """
    from pettriage.models.serving.client import LocalQwenClient

    cfg = C.load_config("default").model
    c = LocalQwenClient(
        base_id=cfg.base_id,
        adapter_path=cfg.adapter_path,
        revision=cfg.revision,
        dtype=cfg.dtype,
        load_in_4bit=cfg.load_in_4bit,
    )
    # 가중치를 올리지 않고 **설정이 전달됐는지**만 본다 — 8GB 를 받지 않는다.
    assert c._revision == cfg.revision  # noqa: SLF001
    assert c._dtype == cfg.dtype  # noqa: SLF001
    assert c._4bit == cfg.load_in_4bit  # noqa: SLF001
    assert hasattr(c, "run") and hasattr(c, "run_raw")


def test_dotenv_example_only_advertises_working_overrides():
    """`.env.example` 이 안내하는 `PETTRIAGE__…` 가 **실제로 먹는지** (D-69).

    이 파일은 처음부터 `PETTRIAGE__SERVE__ENGINE=graph` 를 예시로 들고 있었는데
    `.env` 의 그 줄은 **한 번도 동작하지 않았다.** 안내가 거짓이면 사람은 설정을
    바꿨다고 믿고 안 바뀐 조건으로 측정한다.
    """
    example = ROOT / ".env.example"
    if not example.exists():
        pytest.skip(".env.example 없음 — 소스 트리에서만 검사한다")
    keys = set(re.findall(r"PETTRIAGE__([A-Z0-9_]+)", example.read_text(encoding="utf-8")))
    keys.discard("PROFILE")  # 프로파일은 오버라이드가 아니다
    known = {
        f"{sec}__{name}".upper()
        for sec, m in (
            ("MODEL", C.ModelConfig),
            ("TRAIN", C.TrainConfig),
            ("RETRIEVAL", C.RetrievalConfig),
            ("TRIAGE", C.TriageConfig),
            ("SERVE", C.ServeConfig),
            ("AUTH", C.AuthConfig),
        )
        for name in m.model_fields
    }
    unknown = sorted(k for k in keys if k not in known)
    assert not unknown, f".env.example 이 없는 설정 키를 안내한다: {unknown}"


def test_revision_pin_is_paired_with_base_id():
    """`revision` 을 그대로 둔 채 `base_id` 만 바꾸면 **설정에서 막는다**.

    커밋 해시는 저장소마다 다르다. 안 막으면 `from_pretrained` 가 404 로 죽는데,
    오류가 *"revision 이 저 저장소 것이 아니다"* 라고 말해 주지 않는다 —
    **8GB 를 받다가 만난다.**
    """
    import pydantic

    ok = C.ModelConfig(base_id="Qwen/Qwen3-4B", revision="1cfa9a7208912126459214e8b04321603b3df60c")
    assert ok.revision

    with pytest.raises(pydantic.ValidationError, match="revision"):
        C.ModelConfig(
            base_id="unsloth/Qwen3-4B-unsloth-bnb-4bit",
            revision="1cfa9a7208912126459214e8b04321603b3df60c",
        )

    # 비우면 통과한다 — 재현이 깨지는 것은 04 §8 에 적을 일이지 막을 일이 아니다.
    free = C.ModelConfig(base_id="unsloth/Qwen3-4B-unsloth-bnb-4bit", revision=None)
    assert free.revision is None
