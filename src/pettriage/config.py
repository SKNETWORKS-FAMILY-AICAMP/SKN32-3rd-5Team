"""설정 로딩 — YAML(기본값) + 환경변수(비밀·환경차이).

설계 근거: docs/04_테스트-평가계획.md §8 재현성 · docs/05 §1 축①

    **무엇을 어디에 두는가**

      configs/*.yaml   재현에 필요한 값. 커밋한다.
                       모델 이름, LoRA rank, top-k, 청크 크기, 임계값…
      .env             환경마다 다르거나 비밀인 값. 커밋하지 않는다.
                       API 키, DB 접속 문자열, 트레이싱 on/off…

    실험 결과를 보고할 때 **YAML 파일을 그대로 첨부하면 재현이 된다.**
    파라미터가 코드에 흩어져 있으면 04 §8의 재현성 요건을 만족할 수 없다.

    환경변수가 YAML을 덮어쓴다 — `PETTRIAGE__RETRIEVAL__TOP_K=8` 처럼
    이중 밑줄로 중첩 필드를 지정한다. 임시 실험에 파일을 고칠 필요가 없다.
"""

from __future__ import annotations

import logging
import os
from functools import lru_cache
from pathlib import Path
from typing import Any, Literal

import yaml
from pydantic import BaseModel, Field, SecretStr
from pydantic_settings import BaseSettings, SettingsConfigDict

from . import paths

log = logging.getLogger(__name__)


class ConfigNotFound(RuntimeError):
    """``configs/`` 를 찾지 못했다.

    조용히 기본값으로 되돌아가면 **평가 프로파일이 무시된 채 지표가 산출**된다.
    그 지표는 오염된 것이므로, 기본값 폴백은 명시적으로 허용할 때만 한다
    (``PETTRIAGE_ALLOW_DEFAULT_CONFIG=1``).
    """


# ─────────────────────────────────────────────────────────────
# YAML 로 관리하는 값 — 재현에 필요하다
# ─────────────────────────────────────────────────────────────
class ModelConfig(BaseModel):
    """생성·파인튜닝 모델 (D-42)."""

    base_id: str = "Qwen/Qwen3-4B"
    revision: str | None = None  # 재현성: 모델도 버전을 고정한다
    max_seq_len: int = 4096
    dtype: Literal["bfloat16", "float16", "auto"] = "bfloat16"
    load_in_4bit: bool = True
    adapter_path: str | None = None  # 학습된 LoRA 어댑터. None이면 베이스만


class LoRAConfig(BaseModel):
    r: int = 16
    alpha: int = 32
    dropout: float = 0.05
    target_modules: list[str] = Field(
        default_factory=lambda: [
            "q_proj",
            "k_proj",
            "v_proj",
            "o_proj",
            "gate_proj",
            "up_proj",
            "down_proj",
        ]
    )


class TrainConfig(BaseModel):
    """멀티태스크 QLoRA 학습 (03 §2)."""

    seed: int = 42  # 04 §8 — 시드 고정
    epochs: float = 3.0
    lr: float = 2e-4
    batch_size: int = 2
    grad_accum: int = 8
    warmup_ratio: float = 0.03
    lora: LoRAConfig = Field(default_factory=LoRAConfig)
    #: 태스크별 샘플 비율. 태스크 간섭(03·04 E4) 실험의 조작 변수다.
    task_mix: dict[str, float] = Field(
        default_factory=lambda: {
            "classify": 0.2,
            "slot": 0.2,
            "compress": 0.2,
            "verify": 0.3,  # ④ 근거 검증이 이 프로젝트의 핵심 태스크 (D-05)
            "simplify": 0.1,
        }
    )


class RetrievalConfig(BaseModel):
    """검색 (02 §8)."""

    embedding_model: str = "BAAI/bge-m3"
    top_k: int = 5
    #: 이 값 미만이면 **검색 실패로 간주하고 거절한다** (02 §8.3·§9).
    #:
    #: ⚠️ **임계값 하나로는 거절을 만들 수 없다** (D-46 실측).
    #: 근거 있음 0.547~0.733 / 근거 없음 0.494~0.659 로 분포가 겹친다.
    #: 올리면 근거가 있는 질의가 거절되어 **과소평가**가 된다 (D-13).
    #: 거절은 ① 의도 분류와 ④ 근거 검증이 만든다. 이 값은 최소 방어선일 뿐이다.
    score_threshold: float = 0.50
    rerank: bool = False  # 구현 2단계 이후
    chunk_strategy: Literal["substance", "fixed"] = "substance"  # D-14
    fixed_chunk_size: int = 500  # 비교군 전용 (04 E1)
    #: 벡터DB (D-44). `memory` 는 모델·디스크 없이 도는 테스트용이다.
    store: Literal["chroma", "memory"] = "chroma"
    #: Chroma 영속 디렉터리. 지우고 `build_index.py` 로 통째로 재생성된다.
    persist_dir: str = ".chroma"
    collection: str = "external"


class TriageConfig(BaseModel):
    """트리아지 (D-09 · D-39). **여기 값은 안전에 직결된다.**"""

    #: 규칙 미적중 시 LLM을 부를지. False면 규칙만으로 판정하고 미적중은 거절한다.
    llm_fallback: bool = True
    #: 되묻기 상한 (02 §9). 계약(contracts.MAX_CLARIFY_TURNS)과 반드시 일치해야 한다.
    max_clarify_turns: int = 2
    #: 조류는 정량 임계치가 0건이라 체중·섭취량 슬롯을 요구하지 않는다 (D-09 개정).
    quantitative_species: list[str] = Field(default_factory=lambda: ["dog", "cat"])


class AuthConfig(BaseModel):
    """토큰 파라미터. **비밀이 아니다** — 값이 새도 위조에 쓸 수 없다 (D-41).

    비밀은 서명 키 하나뿐이고 그건 `Secrets` 에 있다.
    """

    algorithm: Literal["HS256", "HS384", "HS512"] = "HS256"
    #: 응급 도메인이라 세션이 짧다. 길게 잡으면 탈취 토큰의 유효기간이 길어진다.
    expire_minutes: int = Field(default=60, ge=5, le=1440)


class ServeConfig(BaseModel):
    host: str = "127.0.0.1"
    port: int = 8000
    engine: Literal["stub", "graph"] = "stub"
    cors_origins: list[str] = Field(default_factory=list)

    #: 기동 시 임베딩 모델을 미리 올린다 (D-53).
    #: 끄면 **첫 질의가 로딩을 맞는다** — 02 §12.4 로 스트리밍이 없어 그 시간이 침묵이 된다.
    #: 노드를 고치며 서버를 자주 재시작하는 개발 중에만 끈다:
    #:     PETTRIAGE__SERVE__WARMUP=false make serve
    warmup: bool = True


class AppConfig(BaseModel):
    """YAML 전체 트리."""

    model: ModelConfig = Field(default_factory=ModelConfig)
    train: TrainConfig = Field(default_factory=TrainConfig)
    retrieval: RetrievalConfig = Field(default_factory=RetrievalConfig)
    triage: TriageConfig = Field(default_factory=TriageConfig)
    serve: ServeConfig = Field(default_factory=ServeConfig)
    auth: AuthConfig = Field(default_factory=AuthConfig)


# ─────────────────────────────────────────────────────────────
# 환경변수로 관리하는 값 — 비밀이거나 환경마다 다르다
# ─────────────────────────────────────────────────────────────
class Secrets(BaseSettings):
    """`.env` 에서 읽는다. **커밋되지 않는다.**"""

    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")

    openai_api_key: SecretStr | None = None
    anthropic_api_key: SecretStr | None = None
    langchain_api_key: SecretStr | None = None
    database_url: str | None = None

    #: JWT 서명 키. **기본값을 주지 않는다.**
    #:
    #: `"change-me-in-production"` 같은 자리표시자를 기본값으로 두면
    #: **아무도 안 바꾼 채 그대로 배포된다.** 키를 아는 사람은 누구나 토큰을 위조한다.
    #: 없으면 `app.auth` 가 명시적으로 실패한다 — 조용히 약한 키로 도는 것보다 낫다.
    #: 만들 때: `python -c "import secrets; print(secrets.token_urlsafe(32))"`
    #:
    #: 알고리즘·만료 시간은 **여기 없다.** 비밀이 아니므로 `configs/*.yaml` 의
    #: `auth` 절로 옮겼다 (D-41 — 파라미터는 설정, 비밀은 환경변수).
    jwt_secret_key: SecretStr | None = None

    data_dir: Path = Field(default_factory=paths.data_dir)
    vectorstore_dir: Path = Field(default_factory=lambda: paths.data_dir().parent / ".chroma")


# ─────────────────────────────────────────────────────────────
# 로딩
# ─────────────────────────────────────────────────────────────
def _deep_merge(base: dict[str, Any], over: dict[str, Any]) -> dict[str, Any]:
    out = dict(base)
    for k, v in over.items():
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = _deep_merge(out[k], v)
        else:
            out[k] = v
    return out


def _parse_scalar(raw: str) -> Any:
    """환경변수 값을 YAML 스칼라로 해석하되, 실패하면 **원문 문자열**로 둔다.

    ``*`` · ``&`` · ``%`` 로 시작하는 값은 YAML 문법상 오류라 그대로 두면
    앱 기동 자체가 죽는다. 설정 하나 때문에 서버가 안 뜨는 것은 과하다.
    """
    try:
        return yaml.safe_load(raw)
    except yaml.YAMLError:
        return raw


def _env_overrides(prefix: str = "PETTRIAGE__") -> dict[str, Any]:
    """`PETTRIAGE__RETRIEVAL__TOP_K=8` → `{"retrieval": {"top_k": 8}}`.

    임시 실험에 YAML을 고치지 않아도 되게 한다.
    **덮어쓴 값은 로그에 남는다** — 실험 결과를 나중에 해석하려면 필수다 (04 §8).

    리스트는 YAML 표기를 쓴다: ``PETTRIAGE__TRIAGE__QUANTITATIVE_SPECIES="[dog, cat]"``
    """
    out: dict[str, Any] = {}
    applied: list[str] = []
    for key, raw in sorted(os.environ.items()):
        if not key.startswith(prefix):
            continue
        node = out
        parts = key[len(prefix) :].lower().split("__")
        for part in parts[:-1]:
            node = node.setdefault(part, {})
        node[parts[-1]] = _parse_scalar(raw)
        applied.append(f"{'.'.join(parts)}={raw}")
    if applied:
        log.warning("환경변수가 설정을 덮었다 — %s", " · ".join(applied))
    return out


def load_config(profile: str = "default") -> AppConfig:
    """`configs/default.yaml` → `configs/<profile>.yaml` → 환경변수 순으로 덮는다.

    Raises:
        ConfigNotFound: `configs/` 를 못 찾았고 기본값 폴백도 허용되지 않은 경우.
    """
    configs = paths.config_dir()
    merged: dict[str, Any] = {}
    loaded_files: list[str] = []

    if configs is not None:
        for name in dict.fromkeys(["default", profile]):
            path = configs / f"{name}.yaml"
            if path.exists():
                merged = _deep_merge(merged, yaml.safe_load(path.read_text("utf-8")) or {})
                loaded_files.append(path.name)

    if not loaded_files:
        msg = (
            f"설정 파일을 찾지 못했다 (profile={profile}). "
            "기본값으로 돌아가면 평가 프로파일이 무시된 채 지표가 산출된다. "
            "PETTRIAGE_CONFIG_DIR 로 경로를 지정하거나 저장소 루트에서 실행할 것."
        )
        if os.getenv("PETTRIAGE_ALLOW_DEFAULT_CONFIG") != "1":
            raise ConfigNotFound(msg)
        log.warning("%s — PETTRIAGE_ALLOW_DEFAULT_CONFIG=1 이라 기본값으로 진행한다.", msg)
    elif profile != "default" and f"{profile}.yaml" not in loaded_files:
        log.warning("프로파일 %s.yaml 이 없다 — default.yaml 만 적용되었다.", profile)

    merged = _deep_merge(merged, _env_overrides())
    return AppConfig.model_validate(merged)


@lru_cache(maxsize=1)
def get_config() -> AppConfig:
    return load_config(os.getenv("PETTRIAGE_PROFILE", "default"))


@lru_cache(maxsize=1)
def get_secrets() -> Secrets:
    return Secrets()


def reset_caches() -> None:
    """캐시를 비운다. **테스트 전용** — 런타임 중에는 부르지 않는다.

    설정이 프로세스 전역으로 고정되면 앞 테스트의 환경변수가 뒤 테스트를 오염시킨다.
    """
    get_config.cache_clear()
    get_secrets.cache_clear()
