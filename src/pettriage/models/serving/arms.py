"""04 §3 비교군 정의 — **갈아 끼우는 지점은 여기 하나다** (D-22 · D-65).

    python scripts/smoke_llm.py --arm D          # 환경이 준비됐나
    python eval/harness/run_eval.py --arm D      # 그 구성으로 잰다

## 왜 한 곳에 모으나

    비교군을 바꾸려면 `PETTRIAGE__MODEL__*` 를 서너 개 손으로 맞춰야 했다.
    하나를 빠뜨리면 **다른 조건으로 재고도 모른다** — `provider=qwen` 인데
    `adapter_path` 가 남아 있으면 D 를 잰다고 생각하며 C 를 잰다.

    04 §8 재현성이 무너지는 방식이 오타(`UnknownConfigKey`)와 같다.
    이름 하나로 고르게 하고, 그 이름이 리포트에 그대로 남는다.

## 네 구성

    none  LLM 없음 — 5태스크 전부 폴백. **코드·규칙만의 기준선**
    A     대형 LLM (OpenAI 호환). `OPENAI_API_KEY` 필요
    A-LC  같은 모델을 **LangChain 으로** — 연동 방식만 다르다 (D-71)
    D     Qwen3-4B 베이스. 로컬 로드
    C     Qwen3-4B + LoRA 어댑터 (파인튜닝)

    ⚠️ `none` 은 비교군이 아니라 **기준선**이다. 04 §3 의 A·C·D 와 나란히 두되,
       *"LLM 이 하는 일이 전부 빠지면 얼마나 되는가"* 를 재는 자리다.
       이 값이 A 와 비슷하면 **LLM 을 넣은 의미가 없다는 뜻**이므로 반드시 함께 낸다.
"""

from __future__ import annotations

import logging

log = logging.getLogger(__name__)

#: 비교군 이름 → `PETTRIAGE__MODEL__*` 환경변수.
#:
#: **값을 여기 적고 코드에서 읽는다.** 문서에만 적으면 손으로 맞추게 되고,
#: 손으로 맞추면 빠뜨린다 (D-69 — 설정은 코드가 읽어야 한다).
ARMS: dict[str, dict[str, str]] = {
    "none": {
        "PETTRIAGE__MODEL__PROVIDER": "none",
    },
    "A": {
        "PETTRIAGE__MODEL__PROVIDER": "api",
        # 기본은 `configs/default.yaml` 의 api_model. 다른 것을 쓰려면
        # `--arm A` 뒤에 PETTRIAGE__MODEL__API_MODEL 을 따로 준다.
    },
    # **같은 모델을 LangChain 으로 부른다** (D-71 · 필수 산출물).
    # A 와 결과가 같아야 하고, 같다는 것이 `LLMClient` 추상이 실제로
    # 갈아 끼울 수 있다는 증거다 (D-21). 다르면 둘 중 하나가 프롬프트를 바꾼 것이다.
    "A-LC": {
        "PETTRIAGE__MODEL__PROVIDER": "langchain",
    },
    "D": {
        "PETTRIAGE__MODEL__PROVIDER": "qwen",
        # **어댑터를 반드시 비운다.** 남아 있으면 D 를 잰다고 생각하며 C 를 잰다.
        "PETTRIAGE__MODEL__ADAPTER_PATH": "null",
    },
    "C": {
        "PETTRIAGE__MODEL__PROVIDER": "qwen",
        "PETTRIAGE__MODEL__ADAPTER_PATH": "artifacts/adapters/qwen3-4b-mt",
    },
}

#: 각 구성이 요구하는 것. `smoke_llm.py` 가 이것을 보고 점검한다.
NEEDS: dict[str, str] = {
    "none": "없음 — 어떤 환경에서도 돈다",
    "A": "OPENAI_API_KEY (.env). OpenAI 호환 엔드포인트면 api_base_url 도",
    "A-LC": "A 와 같고 + pip install -e '.[rag]' (langchain-openai)",
    "D": "pip install -e '.[qwen]' · 가중치 ≈3GB(4bit) / ≈8GB(bf16)",
    "C": "D 와 같고 + artifacts/adapters/qwen3-4b-mt (이서은 팀원 산출물)",
}


#: 비교군이 통제하는 환경변수 접두사. **먼저 비우고 세운다.**
_MANAGED = "PETTRIAGE__MODEL__"


def apply_arm(name: str) -> dict[str, str]:
    """**셸에 남은 `PETTRIAGE__MODEL__*` 를 전부 비우고** 이 비교군의 값만 세운다.

    ⚠️ 비우지 않으면 **앞 실험이 다음 측정에 새어 든다.** 실제로 그랬다 —

        $env:PETTRIAGE__MODEL__BASE_ID="unsloth/Qwen3-4B-unsloth-bnb-4bit"   (Qwen 시도)
        ...
        python ... --arm A      ← PROVIDER 만 덮이고 BASE_ID 는 그대로 남는다
        → 리포트 provenance 에 base_id=unsloth/… 가 박힌다. **거짓이다.**

    `arms.py` 를 만든 이유가 *"하나 빠뜨리면 다른 조건으로 재고도 모른다"* 였는데,
    **덮어쓰기만 해서는 그 문제가 그대로 남는다.** 세션을 새로 여는 것에 기대지 않는다
    (2026-08-02 실측 · 04 §8).

    **상시 설정은 `.env` 에 둔다.** 셸만 비우므로 `.env` 의 값은 살아남는다 —
    호스팅 Qwen(`api_base_url`) 처럼 계속 쓰는 것은 거기 적는다 (D-69).

    ⚠️ **`get_config()` 보다 먼저 불러야 한다.** 설정은 `lru_cache` 라 한 번 읽히면 굳는다.

    Returns:
        `{"applied": {...}, "cleared": [...]}` 를 합친 dict — **지운 것도 보고한다.**
        조용히 지우면 *"왜 내 설정이 안 먹지"* 가 된다.
    """
    import os

    if name not in ARMS:
        raise SystemExit(f"알 수 없는 비교군: {name!r} — {', '.join(ARMS)} 중 하나")

    stale = sorted(k for k in os.environ if k.startswith(_MANAGED))
    for k in stale:
        del os.environ[k]

    applied = dict(ARMS[name])
    os.environ.update(applied)

    leftover = [k for k in stale if k not in applied]
    if leftover:
        log.warning(
            "앞 실험의 환경변수를 지웠다 — %s (상시 설정은 .env 에 둔다)", ", ".join(leftover)
        )

    from ...config import reset_caches
    from .factory import reset_client

    reset_caches()
    reset_client()
    return applied
