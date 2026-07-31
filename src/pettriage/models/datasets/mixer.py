"""태스크 혼합 — `train.task_mix` 비율대로 샘플을 섞는다.

설계 근거: docs/03 §3 · docs/04 E4 (태스크 간섭)

    혼합 비율은 **간섭 실험의 조작 변수**다. 코드에 박아두면 실험이 안 된다.
    그래서 `configs/*.yaml` 에서 읽고, 실제로 몇 건이 들어갔는지 로그로 남긴다.

torch·datasets 를 부르지 않는다 — **GPU 없이 테스트할 수 있어야 한다.**
"""

from __future__ import annotations

import logging
import random
from collections import Counter
from collections.abc import Sequence

from ..tasks import Task
from .schema import TrainSample

log = logging.getLogger(__name__)


class MixTargetUnreachable(ValueError):
    """요구 비율을 채울 샘플이 없다.

    조용히 적게 넣으면 *"비율대로 학습했다"* 는 보고가 거짓이 된다 (04 §8).
    """


def mix(
    samples: Sequence[TrainSample],
    ratios: dict[str, float],
    total: int,
    *,
    seed: int = 42,
    strict: bool = True,
) -> list[TrainSample]:
    """비율대로 뽑아 섞는다.

    Args:
        ratios: 태스크명 → 비율. 합이 1이 아니어도 정규화한다.
        total: 최종 샘플 수.
        strict: 요구 수량을 못 채우면 예외. False면 경고 후 있는 만큼.

    Raises:
        MixTargetUnreachable: strict 이고 특정 태스크의 샘플이 모자랄 때.
    """
    rng = random.Random(seed)
    by_task: dict[str, list[TrainSample]] = {}
    for s in samples:
        by_task.setdefault(str(s.task), []).append(s)

    denom = sum(ratios.values())
    if denom <= 0:
        raise ValueError("task_mix 의 비율 합이 0이다.")

    out: list[TrainSample] = []
    shortfalls: list[str] = []
    for name, ratio in ratios.items():
        want = round(total * ratio / denom)
        pool = by_task.get(name, [])
        if len(pool) < want:
            shortfalls.append(f"{name}: 요구 {want} · 보유 {len(pool)}")
            take = pool[:]
        else:
            take = rng.sample(pool, want)
        out.extend(take)

    if shortfalls:
        msg = "태스크 혼합 비율을 채우지 못했다 — " + " / ".join(shortfalls)
        if strict:
            raise MixTargetUnreachable(msg)
        log.warning(msg)

    rng.shuffle(out)
    log.info("혼합 결과: %s", dict(Counter(str(s.task) for s in out)))
    return out


def check_leakage(train: Sequence[TrainSample], eval_: Sequence[TrainSample]) -> list[str]:
    """학습셋과 평가셋의 입력 중복을 찾는다.

    골든셋 100건(04 §2.3)이 학습에 섞이면 **평가 전체가 무의미해진다.**
    CI에서 돌릴 수 있도록 순수 함수로 둔다.
    """
    seen = {s.input.strip() for s in train}
    return sorted({s.sample_id for s in eval_ if s.input.strip() in seen})


def task_counts(samples: Sequence[TrainSample]) -> dict[str, int]:
    return dict(Counter(str(s.task) for s in samples))


__all__ = ["MixTargetUnreachable", "Task", "check_leakage", "mix", "task_counts"]
