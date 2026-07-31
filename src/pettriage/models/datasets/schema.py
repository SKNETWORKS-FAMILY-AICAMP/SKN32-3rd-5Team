"""학습 샘플 스키마.

설계 근거: docs/03 §3 학습 데이터 생성 (distillation)

    샘플이 **어디서 왔는지**를 필드로 들고 다닌다.
    04 §7 실패 분석에서 *"이 오답은 어떤 데이터에서 왔나"* 에 답해야 하고,
    D-36에 따라 실사용 입력이 학습셋에 섞이지 않았음을 보일 수 있어야 한다.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal

from ..tasks import Task

#: 샘플의 출처. `real_user` 는 **의도적으로 값이 없다** —
#: 학습·평가는 가상 프로필만 쓴다 (D-18 · D-36).
Origin = Literal["distilled", "template", "handwritten", "synthetic_profile"]


@dataclass
class TrainSample:
    sample_id: str
    task: Task
    input: str
    target: str

    origin: Origin
    #: distillation 이면 교사 모델 ID. 재현성 기록 (04 §8).
    teacher: str | None = None
    #: 근거가 된 코퍼스 자료. 실패 분석에서 역추적한다.
    source_ids: list[str] = field(default_factory=list)
    species: str | None = None
    #: 사람이 검수했는가. 04 §2.4 작성자·검수자 분리.
    reviewed_by: str | None = None
    split: Literal["train", "dev", "test"] = "train"

    def to_chat(self) -> list[dict[str, str]]:
        from ..prompts import build_sample

        return build_sample(self.task, self.input, self.target)
