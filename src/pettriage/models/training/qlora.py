"""Qwen3-4B QLoRA 학습 — HF PEFT + TRL.

설계 근거: docs/03 §4 · docs/06 D-42

    실행:  PETTRIAGE_PROFILE=train python -m pettriage.models.training.qlora

무거운 임포트(torch·transformers)를 **함수 안에서** 한다.
GPU 없는 팀원이 이 모듈을 임포트해도 깨지지 않아야 하고,
CI가 스키마 테스트를 돌릴 수 있어야 한다.
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path
from typing import Any

from ...config import AppConfig, get_config

log = logging.getLogger(__name__)


def build_peft_config(cfg: AppConfig) -> Any:
    from peft import LoraConfig

    lora = cfg.train.lora
    return LoraConfig(
        r=lora.r,
        lora_alpha=lora.alpha,
        lora_dropout=lora.dropout,
        target_modules=lora.target_modules,
        bias="none",
        task_type="CAUSAL_LM",
    )


def build_quant_config(cfg: AppConfig) -> Any | None:
    if not cfg.model.load_in_4bit:
        return None
    import torch
    from transformers import BitsAndBytesConfig

    return BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
        bnb_4bit_compute_dtype=torch.bfloat16,
    )


def load_samples(path: Path) -> list[dict[str, Any]]:
    """JSONL 로드. `models/datasets/schema.py` 의 `TrainSample` 직렬화 형식."""
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]


def run_training(data_path: Path, out_dir: Path, cfg: AppConfig | None = None) -> Path:
    """QLoRA 학습을 돌리고 어댑터 경로를 반환한다."""
    import torch
    from datasets import Dataset
    from peft import get_peft_model
    from transformers import AutoModelForCausalLM, AutoTokenizer, set_seed
    from trl import SFTConfig, SFTTrainer

    cfg = cfg or get_config()
    set_seed(cfg.train.seed)  # 04 §8 재현성

    from ..datasets.mixer import mix
    from ..datasets.schema import TrainSample
    from ..tasks import Task

    raw = load_samples(data_path)
    samples = [TrainSample(**{**r, "task": Task(r["task"])}) for r in raw]
    train = [s for s in samples if s.split == "train"]
    dev = [s for s in samples if s.split == "dev"]

    # 골든셋 오염 검사 — 여기서 막지 않으면 평가 전체가 무의미해진다
    from ..datasets.mixer import check_leakage

    leaked = check_leakage(train, dev)
    if leaked:
        raise RuntimeError(f"학습셋이 평가셋과 겹친다 ({len(leaked)}건): {leaked[:5]}")

    mixed = mix(train, cfg.train.task_mix, total=len(train), seed=cfg.train.seed)

    tok = AutoTokenizer.from_pretrained(cfg.model.base_id, revision=cfg.model.revision)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token

    ds = Dataset.from_list([{"messages": s.to_chat()} for s in mixed])

    model = AutoModelForCausalLM.from_pretrained(
        cfg.model.base_id,
        revision=cfg.model.revision,
        quantization_config=build_quant_config(cfg),
        torch_dtype=torch.bfloat16 if cfg.model.dtype == "bfloat16" else "auto",
        device_map="auto",
    )
    model = get_peft_model(model, build_peft_config(cfg))
    model.print_trainable_parameters()

    trainer = SFTTrainer(
        model=model,
        train_dataset=ds,
        processing_class=tok,
        args=SFTConfig(
            output_dir=str(out_dir),
            num_train_epochs=cfg.train.epochs,
            learning_rate=cfg.train.lr,
            per_device_train_batch_size=cfg.train.batch_size,
            gradient_accumulation_steps=cfg.train.grad_accum,
            warmup_ratio=cfg.train.warmup_ratio,
            max_seq_length=cfg.model.max_seq_len,
            bf16=True,
            logging_steps=10,
            save_strategy="epoch",
            seed=cfg.train.seed,
            report_to=[],
        ),
    )
    trainer.train()
    trainer.save_model(str(out_dir))

    # 어떤 설정으로 학습했는지 어댑터 옆에 남긴다 — 04 §8
    (out_dir / "train_config.json").write_text(cfg.model_dump_json(indent=2), encoding="utf-8")
    log.info("어댑터 저장: %s", out_dir)
    return out_dir


def main() -> int:
    ap = argparse.ArgumentParser(description="Qwen3-4B QLoRA 멀티태스크 학습")
    ap.add_argument("--data", type=Path, required=True, help="학습 샘플 JSONL")
    ap.add_argument("--out", type=Path, default=Path("artifacts/adapters/qwen3-4b-mt"))
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    run_training(args.data, args.out)
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
