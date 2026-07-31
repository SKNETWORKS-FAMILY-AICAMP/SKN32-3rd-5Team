# 학습 이미지 — GPU 전용 (D-42).
#
#   빌드:  docker build -f docker/train.Dockerfile -t pettriage-train .
#   실행:  docker compose --profile train run --rm trainer \
#            --data data/train/samples.jsonl --out artifacts/adapters/qwen3-4b-mt
#
# CUDA 버전은 팀 GPU에 맞춰 태그를 고정한다. 바꾸면 constraints.txt 의 torch 도 함께 본다.

FROM pytorch/pytorch:2.5.1-cuda12.4-cudnn9-runtime

WORKDIR /app
ENV PIP_NO_CACHE_DIR=1 \
    PYTHONPATH=/app/src \
    PYTHONUNBUFFERED=1 \
    PETTRIAGE_ROOT=/app \
    PETTRIAGE_PROFILE=train \
    HF_HOME=/cache/hf

COPY pyproject.toml constraints.txt README.md LICENSE ./
COPY src ./src
COPY configs ./configs

# torch 는 베이스 이미지 것을 쓴다 — 재설치하면 CUDA 빌드가 깨진다
RUN pip install '.[train]' -c constraints.txt --no-deps \
 && pip install transformers peft trl bitsandbytes accelerate datasets -c constraints.txt

ENTRYPOINT ["python", "-m", "pettriage.models.training.qlora"]
