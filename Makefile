.PHONY: help install serve test lint fmt verify train up down docker clean

help:            ## 이 목록
	@grep -E '^[a-z-]+:.*?##' $(MAKEFILE_LIST) | awk 'BEGIN{FS=":.*?## "}{printf "  \033[36m%-10s\033[0m %s\n", $$1, $$2}'

install:         ## 개발 환경 설치 (GPU 없이 API·테스트까지)
	pip install -e '.[api,rag,ingest,dev]' -c constraints.txt
	pre-commit install

serve:           ## FastAPI + 데모 프론트 → http://127.0.0.1:8000
	uvicorn pettriage.app.main:app --reload --app-dir src

test:            ## 안전 장치 회귀 테스트
	pytest

lint:            ## 정적 검사
	ruff check src tests scripts
	ruff format --check src tests scripts

fmt:             ## 자동 정리
	ruff check --fix src tests scripts
	ruff format src tests scripts

verify:          ## 층 0 — 코퍼스 정합성 + 자료 유출 확인
	python scripts/verify_corpus.py
	bash scripts/check_no_data.sh

train:           ## Qwen3-4B QLoRA 학습 (GPU 필요)
	PETTRIAGE_PROFILE=train python -m pettriage.models.training.qlora \
		--data data/train/samples.jsonl --out artifacts/adapters/qwen3-4b-mt

up:              ## API + pgvector 기동
	docker compose up --build

down:            ## 컨테이너 정리
	docker compose down

docker:          ## API 이미지만 빌드
	docker build -t pettriage-api .

clean:           ## 캐시·빌드 산출물 삭제
	find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true
	rm -rf .pytest_cache .ruff_cache .mypy_cache .coverage htmlcov build dist *.egg-info
