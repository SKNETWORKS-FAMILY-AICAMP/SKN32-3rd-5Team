.PHONY: help install serve test todo lint fmt verify facts golden eval index train up down docker clean

help:            ## 이 목록
	@grep -E '^[a-z-]+:.*?##' $(MAKEFILE_LIST) | awk 'BEGIN{FS=":.*?## "}{printf "  \033[36m%-10s\033[0m %s\n", $$1, $$2}'

install:         ## 개발 환경 설치 (GPU 없이 API·테스트까지)
	pip install -e '.[api,rag,ingest,dev]' -c constraints.txt
	@git rev-parse --git-dir >/dev/null 2>&1 \
		&& pre-commit install \
		|| echo "· git 저장소가 아니라 pre-commit 훅은 건너뛴다 (설치는 완료)"

serve:           ## FastAPI + 데모 프론트 → http://127.0.0.1:8000
	uvicorn pettriage.app.main:app --reload --app-dir src

test:            ## 안전 장치 회귀 테스트
	pytest

todo:            ## 남은 일 목록 — 구현하면 초록이 된다
	pytest -m todo

lint:            ## 정적 검사
	ruff check src tests scripts
	ruff format --check src tests scripts

fmt:             ## 자동 정리
	ruff check --fix src tests scripts
	ruff format src tests scripts

verify:          ## 층 0 — 코퍼스 정합성 + 자료 유출 확인
	python scripts/verify_corpus.py
	bash scripts/check_no_data.sh

facts:           ## 사실 표 검사 (WS1) — 01e 지침
	python scripts/check_facts.py

golden:          ## 골든셋 검사 (WS4) — 04a 지침
	python scripts/check_goldenset.py

eval:            ## 평가 하네스 — 골든셋 채점 (04 §4). 엔진은 configs 의 serve.engine
	python eval/harness/run_eval.py --json eval/reports/latest.json

index:           ## 사실 표 → 청크 (적재는 --store chroma)
	python scripts/build_index.py

train:           ## Qwen3-4B QLoRA 학습 (GPU 필요)
	PETTRIAGE_PROFILE=train python -m pettriage.models.training.qlora \
		--data data/train/samples.jsonl --out artifacts/adapters/qwen3-4b-mt

up:              ## API + MySQL 기동
	docker compose up --build

db:              ## MySQL 만 기동 — 로컬에 설치하지 않는다 (D-48)
	docker compose up -d db

down:            ## 컨테이너 정리
	docker compose down

docker:          ## API 이미지만 빌드
	docker build -t pettriage-api .

clean:           ## 캐시·빌드 산출물 삭제
	find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true
	rm -rf .pytest_cache .ruff_cache .mypy_cache .coverage htmlcov build dist *.egg-info
