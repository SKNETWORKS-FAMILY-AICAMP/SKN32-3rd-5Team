# 기여 규약

> 이 저장소는 **공개**다. 자료 원문과 학습 산출물은 커밋하지 않는다 (D-29 · D-41).

## 시작

```bash
python -m venv .venv && source .venv/bin/activate
make install      # 의존성 + pre-commit 훅
make test         # 68 tests
make serve        # http://127.0.0.1:8000
```

GPU가 없어도 여기까지는 전부 돈다. 학습 의존성은 `.[train]` 으로 분리되어 있다.

## 브랜치와 PR

| 항목 | 규칙 |
|---|---|
| 브랜치 | `feature/<ws>-<작업명>` · 예: `feature/ws2-graph-engine` |
| `main` | 보호. 직접 푸시하지 않는다 |
| PR | CI 통과 + 리뷰 1인 |
| 커밋 | `feat(ws2): ...` · `fix(ws1): ...` · `docs: ...` |

## 반드시 지키는 것

**① 새 판단 지점이 생기면 3문을 먼저 적용한다** (docs/05 §1).
결과를 `docs/06_설계결정기록.md` 에 D 번호로 남기고, **트레이드오프 문장을 함께 적는다.**
감수한 것을 적지 않은 "왜"는 자기 합리화다.

**② LLM에 맡기기로 했다면 같은 PR에 검증 코드를 넣는다** (docs/05 §6).

**③ 안전 관련 조정은 위험도 상향 방향으로만 한다.**
`src/pettriage/triage/` 를 고치는 PR은 `tests/test_triage_gate.py` 가 반드시 통과해야 한다.

**④ 자료를 수집하기 전에 게이트를 통과시킨다** (docs/05 §8.1).
받고 나서 버리지 않는다.

**⑤ 파라미터를 코드에 박지 않는다.** `configs/*.yaml` 로 뺀다.
실험 보고 시 그 YAML과 `constraints.txt` 커밋 해시를 함께 적으면 재현이 성립한다 (04 §8).

**⑥ 무거운 임포트(torch·transformers)는 함수 안에서 한다.**
모듈 최상단으로 올리면 GPU 없는 팀원과 CI가 깨진다. CI의 `test` job이 `.[api,dev]` 만
설치하는 것으로 이 제약이 강제된다.

## 커밋하지 않는 것

| 대상 | 이유 |
|---|---|
| `data/` 아래 자료 파일 | 이용약관 제약 (D-29). `git add -f` 를 쓰지 않는다 |
| `artifacts/` 학습 어댑터 | 용량. 재현은 `configs/` + `constraints.txt` + 시드로 한다 |
| `.env` | 비밀 |

`pre-commit` 훅과 CI의 `no-data-committed` job이 이중으로 막는다.

## 워크스트림별 진입점

| WS | 담당 | 주 작업 파일 |
|---|---|---|
| WS1 데이터 | 수집·전처리·인덱싱 | `src/pettriage/ingest/` · `data/manifests/` |
| WS2 RAG | 그래프·검색·판정 | `src/pettriage/graph/` · `retrieval/` · `triage/` |
| WS3 sLLM | 학습·서빙 | `src/pettriage/models/` · `configs/train.yaml` |
| WS4 평가 | 골든셋·지표 | `eval/` · `src/pettriage/tools/` |
| WS5 UI | 화면 | `web/` · `src/pettriage/app/` |

## 문서

문서가 기준이고 코드가 따라간다. 코드와 문서가 어긋나면 **어느 쪽이 맞는지 먼저 정하고**
한쪽만 고치지 않는다. 다이어그램(`docs/` 밖 배포본)도 같은 규칙을 따른다.
