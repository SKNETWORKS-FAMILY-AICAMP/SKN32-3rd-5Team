# configs/

**재현에 필요한 값은 전부 여기 있고, 커밋된다.**
비밀·환경차이(API 키·DB 주소)는 `.env` 로 가며 커밋되지 않는다.

| 파일 | 언제 |
|---|---|
| `default.yaml` | 항상 먼저 읽힌다. 모든 기본값의 단일 출처 |
| `train.yaml` | `PETTRIAGE_PROFILE=train` — GPU 학습 |
| `eval.yaml` | `PETTRIAGE_PROFILE=eval` — 골든셋 채점 |

병합 순서는 `default.yaml` → `<profile>.yaml` → **환경변수**다.

```bash
# 파일을 고치지 않고 한 번만 바꿔 실행
PETTRIAGE__RETRIEVAL__TOP_K=8 make serve
```

## 왜 파라미터를 코드에서 뺐나

04 §8이 재현성을 요구한다. 파라미터가 코드에 흩어져 있으면
*"그때 top-k가 몇이었더라"* 에 답할 수 없다.
**실험 결과를 보고할 때 이 폴더의 YAML을 그대로 첨부하면 재현이 된다.**

## 안전에 직결되는 값

`triage.max_clarify_turns` 는 `app/contracts.py` 의 `MAX_CLARIFY_TURNS` 와
**반드시 같아야 한다** — 테스트가 이 일치를 검사한다.
`retrieval.score_threshold` 를 낮추면 거절이 줄고 **환각이 는다.**
골든셋 없이 조정하지 않는다.
