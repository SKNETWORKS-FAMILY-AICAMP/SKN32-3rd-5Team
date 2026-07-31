# eval/reports/

평가 실행 결과. 파일명에 실행 일자와 비교군을 넣는다.

```
2026-08-___A_대형LLM.md
2026-08-___C_qwen3-4b-adapter.md
비교표_최종.md
```

각 보고서 머리말에 다음을 적는다 (04 §8).

```yaml
코퍼스: <data/manifests 커밋 해시>
설정:   <configs 커밋 해시> · profile=eval
의존성: <constraints.txt 커밋 해시>
시드:   42
엔진:   engine=... engine_configured=...   # 둘이 다르면 그 결과는 폐기한다
```
