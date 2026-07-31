# scripts/

셸에서 바로 부르는 얇은 진입점. **로직은 여기 두지 않는다.**

```
verify_corpus.py    → pettriage.tools.verify_corpus 를 부르는 래퍼
check_no_data.sh    자료 파일 커밋 차단 (pre-commit · CI 공용)
```

Python 로직을 `scripts/` 에 두면 `pip install` 후 임포트할 수 없어
콘솔 스크립트(`pettriage-verify`)가 깨진다. 그래서 실제 구현은
`src/pettriage/tools/` 에 있다.
