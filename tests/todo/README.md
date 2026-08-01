# tests/todo/ — 남은 일 목록

**여기 있는 테스트는 아직 실패한다.** 구현이 안 됐기 때문이다.
`make test` 에서는 제외되므로 CI 는 초록을 유지한다.

```bash
pytest -m todo              # 남은 일 전체
pytest -m todo -k slot      # 한 노드만
pytest -m todo -x           # 첫 실패에서 멈추기
```

**이 테스트를 초록으로 만드는 것이 완료 기준이다.** 문서를 다 읽지 않아도
테스트가 요구 조건을 말해준다. 조건을 바꿔야 한다고 판단되면 **먼저 팀장과 확인**한다 —
여기 적힌 것 대부분이 06 의 설계 결정에서 나왔다.

구현이 끝나면 `src/pettriage/graph/nodes/__init__.py` 의
`NODES_IMPLEMENTED` 를 `True` 로 바꾼다. 그래야 `serve.engine=graph` 가 뜬다.
