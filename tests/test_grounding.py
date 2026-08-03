"""④ 근거 검증 — **작동하는지 확인한 적이 없었다.**

04 는 ④의 지표를 *"근거없음 탐지 재현율 — 놓치면 환각이 나간다"* 로 정해 두었다.
그런데 2026-08-03 확인에서 두 가지가 드러났다.

  ① `verdicts` 는 상태에 있는데 **아무도 집계하지 않았다** (D-75 와 같은 끊김).
  ② `Task.VERIFY` 는 **어디서도 호출되지 않는다** — 프롬프트·라벨(D-73)까지 있는데
     `verify_grounding` 은 2-gram 문자 일치율로 판정한다. 부르지 않으니 실패도
     안 남고, 그래서 폴백 집계에서 **성공한 것처럼 보였다.**

측정 지표(재현율)는 정답 라벨이 없어 만들 수 없다. 대신 **음성 대조**를 둔다 —
근거에 없는 문장을 일부러 넣고 걸리는지 본다. 걸리지 않으면 검증기는 **장식**이다.
"""

from __future__ import annotations

import pytest

from pettriage.graph.nodes.verify import _judge_sentence, verify_grounding

_CONTEXT = (
    "개가 테오브로민 20 mg/kg 을 섭취하면 임상징후가 나타난다. "
    "40-50 mg/kg 에서 중증 징후가 관찰되고 60 mg/kg 에서 발작이 발생한다."
)


class Test음성대조:
    """**근거에 없는 문장을 걸러내는가.** 못 걸르면 이 노드는 장식이다."""

    @pytest.mark.parametrize(
        "sentence",
        [
            "고양이는 백신을 맞으면 초콜릿을 먹어도 안전합니다",
            "포도는 조류에게 아무런 해가 없습니다",
            "집에서 과산화수소를 먹여 구토를 유도하세요",
        ],
    )
    def test_근거에_없는_문장은_근거없음(self, sentence):
        assert _judge_sentence(sentence, _CONTEXT) == "근거없음"

    def test_근거에_있는_문장은_근거있음(self):
        assert (
            _judge_sentence("테오브로민 20 mg/kg 에서 임상징후가 나타난다", _CONTEXT) == "근거있음"
        )

    def test_근거가_비면_근거없음(self):
        """**애매하면 근거없음 쪽으로** — 놓친 환각이 나가는 것보다 낫다."""
        assert _judge_sentence("무슨 말이든", "") == "근거없음"


class Test집계가_응답까지_나온다:
    """`verdicts` 가 상태에만 남고 아무도 안 읽던 것을 막는다 (D-75)."""

    def test_판정_결과를_돌려준다(self):
        out = verify_grounding(
            {
                "draft": "테오브로민 20 mg/kg 에서 임상징후가 나타난다. 백신을 맞으면 안전합니다.",
                "context": _CONTEXT,
            }  # type: ignore[arg-type]
        )
        verdicts = out["verdicts"]
        assert len(verdicts) == 2
        assert [v["verdict"] for v in verdicts] == ["근거있음", "근거없음"]

    def test_엔진이_응답에_싣는다(self):
        from pettriage.graph.engine import GraphEngine

        state = {
            "verdicts": [
                {"sentence": "a", "verdict": "근거있음"},
                {"sentence": "b", "verdict": "근거없음"},
                {"sentence": "c", "verdict": "모순"},
            ],
            "retry_count": 1,
        }
        g = GraphEngine._audit(state)  # type: ignore[arg-type]
        assert g["grounding"].checked == 3
        assert g["grounding"].unsupported == 1
        assert g["grounding"].contradicted == 1
        assert g["grounding"].retried is True

    def test_검증이_안_돌면_None(self):
        """**0건과 "안 돌았다"는 다르다.** 0으로 채우면 전건 통과처럼 보인다."""
        from pettriage.graph.engine import GraphEngine

        assert GraphEngine._audit({})["grounding"] is None  # type: ignore[arg-type]


class Test문서와코드가어긋난다:
    """🔴 **2026-08-03까지 05 §4 는 ④를 LLM 태스크로 적어 두었는데 호출부가 없었다.**

    `_llm_judge_sentence` 로 배선하며 해소됐다 — `verify` 가 `WIRED` 로 옮겨갔다.
    이 클래스 이름은 그 사고를 기록해 둔다. 다음에 어긋나면 여기 다시 채운다.
    """

    def test_verify_는_배선돼_있다(self):
        from pettriage.graph.fallbacks import UNWIRED, WIRED

        assert "verify" in WIRED
        assert "verify" not in UNWIRED

    def test_표시가_실제와_맞는다(self):
        """선언과 코드가 어긋나면 **선언 쪽이 거짓말**이 된다. 실제 호출부를 센다."""
        import inspect

        from pettriage.graph.fallbacks import UNWIRED, WIRED
        from pettriage.graph.nodes import classify, generate, slots, verify
        from pettriage.models.tasks import Task

        src = "\n".join(inspect.getsource(m) for m in (classify, slots, generate, verify))
        for name in WIRED:
            task = Task(name)
            assert f"Task.{task.name}" in src, f"{name} 은 배선됐다고 적혔는데 호출부가 없다"
        for name in UNWIRED:
            task = Task(name)
            called = f"client.run(Task.{task.name}" in src or f"_call_llm(Task.{task.name}" in src
            assert not called, f"{name} 이 이제 배선됐다 — UNWIRED 에서 빼고 05 §4 와 맞출 것"
