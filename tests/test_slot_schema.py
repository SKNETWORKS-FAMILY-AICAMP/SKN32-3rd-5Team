"""D-86 — **모델은 우리가 읽는 키 이름을 모른다.**

2026-08-03 실측. ② 슬롯 프롬프트에 JSON 스키마가 한 글자도 없었고, 모델은
키 이름을 지어냈다 —

    {'species': '개', 'concern': '목캔디'}                  ← 키가 concern
    {'species': '개', 'action': '…', 'item': '실리카겔'}      ← 키가 item
    {'species': '개', 'symptom': None, 'substance': '계피가루'} ← 유일하게 substance

코드는 `llm.get("substance")` 만 보므로 **뽑아 놓고 버렸다.** 여섯 건 중 하나만
읽혔다. 종도 마찬가지였다 — 여섯 건 전부 `'개'`·`'고양이'` 를 냈는데 코드는
`dog·cat·bird` 만 받아 버렸고, 값이 truthy 라서 **키워드 폴백도 건너뛰었다.**

D-73 이 ①분류에서 겪은 것과 **같은 사고**다. 그때 만든 처방(`labels` 를 단일
출처로 두고 프롬프트에 싣는다)을 ②에는 안 했던 것이다.

**그래서 지표가 거짓말을 했다** — JSON 파싱은 성공했으니 폴백이 안 찍혔고,
리포트는 *"3태스크 전부 모델 100%"* 를 냈다. 실제로는 키워드 폴백이 일하고 있었다.
"""

from __future__ import annotations

from pettriage.models.prompts import system_prompt
from pettriage.models.tasks import SPECS, Task


class Test단일출처:
    """스키마를 두 곳에 적으면 어긋난다. 어긋나는 순간 이 사고가 재발한다 (D-22)."""

    def test_스키마가_프롬프트에_실린다(self):
        prompt = system_prompt(Task.SLOT)
        for key in SPECS[Task.SLOT].output_keys:
            assert f'"{key}"' in prompt, f"{key} 가 프롬프트에 없다 — 모델이 알 수 없다"

    def test_종_값이_영문_코드로_지시된다(self):
        """`'개'` 로 오면 코드가 못 읽는다. 무엇을 내야 하는지 프롬프트가 말해야 한다."""
        prompt = system_prompt(Task.SLOT)
        for code in ("dog", "cat", "bird"):
            assert code in prompt

    def test_출력_키가_전부_상태_스키마에_있다(self):
        """모델에게 요구한 키를 상태가 못 받으면 그 값은 갈 곳이 없다."""
        from pettriage.graph.state import Slots

        unknown = set(SPECS[Task.SLOT].output_keys) - set(Slots.__annotations__)
        assert not unknown, f"Slots 에 없는 키를 모델에게 요구한다: {sorted(unknown)}"

    def test_라벨_태스크는_스키마가_없다(self):
        """①분류는 라벨 하나를 낸다. 스키마 블록이 붙으면 출력 형식이 흔들린다."""
        assert not SPECS[Task.CLASSIFY].output_keys


class Test종정규화:
    def test_한국어_종을_코드로_올린다(self):
        from pettriage.graph.nodes.slots import _normalize_species

        assert _normalize_species("개") == "dog"
        assert _normalize_species("강아지") == "dog"
        assert _normalize_species("고양이") == "cat"
        assert _normalize_species("앵무새") == "bird"

    def test_영문_코드는_그대로(self):
        from pettriage.graph.nodes.slots import _normalize_species

        assert _normalize_species("dog") == "dog"
        assert _normalize_species("DOG") == "dog"

    def test_모르는_값은_None(self):
        """**추측하지 않는다.** 종을 잘못 읽으면 포유류 기준이 조류에 간다 (D-10)."""
        from pettriage.graph.nodes.slots import _normalize_species

        assert _normalize_species("햄스터") is None
        assert _normalize_species(None) is None
        assert _normalize_species("") is None


class Test스키마밖출력:
    def test_지어낸_키를_잡는다(self):
        from pettriage.graph.nodes.slots import _off_schema_keys

        assert _off_schema_keys({"species": "dog", "concern": "목캔디"}) == ["concern"]
        assert _off_schema_keys({"action": "…", "item": "실리카겔"}) == ["action", "item"]

    def test_스키마를_지키면_비어_있다(self):
        from pettriage.graph.nodes.slots import _off_schema_keys

        assert _off_schema_keys({"species": "dog", "substance": "초콜릿"}) == []
        assert _off_schema_keys(None) == []
