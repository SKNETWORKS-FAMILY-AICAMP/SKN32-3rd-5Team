"""D-85 — **없음과 모름은 다르다.**

    없음   사용자가 물질을 말하지 않았다        → 되묻는다 (D-49)
    모름   말했는데 코퍼스에 그 물질이 없다      → **근거없음 거절**

2026-08-03 60건 측정에서 여섯 건이 같은 자리에 걸렸다 —

    G-020 목캔디 · G-026 달팽이약 · G-049 실리카겔
    G-050 매니큐어 · G-051 모기향 · G-052 계피가루

전부 코퍼스에 없는 물질인데 시스템은 *"무엇을 먹었나요?"* 를 되물었다.
**사용자는 이미 말했다.** 되물으면 같은 답이 돌아오고, 그 사이 시간이 간다 —
D-68 이 종밖 물질에서 내린 결론(*"응급 상황에서 못 쓸 질문은 거절보다 나쁘다"*)과
구조가 똑같다. 골든셋도 여섯 건 모두 `refused / 근거없음` 을 기대하고 있었다.
"""

from __future__ import annotations

from pettriage.graph.build import _after_slots
from pettriage.graph.nodes.slots import _is_vague, _resolvable


class Test모호어:
    """🔴 **이 목록이 판단의 입력을 지킨다.**

    LLM 은 슬롯을 비우는 대신 `substance: "뭔가"` 를 채워 넣는다. 그것을
    *"말했다"* 로 읽으면 **되물어야 할 것을 거절**하게 된다 — 방향이 반대인 사고다.
    """

    def test_모른다는_말은_물질이_아니다(self):
        for surface in ("뭔가", "무언가", "뭔가 하얀 가루", "정체불명 알약", "이상한 걸", "미상"):
            assert _is_vague(surface), f"{surface!r} 을 물질 이름으로 읽었다"

    def test_비어_있으면_말하지_않은_것이다(self):
        assert _is_vague(None)
        assert _is_vague("")

    def test_진짜_물질_이름은_통과한다(self):
        for surface in ("목캔디", "달팽이약", "실리카겔", "매니큐어", "모기향", "계피가루"):
            assert not _is_vague(surface), f"{surface!r} 을 모호어로 읽었다"


class Test갈림길:
    """라우터는 순수 함수다. 그래프를 안 돌리고 행선지만 본다."""

    def test_말했는데_모르면_되묻지_않는다(self):
        st = {"unknown_substance": "목캔디", "missing_slots": ["substance"]}
        assert _after_slots(st) == "refuse_nohit", "되물어도 같은 답이 온다 (D-85)"

    def test_말하지_않았으면_되묻는다(self):
        """**이쪽을 깨뜨리면 안 된다.** 되묻기는 이 시스템의 핵심 안전 동작이다."""
        st = {"missing_slots": ["substance"]}
        assert _after_slots(st) == "clarify"

    def test_종밖은_그대로_거절이다(self):
        st = {"off_species_substance": "향초", "missing_slots": []}
        assert _after_slots(st) == "refuse_nohit"

    def test_모호_후보가_있으면_검색으로_간다(self):
        """D-62 — 모호는 실패가 아니다. 후보를 전부 검색어로 넘긴다."""
        st = {"substance_candidates": ["세제A", "세제B"], "missing_slots": []}
        assert _after_slots(st) == "filter"


class Test구표면형:
    """D-88 — **아는 물질을 모른다고 거절하지 않는다.**

    D-86 으로 ②슬롯이 진짜 일하기 시작하자 모델이 구(句)를 뽑아 오게 됐고,
    그 값이 폐쇄 목록에서 떨어지면서 D-85 가 발동했다. 60건 실측 4건.
    """

    def test_폐쇄목록에_오르면_안다(self):
        assert _resolvable("초콜릿", "dog")
        assert _resolvable("양파", "dog")

    def test_모호도_안다(self):
        """후보가 여럿인 것은 실패가 아니다 (D-62)."""
        assert _resolvable("세제", "cat")

    def test_종밖도_안다(self):
        """물질은 아는데 이 종 자료가 없는 것 — 되묻기가 아니라 거절로 간다 (D-68)."""
        assert _resolvable("향초", "cat")

    def test_구는_그대로는_못_오른다(self):
        """이것이 D-88 이 존재하는 이유다."""
        for surface in ("세제 거품", "니코틴 껌 조각", "양파국", "감기약"):
            assert not _resolvable(surface, "cat"), f"{surface!r} 이 그냥 올라간다"

    def test_문장으로_읽으면_안쪽_어휘가_잡힌다(self):
        """🔴 `mention_in` 은 **문장**용이다. 표면형에 쓰면 부분 매칭이 터진다 (D-87)."""
        from pettriage.compute.vocabulary import mention_in

        assert mention_in("고양이가 세제 거품을 핥았어요", "cat") == "세제"
        assert mention_in("앵무새가 니코틴 껌 조각을 쪼아 먹었어요", "bird") == "니코틴 껌"

    def test_D85_여섯개는_그대로_거절이다(self):
        """**이쪽을 깨뜨리면 안 된다.** 문장을 훑어도 나오지 않아야 D-85 가 산다."""
        from pettriage.compute.vocabulary import mention_in

        for surface in ("목캔디", "달팽이약", "실리카겔", "매니큐어", "계피가루"):
            assert not _resolvable(surface, "dog")
            assert mention_in(f"강아지가 {surface}를 먹었어요", "dog") is None

    def test_모호어는_문장으로_내려가지_않는다(self):
        """사용자가 *모른다*고 말했다. 문장을 뒤져 물질을 세우면 그게 추정이다 (D-49)."""
        assert _is_vague("뭔가")
        assert _is_vague("뭔지 모르겠")
