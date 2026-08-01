"""연락처 차단 (D-47).

**응급 상황에서 걸리지 않는 번호를 주는 것은 오답보다 나쁘다.**
전화를 걸어보는 시간만큼 병원 도착이 늦어지기 때문이다.
"""

from __future__ import annotations

import pytest

from pettriage.safety import GUIDANCE, ScrubResult, has_contact, scrub_contacts


class TestHasContact:
    @pytest.mark.parametrize(
        "sentence",
        [
            "ASPCA APCC(888-426-4435)에 연락하세요.",
            "Pet Poison Helpline 855-764-7661 로 전화한다.",
            "즉시 1-855-764-7661 에 연락한다.",
            "(888) 426-4435 로 문의할 것.",
            "888.426.4435 로 상담하세요.",
            "24시간 상담: 855 289 0358",
        ],
    )
    def test_번호가_있으면_잡는다(self, sentence: str) -> None:
        assert has_contact(sentence)

    @pytest.mark.parametrize(
        "sentence",
        [
            "초콜릿은 개에게 20 mg/kg 이상에서 임상 징후가 나타난다.",
            "주목의 최소 치사량은 2.3 g leaves/kg 다.",
            "포도는 3 g/kg 에서 사망 보고가 있다.",
            "가까운 동물병원에 즉시 연락한다.",
            "구토·설사·떨림이 나타나면 병원에 연락한다.",
        ],
    )
    def test_용량과_국내안내는_남긴다(self, sentence: str) -> None:
        assert not has_contact(sentence)

    def test_기관명만_있고_지시가_없으면_남긴다(self) -> None:
        """*"APCC 자료에 따르면"* 은 출처 인용이지 연락 지시가 아니다."""
        assert not has_contact("ASPCA APCC 자료에 따르면 아보카도는 조류에게 치명적이다.")

    def test_기관명에_연락_지시가_붙으면_잡는다(self) -> None:
        """번호를 안 적어도 **국내에서 걸 수 없는 창구**로 보내는 것은 같다."""
        assert has_contact("Pet Poison Helpline 에 연락하세요.")


class TestScrub:
    def test_해당_문장만_빼고_안내를_붙인다(self) -> None:
        r = scrub_contacts("초콜릿은 개에게 독성이 있다. ASPCA APCC(888-426-4435)에 연락하세요.")
        assert "888-426-4435" not in r.text
        assert "초콜릿은 개에게 독성이 있다." in r.text
        assert r.text.endswith(GUIDANCE)
        assert r.removed == ["ASPCA APCC(888-426-4435)에 연락하세요."]
        assert r.changed

    def test_연락처가_없으면_원문_그대로다(self) -> None:
        text = "초콜릿은 개에게 20 mg/kg 이상에서 임상 징후가 나타난다. (출처: MDPI, S-034)"
        r = scrub_contacts(text)
        assert r.text == text
        assert not r.changed

    def test_전부_연락처면_안내만_남는다(self) -> None:
        """**빈 답을 내는 것이 틀린 번호를 주는 것보다 낫다.**"""
        r = scrub_contacts("Pet Poison Helpline 855-764-7661 로 전화한다.")
        assert r.text == GUIDANCE
        assert len(r.removed) == 1

    def test_뺀_것을_기록한다(self) -> None:
        """무엇을 뺐는지 남기지 않으면 검증할 수 없다 (04 §8)."""
        r = scrub_contacts("A는 위험하다. 855-764-7661 로 연락. B도 위험하다.")
        assert len(r.removed) == 1
        assert "A는 위험하다." in r.text
        assert "B도 위험하다." in r.text

    @pytest.mark.parametrize("text", ["", "   ", "\n"])
    def test_빈_입력(self, text: str) -> None:
        assert scrub_contacts(text) == ScrubResult(text, [])

    def test_용량_수치를_번호로_오인하지_않는다(self) -> None:
        """`2.8-36.4 g/kg` 같은 범위가 전화번호로 잡히면 답변이 통째로 사라진다."""
        text = "건포도는 2.8 mg/kg 에서 보고되었고 같은 논문 표는 2.8-36.4 g/kg 로 적는다."
        assert scrub_contacts(text).text == text


class TestFinalizeNode:
    """그래프 마지막 관문 (D-47). **거절 응답도 통과시킨다.**"""

    def test_연락처가_있으면_바꾸고_기록한다(self) -> None:
        from pettriage.graph.nodes import finalize

        out = finalize({"answer": "위험하다. ASPCA APCC(888-426-4435)에 연락하세요."})
        assert "888-426-4435" not in out["answer"]
        assert out["removed_contacts"]

    def test_연락처가_없으면_상태를_안_건드린다(self) -> None:
        """바뀐 키만 돌려준다 (02 §6.1). 빈 dict 가 정답이다."""
        from pettriage.graph.nodes import finalize

        assert finalize({"answer": "초콜릿은 개에게 독성이 있다."}) == {}

    def test_답이_비면_아무것도_안_한다(self) -> None:
        from pettriage.graph.nodes import finalize

        assert finalize({"status": "clarify"}) == {}


class TestGuidanceWording:
    """안내 문장은 **사용자에게 그대로 나가는 출력**이다 (D-47)."""

    def test_존댓말이다(self) -> None:
        """청크는 평서체지만 화면에 뜨는 문장은 보호자가 읽는 말이다."""
        assert GUIDANCE.endswith("주세요.")
        assert "습니다" in GUIDANCE
        assert "한다." not in GUIDANCE

    def test_특정_기관을_지목하지_않는다(self) -> None:
        """국내에 미국 APCC 에 대응하는 공식 창구가 없다 — 있는 것처럼 적으면 환각이다."""
        assert "동물병원" in GUIDANCE
        assert not any(c.isdigit() and c not in "24" for c in GUIDANCE.replace("24시", ""))

    def test_안내_문장_자체는_걸리지_않는다(self) -> None:
        """`연락` 이 들어 있어도 기관명·번호가 없으므로 남아야 한다."""
        assert not has_contact(GUIDANCE)

    def test_두_번_돌려도_안내가_겹치지_않는다(self) -> None:
        """`finalize` 가 두 번 불려도 결과가 같아야 한다 (멱등)."""
        once = scrub_contacts("위험하다. 855-764-7661 로 연락한다.")
        twice = scrub_contacts(once.text)
        assert twice.text == once.text
        assert not twice.changed
        assert once.text.count(GUIDANCE) == 1
