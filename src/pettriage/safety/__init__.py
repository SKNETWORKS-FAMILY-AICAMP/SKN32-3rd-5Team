"""생성물이 사용자에게 나가기 전에 코드가 거는 안전 장치.

`triage/` 가 **등급을 올리는** 쪽이라면 여기는 **내용을 빼는** 쪽이다.
둘 다 LLM에게 맡기지 않는다 (D-38 · 축① "결정론은 코드로").
"""

from .contacts import (
    GUIDANCE,
    ScrubResult,
    ScrubResult2,
    has_contact,
    scrub_contacts,
    scrub_items,
)

__all__ = [
    "GUIDANCE",
    "ScrubResult",
    "ScrubResult2",
    "has_contact",
    "scrub_contacts",
    "scrub_items",
]
