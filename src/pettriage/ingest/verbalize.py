"""사실 표 → 문장. **코드가 한다. LLM 호출이 없다.**

설계 근거: docs/06_설계결정기록.md · D-38

이 모듈에 LLM 클라이언트를 import하는 순간 D-38이 깨진다.
문장 생성이 비결정적이 되면 검증 대상이 필드에서 문장으로 되돌아간다.
"""

from __future__ import annotations

from ..schemas import Chunk, Fact
from .templates import TEMPLATES


class NoTemplate(KeyError):
    """해당 doc_type의 템플릿이 없다. 임의 생성하지 않고 실패한다."""


def verbalize(fact: Fact) -> str:
    """사실 1건을 한국어 문장으로 만든다.

    결측 필드의 절은 생략된다 — "정보 없음"을 출력하지 않는다.
    """
    try:
        template = TEMPLATES[fact.doc_type]
    except KeyError as exc:  # pragma: no cover - 스키마가 막아준다
        raise NoTemplate(
            f"'{fact.doc_type}' 템플릿이 없다. templates.py에 정의하기 전까지 적재하지 않는다."
        ) from exc
    return template.render(fact)


def to_chunk(fact: Fact, *, chunk_id: str | None = None) -> Chunk:
    """경로 ② — 사실 추출본을 청크로. ``quote`` 는 비운다 (D-37)."""
    return Chunk(
        chunk_id=chunk_id or f"c-{fact.fact_id}",
        text=verbalize(fact),
        source_id=fact.source_id,
        publisher=fact.publisher,
        species=fact.species,
        doc_type=fact.doc_type,
        substance=fact.substance,
        route="사실추출",
        quote=None,  # 경로 ②는 원문을 담지 않는다
        locator=fact.locator,
        fact_ids=[fact.fact_id],
    )
