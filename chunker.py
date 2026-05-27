from __future__ import annotations

import re


_DEFAULT_BOUNDARY = re.compile(r"^(问|Q|q)[：:]")


def _is_qa_boundary(para: str) -> bool:
    return bool(_DEFAULT_BOUNDARY.match(para.strip()))


def _compile_boundaries(patterns: list[str]) -> list[re.Pattern]:
    return [re.compile(p) for p in patterns]


def chunk_text(
    text: str,
    chunk_size: int = 512,
    overlap: int = 64,
    boundary_patterns: list[str] | None = None,
) -> list[str]:
    if boundary_patterns:
        _checks = _compile_boundaries(boundary_patterns)

        def _is_boundary(para: str) -> bool:
            return any(p.match(para.strip()) for p in _checks)
    else:
        _is_boundary = _is_qa_boundary

    paragraphs = re.split(r"\n\s*\n", text.strip())
    chunks: list[str] = []
    buffer = ""
    prev_tail = ""

    for para in paragraphs:
        para = para.strip()
        if not para:
            continue

        if _is_boundary(para) and buffer:
            chunks.append(buffer)
            buffer = para
            prev_tail = para
            continue

        if len(buffer) + len(para) + 1 <= chunk_size:
            buffer = (buffer + "\n\n" + para).strip()
        else:
            if buffer:
                chunks.append(buffer)
            if len(para) > chunk_size:
                for i in range(0, len(para), chunk_size - overlap):
                    chunks.append(para[i : i + chunk_size])
                buffer = ""
            else:
                prefix = prev_tail[-overlap:] if overlap > 0 and prev_tail else ""
                buffer = (prefix + para) if prefix else para
        prev_tail = para

    if buffer:
        chunks.append(buffer)

    return chunks


def chunk_documents(
    documents: list[tuple[str, str, str]],
    chunk_size: int = 512,
    overlap: int = 64,
    boundary_patterns: list[str] | None = None,
) -> list[tuple[str, str, str, int]]:
    result: list[tuple[str, str, str, int]] = []
    for filename, ext, content in documents:
        chunks = chunk_text(content, chunk_size, overlap, boundary_patterns)
        for i, chunk in enumerate(chunks):
            result.append((filename, ext, chunk, i))
    return result
