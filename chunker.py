from __future__ import annotations

import re


def _is_qa_boundary(para: str) -> bool:
    return bool(re.match(r"^(问|Q|q)[：:]", para.strip()))


def chunk_text(text: str, chunk_size: int = 512, overlap: int = 64) -> list[str]:
    paragraphs = re.split(r"\n\s*\n", text.strip())
    chunks: list[str] = []
    buffer = ""
    prev_tail = ""

    for para in paragraphs:
        para = para.strip()
        if not para:
            continue

        if _is_qa_boundary(para) and buffer:
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
) -> list[tuple[str, str, str, int]]:
    result: list[tuple[str, str, str, int]] = []
    for filename, ext, content in documents:
        chunks = chunk_text(content, chunk_size, overlap)
        for i, chunk in enumerate(chunks):
            result.append((filename, ext, chunk, i))
    return result
