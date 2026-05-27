from __future__ import annotations

import re


def chunk_text(text: str, chunk_size: int = 512, overlap: int = 64) -> list[str]:
    paragraphs = re.split(r"\n\s*\n", text.strip())
    chunks: list[str] = []
    buffer = ""

    for para in paragraphs:
        para = para.strip()
        if not para:
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
                buffer = para

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
