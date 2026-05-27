from __future__ import annotations

from pathlib import Path

import chromadb
from chromadb.config import Settings

from embedder import Embedder


def _split_qa(text: str) -> tuple[str, str | None]:
    for sep in ("\n答:", "\n答：", "\nA:", "\nA："):
        if sep in text:
            q, a = text.split(sep, maxsplit=1)
            return (q.strip(), (sep[1:] + a).strip())
    return (text, None)


class VectorStore:
    def __init__(self, persist_dir: str | Path, collection_name: str, embedder: Embedder):
        self.embedder = embedder
        client = chromadb.PersistentClient(
            path=str(persist_dir),
            settings=Settings(anonymized_telemetry=False),
        )
        self.collection = client.get_or_create_collection(
            name=collection_name,
            metadata={"hnsw:space": "cosine"},
        )

    def add_documents(
        self,
        chunks: list[tuple[str, str, str, int]],
    ) -> None:
        if not chunks:
            return

        texts = []
        ids = []
        metadatas = []
        for c in chunks:
            question, answer = _split_qa(c[2])
            texts.append(question)
            ids.append(f"{c[0]}::chunk::{c[3]}")
            meta = {"filename": c[0], "ext": c[1], "chunk_index": c[3]}
            if answer:
                meta["answer"] = answer
            metadatas.append(meta)
        embeddings = self.embedder.embed(texts)

        self.collection.add(
            ids=ids,
            embeddings=embeddings,
            documents=texts,
            metadatas=metadatas,
        )

    def search(self, query: str, top_k: int = 10, threshold: float | None = None) -> list[dict]:
        if self.count == 0:
            return []

        query_emb = self.embedder.embed_query(query)
        results = self.collection.query(
            query_embeddings=[query_emb],
            n_results=min(top_k, self.count),
        )
        output: list[dict] = []
        for i in range(len(results["ids"][0])):
            dist = results["distances"][0][i] if results.get("distances") else None
            if threshold is not None and dist is not None and dist > threshold:
                break
            output.append({
                "id": results["ids"][0][i],
                "document": results["documents"][0][i] if results.get("documents") else None,
                "metadata": results["metadatas"][0][i],
                "distance": dist,
            })
        return output

    @property
    def count(self) -> int:
        return self.collection.count()
