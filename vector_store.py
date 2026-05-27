from __future__ import annotations

from pathlib import Path

import chromadb
from chromadb.config import Settings

from embedder import Embedder


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

        texts = [c[2] for c in chunks]
        ids = [f"{c[0]}::chunk::{c[3]}" for c in chunks]
        metadatas = [
            {"filename": c[0], "ext": c[1], "chunk_index": c[3]} for c in chunks
        ]
        embeddings = self.embedder.embed(texts)

        self.collection.add(
            ids=ids,
            embeddings=embeddings,
            documents=texts,
            metadatas=metadatas,
        )

    def search(self, query: str, top_k: int = 5) -> list[dict]:
        query_emb = self.embedder.embed_query(query)
        results = self.collection.query(
            query_embeddings=[query_emb],
            n_results=top_k,
        )
        output: list[dict] = []
        for i in range(len(results["ids"][0])):
            output.append({
                "id": results["ids"][0][i],
                "document": results["documents"][0][i] if results.get("documents") else None,
                "metadata": results["metadatas"][0][i],
                "distance": results["distances"][0][i] if results.get("distances") else None,
            })
        return output

    @property
    def count(self) -> int:
        return self.collection.count()
