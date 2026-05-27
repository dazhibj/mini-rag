import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from chunker import chunk_documents
from config import (
    CHROMA_DIR,
    CHUNK_OVERLAP,
    CHUNK_SIZE,
    COLLECTION_NAME,
    DISPLAY_MAX_CHARS,
    EMBEDDING_MODEL,
    HF_ENDPOINT,
    KNOWLEDGE_DIR,
    SCORE_THRESHOLD,
    TOP_K,
)
from embedder import Embedder
from loader import load_documents
from retriever import format_results, print_results
from vector_store import VectorStore

_embedder_cache: dict[str, Embedder] = {}


def _get_embedder() -> Embedder:
    key = f"{EMBEDDING_MODEL}::{HF_ENDPOINT}"
    if key not in _embedder_cache:
        _embedder_cache[key] = Embedder(EMBEDDING_MODEL, HF_ENDPOINT)
    return _embedder_cache[key]


def _get_store() -> VectorStore:
    return VectorStore(CHROMA_DIR, COLLECTION_NAME, _get_embedder())


def cmd_index():
    print(f"Loading documents from {KNOWLEDGE_DIR} ...")
    docs = list(load_documents(KNOWLEDGE_DIR))
    print(f"Found {len(docs)} documents.")

    print("Chunking ...")
    chunks = chunk_documents(docs, chunk_size=CHUNK_SIZE, overlap=CHUNK_OVERLAP)
    print(f"Created {len(chunks)} chunks.")

    print(f"Loading embedding model ({EMBEDDING_MODEL}) ...")
    store = _get_store()

    print("Indexing into ChromaDB ...")
    store.add_documents(chunks)
    print(f"Done. Total vectors: {store.count}")


def cmd_query(query: str):
    store = _get_store()
    results = store.search(query, top_k=TOP_K, threshold=SCORE_THRESHOLD)
    print_results(format_results(results), max_chars=DISPLAY_MAX_CHARS)


def cmd_prompt(query: str):
    store = _get_store()
    results = store.search(query, top_k=TOP_K, threshold=SCORE_THRESHOLD)
    items = format_results(results)

    if not items:
        print("[No relevant context found.]")
        return

    print("请你基于以下参考资料回答用户的问题。\n")
    print("参考资料：")
    for i, item in enumerate(items, 1):
        print(f"[{i}] ({item.filename}) score={item.score:.4f}")
        print(item.content)
        print()
    print("用户问题：" + query)


def main():
    parser = argparse.ArgumentParser(description="RAG Embedding Tool")
    sub = parser.add_subparsers(dest="command", required=True)

    sub.add_parser("index", help="Index all documents in knowledge_base/")

    query_parser = sub.add_parser("query", help="Search the index")
    query_parser.add_argument("query", help="Search query text")

    prompt_parser = sub.add_parser("prompt", help="Generate a complete RAG prompt for an LLM")
    prompt_parser.add_argument("query", help="Search query text")

    args = parser.parse_args()

    if args.command == "index":
        cmd_index()
    elif args.command == "query":
        cmd_query(args.query)
    elif args.command == "prompt":
        cmd_prompt(args.query)


if __name__ == "__main__":
    main()
