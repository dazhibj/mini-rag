# mini-rag — RAG Embedding Tool

## Commands
```bash
source .venv/bin/activate
python main.py index         # build vector index from knowledge_base/
python main.py query "..."   # search and print results
python main.py prompt "..."  # generate a complete RAG prompt for an LLM
python -m pytest tests/ -v   # run tests (35 tests, coverage ~95%)
python -m coverage run -m pytest tests/ && python -m coverage report -m
```

## Key Design
- **Chunker**: paragraph-boundary splitting + character-based chunk_size/overlap
- **Embedder**: local fallback (ChromaDB ONNX all-MiniLM-L6-v2) or transformers model
- **Vector store**: ChromaDB with cosine similarity
- **Loader**: supports `.txt`, `.md`, `.xlsx` via `knowledge_base/`
- **Knowledge dir**: overridable via `KNOWLEDGE_DIR` env var (default: `knowledge_base/`)

## Config (config.py)
- Model: `sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2` (multilingual/Chinese)
- Mirror: `https://hf-mirror.com` (China-friendly HuggingFace mirror)
- Chunk: size=512, overlap=64, top_k=5

## Repository
- GitHub: `git@github.com:dazhibj/mini-rag.git` (public, MIT license)
- Author: 18601024482@163.com

## Recent Work / Current State
- All edge cases fixed (empty search, corrupt excel, paragraph overlap)
- Embedder cached across commands in main.py
- No internet access in dev environment — BAAI/bge-m3 (2.2GB) too large, falls back to local ONNX
- Multilingual MiniLM (1.1GB) works via hf-mirror.com
- Python 3.12, CPU-only torch

## TODO
- **Extend chunker**: add support for custom delimiters/methods beyond current Q&A boundary detection (e.g., markdown headers, code blocks, regex-based splits)

## Conventions
- No comments in code (opencode style preference)
- Minimal output, concise responses
- Tests before committing
