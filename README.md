# RAG Embedding Tool

A lightweight Retrieval-Augmented Generation (RAG) embedding tool. Index documents (`.txt`, `.md`, `.xlsx`) into a local vector database and retrieve relevant context for LLM prompts.

## Quick Start

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

# Put your files in knowledge_base/ then index
python main.py index

# Search
python main.py query "你的问题"

# Get a complete RAG prompt (for agent integration)
python main.py prompt "你的问题"
```

## Usage

### `index` — build the vector index

Scans the knowledge directory, chunks documents, and stores embeddings in ChromaDB.

```bash
python main.py index                         # uses knowledge_base/
KNOWLEDGE_DIR=/path/to/docs python main.py index  # custom directory
```

### `query` — search the index

```bash
python main.py query "退款政策是什么"
python main.py query "退款政策是什么" --threshold 0.4
```

### `prompt` — generate a complete RAG prompt

Outputs a ready-to-use prompt with retrieved context + user question. Designed for agent integration.

```bash
python main.py prompt "退款政策是什么"
python main.py prompt "退款政策是什么" --threshold 0.35  # stricter filter
```

**Agent integration:**

```python
import subprocess

def rag_query(question: str, threshold: float | None = None,
              knowledge_dir: str | None = None) -> str:
    cmd = ["python", "main.py", "prompt", question]
    if threshold is not None:
        cmd += ["--threshold", str(threshold)]
    env = {}
    if knowledge_dir:
        env["KNOWLEDGE_DIR"] = knowledge_dir
    result = subprocess.run(
        cmd, capture_output=True, text=True, cwd="/path/to/project",
        env=env or None,
    )
    return result.stdout  # complete prompt with context

prompt = rag_query("客户如何退货", threshold=0.4)
llm_response = your_llm.chat(prompt)
```

## Configuration

Edit `config.py`:

| Setting | Default | Description |
|---------|---------|-------------|
| `EMBEDDING_MODEL` | `sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2` | Model name or `"local"` for ChromaDB ONNX fallback |
| `HF_ENDPOINT` | `https://hf-mirror.com` | HuggingFace mirror (set empty for default endpoint) |
| `CHUNK_SIZE` | `512` | Max characters per chunk |
| `CHUNK_OVERLAP` | `64` | Overlap between chunks |
| `TOP_K` | `10` | Max results to retrieve (acts as cap) |
| `SCORE_THRESHOLD` | `0.5` | Cosine distance threshold — results above this are discarded (lower = stricter) |
| `KNOWLEDGE_DIR` env var | `knowledge_base/` | Override knowledge base path at runtime |

Set `EMBEDDING_MODEL = "local"` to use ChromaDB's built-in ONNX model (faster startup, English-optimized).

### Chinese embedding models

The default model supports 50+ languages including Chinese. For better Chinese quality with internet:

```
EMBEDDING_MODEL = "BAAI/bge-m3"     # Best quality, ~3GB, slower on CPU
EMBEDDING_MODEL = "shibing624/text2vec-base-chinese"  # Chinese-specific, ~400MB
```

## Testing

```bash
source .venv/bin/activate
python -m pytest tests/ -v
```

## Project Structure

```
├── main.py              # CLI: index / query / prompt
├── config.py            # Settings (env vars, model, chunking, threshold)
├── embedder.py          # Embedding (transformers or ChromaDB ONNX fallback)
├── loader.py            # Document loading (.txt, .md, .xlsx)
├── chunker.py           # Text chunking (Q&A boundary aware)
├── vector_store.py      # ChromaDB persistence & search (with score threshold)
├── retriever.py         # Result formatting
├── knowledge_base/      # Place your documents here
├── chroma_db/           # Vector index (auto-created)
└── tests/
    ├── test_chunker.py
    ├── test_loader.py
    ├── test_embedder.py
    ├── test_retriever.py
    └── test_vector_store.py
```

## Notes

- **Q&A chunking**: The chunker automatically detects `问:` / `Q:`-prefixed paragraphs
  and keeps each Q&A pair as its own chunk, preventing unrelated pairs from being
  merged together.
- **Q&A embedding**: At index time, each Q&A pair is split — only the question is
  embedded as the vector (for better query match), while the answer is stored in
  metadata. Both question and answer are shown in search results.
- **Relevance filtering**: Results are filtered by cosine distance threshold — results
  above `SCORE_THRESHOLD` are discarded, so you get only relevant context, not a
  fixed count of results.
