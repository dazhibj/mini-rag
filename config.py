import os
from pathlib import Path

BASE_DIR = Path(__file__).parent
KNOWLEDGE_DIR = Path(os.environ.get("KNOWLEDGE_DIR", BASE_DIR / "knowledge_base"))
CHROMA_DIR = BASE_DIR / "chroma_db"

EMBEDDING_MODEL = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
HF_ENDPOINT = "https://hf-mirror.com"
CHUNK_SIZE = 512
CHUNK_OVERLAP = 64
COLLECTION_NAME = "documents"
TOP_K = 5
DISPLAY_MAX_CHARS = 500
