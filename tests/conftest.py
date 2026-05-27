import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import pytest


@pytest.fixture
def sample_chunks():
    return [
        ("doc1.txt", ".txt", "这是第一份文档的内容。", 0),
        ("doc2.md", ".md", "## 标题\n\n这是第二份文档。", 0),
        ("doc1.txt", ".txt", "这是第一份文档的第二段。", 1),
    ]


@pytest.fixture
def chroma_tmp_dir(tmp_path: Path):
    return str(tmp_path / "chroma_test")
