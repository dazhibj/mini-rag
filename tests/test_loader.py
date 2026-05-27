from pathlib import Path

import pytest

from loader import SUPPORTED_EXTS, load_documents, load_excel_file, load_text_file


class TestLoadTextFile:
    def test_read_txt(self, tmp_path: Path):
        f = tmp_path / "test.txt"
        f.write_text("你好世界", encoding="utf-8")
        assert load_text_file(f) == "你好世界"

    def test_read_md(self, tmp_path: Path):
        f = tmp_path / "test.md"
        f.write_text("# Title\ncontent", encoding="utf-8")
        assert load_text_file(f) == "# Title\ncontent"

    def test_nonexistent_file(self, tmp_path: Path):
        assert load_text_file(tmp_path / "nope.txt") is None

    def test_corrupted_excel_returns_none(self, tmp_path: Path):
        f = tmp_path / "bad.xlsx"
        f.write_text("not an excel file")
        result = load_excel_file(f)
        assert result is None


class TestLoadDocuments:
    def test_load_txt(self, tmp_path: Path):
        f = tmp_path / "hello.txt"
        f.write_text("test content", encoding="utf-8")
        results = list(load_documents(tmp_path))
        assert len(results) == 1
        rel, ext, content = results[0]
        assert rel == "hello.txt"
        assert ext == ".txt"
        assert content == "test content"

    def test_skip_unsupported(self, tmp_path: Path):
        (tmp_path / "image.png").write_text("fake")
        results = list(load_documents(tmp_path))
        assert results == []

    def test_supported_extensions(self):
        assert ".txt" in SUPPORTED_EXTS
        assert ".md" in SUPPORTED_EXTS
        assert ".xlsx" in SUPPORTED_EXTS
