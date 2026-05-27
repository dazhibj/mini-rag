import pytest

from chunker import chunk_documents, chunk_text


class TestChunkText:
    def test_empty_text(self):
        assert chunk_text("") == []
        assert chunk_text("   ") == []
        assert chunk_text("\n\n\n") == []

    def test_single_short_paragraph(self):
        result = chunk_text("Hello world", chunk_size=100)
        assert result == ["Hello world"]

    def test_multiple_paragraphs(self):
        text = "第一段。\n\n第二段。\n\n第三段。"
        result = chunk_text(text, chunk_size=100)
        assert len(result) == 1
        assert "第一段" in result[0]
        assert "第二段" in result[0]

    def test_paragraph_exceeds_chunk_size(self):
        long_para = "a" * 20 + " " + "b" * 20
        result = chunk_text(long_para, chunk_size=15, overlap=0)
        assert len(result) > 1

    def test_overlap(self):
        text = "abcdefghijklmnopqrstuvwxyz"
        result = chunk_text(text, chunk_size=10, overlap=5)
        if len(result) > 1:
            assert result[0][-5:] == result[1][:5]


class TestChunkDocuments:
    def test_empty(self):
        assert chunk_documents([], chunk_size=100) == []

    def test_single_document(self):
        docs = [("test.txt", ".txt", "段落一。\n\n段落二。")]
        result = chunk_documents(docs, chunk_size=100)
        assert len(result) == 1
        filename, ext, content, idx = result[0]
        assert filename == "test.txt"
        assert ext == ".txt"
        assert idx == 0

    def test_multiple_documents(self):
        docs = [
            ("a.txt", ".txt", "内容A"),
            ("b.txt", ".txt", "内容B\n\n更多内容B"),
        ]
        result = chunk_documents(docs, chunk_size=50)
        assert len(result) >= 2
        assert result[0][0] == "a.txt"
        assert result[1][0] == "b.txt"
