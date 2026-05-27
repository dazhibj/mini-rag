import pytest

from chunker import _is_qa_boundary, chunk_documents, chunk_text


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

    def test_overlap_within_long_paragraph(self):
        text = "abcdefghijklmnopqrstuvwxyz"
        result = chunk_text(text, chunk_size=10, overlap=5)
        if len(result) > 1:
            assert result[0][-5:] == result[1][:5]

    def test_overlap_between_paragraph_chunks(self):
        paras = "\n\n".join(["A" * 50, "B" * 50, "C" * 50])
        result = chunk_text(paras, chunk_size=60, overlap=10)
        if len(result) > 1:
            tail = result[0][-10:]
            head = result[1][:10]
            assert tail == head, f"overlap failed: {tail!r} != {head!r}"

    def test_qa_boundary_force_split(self):
        text = "问: 第一个问题\n答: 第一个答案\n\n问: 第二个问题\n答: 第二个答案\n\n问: 第三个问题\n答: 第三个答案"
        result = chunk_text(text, chunk_size=500)
        assert len(result) >= 3
        assert "第一个问题" in result[0]
        assert "第二个问题" in result[1]
        assert "第三个问题" in result[2]

    def test_qa_boundary_mixed_with_normal_paragraphs(self):
        text = "普通段落。\n\n问: 问题一\n答: 答案一\n\n问: 问题二\n答: 答案二"
        result = chunk_text(text, chunk_size=500)
        assert any("普通段落" in c for c in result)
        assert any("问题一" in c for c in result)
        assert any("问题二" in c for c in result)

    def test_is_qa_boundary(self):
        assert _is_qa_boundary("问: 你好吗") is True
        assert _is_qa_boundary("答: 我很好") is False
        assert _is_qa_boundary("Q: hello") is True
        assert _is_qa_boundary("q: world") is True
        assert _is_qa_boundary("普通段落") is False
        assert _is_qa_boundary("") is False


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
