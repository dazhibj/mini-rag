import pytest

from embedder import Embedder
from vector_store import VectorStore, _split_qa


@pytest.fixture
def local_store(chroma_tmp_dir):
    embedder = Embedder("local")
    return VectorStore(chroma_tmp_dir, "test_collection", embedder)


class TestVectorStore:
    def test_empty_count(self, local_store):
        assert local_store.count == 0

    def test_add_and_count(self, local_store, sample_chunks):
        local_store.add_documents(sample_chunks)
        assert local_store.count == 3

    def test_add_empty(self, local_store):
        local_store.add_documents([])
        assert local_store.count == 0

    def test_search_empty_collection(self, local_store):
        assert local_store.search("anything") == []

    def test_search_returns_results(self, local_store, sample_chunks):
        local_store.add_documents(sample_chunks)
        results = local_store.search("文档", top_k=5)
        assert len(results) > 0
        assert "id" in results[0]
        assert "document" in results[0]
        assert "metadata" in results[0]
        assert "distance" in results[0]

    def test_search_respects_top_k(self, local_store, sample_chunks):
        local_store.add_documents(sample_chunks)
        results = local_store.search("文档", top_k=2)
        assert len(results) == min(2, local_store.count)

    def test_search_metadata(self, local_store, sample_chunks):
        local_store.add_documents(sample_chunks)
        results = local_store.search("第一份文档", top_k=1)
        r = results[0]
        assert r["metadata"]["filename"] == "doc1.txt"
        assert r["metadata"]["chunk_index"] in (0, 1)

    def test_search_threshold_filters_distant_results(self, local_store, sample_chunks):
        local_store.add_documents(sample_chunks)
        all_results = local_store.search("文档", top_k=10)
        filtered = local_store.search("文档", top_k=10, threshold=0.0)
        assert len(filtered) <= len(all_results)

    def test_search_threshold_none_returns_all(self, local_store, sample_chunks):
        local_store.add_documents(sample_chunks)
        results = local_store.search("文档", top_k=10, threshold=None)
        assert len(results) > 0

    def test_persistence(self, chroma_tmp_dir, sample_chunks):
        embedder = Embedder("local")
        store1 = VectorStore(chroma_tmp_dir, "persist_test", embedder)
        store1.add_documents(sample_chunks)
        assert store1.count == 3

        store2 = VectorStore(chroma_tmp_dir, "persist_test", embedder)
        assert store2.count == 3


class TestSplitQA:
    def test_qa_pair(self):
        q, a = _split_qa("问: 你好吗\n答: 我很好")
        assert q == "问: 你好吗"
        assert a == "答: 我很好"

    def test_non_qa(self):
        q, a = _split_qa("普通段落文字")
        assert q == "普通段落文字"
        assert a is None

    def test_qa_chinese_colon(self):
        q, a = _split_qa("问：你好吗\n答：我很好")
        assert q == "问：你好吗"
        assert a == "答：我很好"

    def test_qa_english(self):
        q, a = _split_qa("Q: hello\nA: world")
        assert q == "Q: hello"
        assert a == "A: world"

    def test_qa_no_answer_stays_intact(self):
        q, a = _split_qa("问: 只有问题")
        assert q == "问: 只有问题"
        assert a is None

    def test_qa_stores_answer_in_metadata(self, local_store):
        chunks = [("faq.txt", ".txt", "问: 问题\n答: 答案", 0)]
        local_store.add_documents(chunks)
        results = local_store.search("问题", top_k=5)
        assert len(results) > 0
        assert results[0]["metadata"].get("answer") == "答: 答案"
