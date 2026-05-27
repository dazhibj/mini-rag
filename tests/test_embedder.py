import pytest

from embedder import Embedder


class TestEmbedderLocal:
    def test_local_mode(self):
        e = Embedder("local")
        assert e._model is None
        assert e.dimension == 384

    def test_embed_returns_expected_shape(self):
        e = Embedder("local")
        result = e.embed(["你好"])
        assert len(result) == 1
        assert len(result[0]) == 384

    def test_embed_multiple_texts(self):
        e = Embedder("local")
        result = e.embed(["你好", "世界"])
        assert len(result) == 2
        assert len(result[0]) == 384
        assert len(result[1]) == 384

    def test_embed_query(self):
        e = Embedder("local")
        result = e.embed_query("测试")
        assert len(result) == 384

    def test_deterministic(self):
        e = Embedder("local")
        a = e.embed(["固定文本"])
        b = e.embed(["固定文本"])
        for va, vb in zip(a[0], b[0]):
            assert va == vb
