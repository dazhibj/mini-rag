from retriever import SearchResult, format_results, print_results


class TestFormatResults:
    def test_empty(self):
        assert format_results([]) == []

    def test_single_result(self):
        raw = [
            {
                "id": "doc.txt::chunk::0",
                "document": "测试内容",
                "metadata": {"filename": "doc.txt", "ext": ".txt", "chunk_index": 0},
                "distance": 0.5,
            }
        ]
        results = format_results(raw)
        assert len(results) == 1
        r = results[0]
        assert isinstance(r, SearchResult)
        assert r.id == "doc.txt::chunk::0"
        assert r.content == "测试内容"
        assert r.filename == "doc.txt"
        assert r.chunk_index == 0
        assert r.score == 0.5

    def test_multiple_results(self):
        raw = [
            {"id": f"doc{i}.txt::chunk::0", "document": f"内容{i}", "metadata": {"filename": f"doc{i}.txt", "ext": ".txt", "chunk_index": 0}, "distance": 0.1 * i}
            for i in range(3)
        ]
        results = format_results(raw)
        assert len(results) == 3
        assert results[0].score < results[1].score


class TestPrintResults:
    def test_empty(self, capsys):
        print_results([])
        captured = capsys.readouterr()
        assert "No results found" in captured.out

    def test_with_results(self, capsys):
        results = [
            SearchResult(id="a::0", content="hello", filename="a.txt", chunk_index=0, score=0.3)
        ]
        print_results(results)
        captured = capsys.readouterr()
        assert "a.txt" in captured.out
        assert "hello" in captured.out
