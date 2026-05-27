from dataclasses import dataclass


@dataclass
class SearchResult:
    id: str
    content: str
    filename: str
    chunk_index: int
    score: float


def format_results(results: list[dict]) -> list[SearchResult]:
    return [
        SearchResult(
            id=r["id"],
            content=r["document"],
            filename=r["metadata"]["filename"],
            chunk_index=r["metadata"]["chunk_index"],
            score=r["distance"],
        )
        for r in results
    ]


def print_results(results: list[SearchResult]) -> None:
    if not results:
        print("No results found.")
        return
    for i, r in enumerate(results, 1):
        print(f"{'='*60}")
        print(f"Result {i}  (score: {r.score:.4f})")
        print(f"Source: {r.filename}  chunk #{r.chunk_index}")
        print(f"{'-'*60}")
        print(r.content[:500])
        print()
