from dataclasses import dataclass

from config import DISPLAY_MAX_CHARS


@dataclass
class SearchResult:
    id: str
    content: str
    filename: str
    chunk_index: int
    score: float
    answer: str | None = None


def format_results(results: list[dict]) -> list[SearchResult]:
    return [
        SearchResult(
            id=r["id"],
            content=r["document"],
            filename=r["metadata"]["filename"],
            chunk_index=r["metadata"]["chunk_index"],
            score=r["distance"],
            answer=r["metadata"].get("answer"),
        )
        for r in results
    ]


def print_results(results: list[SearchResult], max_chars: int = DISPLAY_MAX_CHARS) -> None:
    if not results:
        print("No results found.")
        return
    for i, r in enumerate(results, 1):
        print(f"{'='*60}")
        print(f"Result {i}  (score: {r.score:.4f})")
        print(f"Source: {r.filename}  chunk #{r.chunk_index}")
        print(f"{'-'*60}")
        if r.answer:
            print(r.content[:max_chars])
            print(r.answer[:max_chars])
        else:
            print(r.content[:max_chars])
        print()
