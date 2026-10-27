"""Retrieval-only sanity check - no LLM call, no cost.

Usage:
    python scripts/smoke_test.py "What is a trade bloc?"
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import rag  # noqa: E402


def main() -> None:
    if len(sys.argv) < 2:
        print('Usage: python scripts/smoke_test.py "your question here"')
        sys.exit(1)

    question = " ".join(sys.argv[1:])
    print(f"Question: {question}\n")

    embedder = rag.get_embedder()
    collection = rag.get_collection()
    chunks = rag.retrieve(question, embedder, collection)

    if not chunks:
        print("No chunks returned - has ingest.py been run yet?")
        return

    for i, chunk in enumerate(chunks, start=1):
        print(f"--- Result {i} | page {chunk['page']} | distance {chunk['distance']:.3f} ---")
        print(chunk["text"][:400])
        print()


if __name__ == "__main__":
    main()
