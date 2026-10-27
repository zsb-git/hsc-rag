"""Extracts text from the source PDF, chunks it per page, embeds each chunk
locally, and (re)stores everything in the local Chroma collection.

Safe to re-run: chunk IDs are deterministic, so re-ingestion upserts rather
than duplicating rows.
"""

import re
import sys

import chromadb
import fitz  # pymupdf
from sentence_transformers import SentenceTransformer

import config


def extract_pages(pdf_path) -> list[tuple[int, str]]:
    doc = fitz.open(pdf_path)
    return [(i + 1, doc[i].get_text("text")) for i in range(len(doc))]


def chunk_page_text(text: str, chunk_size: int, overlap: int) -> list[str]:
    paragraphs = [p.strip() for p in text.split("\n\n") if p.strip()]

    raw_chunks: list[str] = []
    current = ""
    for para in paragraphs:
        if len(current) + len(para) + 2 <= chunk_size:
            current = f"{current}\n\n{para}" if current else para
            continue

        if current:
            raw_chunks.append(current)

        if len(para) <= chunk_size:
            current = para
        else:
            # Paragraph itself is too long - split on sentence boundaries.
            sentences = re.split(r"(?<=[.!?])\s+", para)
            sub = ""
            for sentence in sentences:
                if len(sub) + len(sentence) + 1 <= chunk_size:
                    sub = f"{sub} {sentence}".strip()
                else:
                    if sub:
                        raw_chunks.append(sub)
                    sub = sentence
            current = sub

    if current:
        raw_chunks.append(current)

    if overlap <= 0 or len(raw_chunks) <= 1:
        return raw_chunks

    overlapped = [raw_chunks[0]]
    for i in range(1, len(raw_chunks)):
        tail = raw_chunks[i - 1][-overlap:]
        overlapped.append(f"{tail} {raw_chunks[i]}")
    return overlapped


def build_chunks(pages: list[tuple[int, str]]) -> tuple[list[str], list[str], list[dict]]:
    ids, documents, metadatas = [], [], []
    for page_num, page_text in pages:
        page_chunks = chunk_page_text(page_text, config.CHUNK_SIZE, config.CHUNK_OVERLAP)
        for idx, chunk in enumerate(page_chunks):
            ids.append(f"topic1_p{page_num}_c{idx}")
            documents.append(chunk)
            metadatas.append(
                {
                    "source_file": config.PDF_PATH.name,
                    "page": page_num,
                    "chunk_index": idx,
                }
            )
    return ids, documents, metadatas


def main() -> None:
    if not config.PDF_PATH.exists():
        print(f"PDF not found at: {config.PDF_PATH}")
        print("Place the PDF there (see README) and re-run this script.")
        sys.exit(1)

    print(f"Reading {config.PDF_PATH.name} ...")
    pages = extract_pages(config.PDF_PATH)
    print(f"Extracted {len(pages)} pages.")

    ids, documents, metadatas = build_chunks(pages)
    print(f"Built {len(documents)} chunks.")

    print(f"Loading embedding model '{config.EMBEDDING_MODEL_NAME}' (first run downloads it)...")
    embedder = SentenceTransformer(config.EMBEDDING_MODEL_NAME)

    print("Embedding chunks...")
    embeddings = embedder.encode(
        documents, batch_size=32, show_progress_bar=True, normalize_embeddings=True
    ).tolist()

    client = chromadb.PersistentClient(path=config.CHROMA_DB_PATH)
    collection = client.get_or_create_collection(
        name=config.COLLECTION_NAME,
        metadata={"hnsw:space": "cosine"},
    )

    collection.upsert(ids=ids, documents=documents, embeddings=embeddings, metadatas=metadatas)

    print()
    print("--- Sample chunk ---")
    print(f"id: {ids[0]}")
    print(f"metadata: {metadatas[0]}")
    print(f"text: {documents[0][:300]}...")
    print()
    print(f"Stored {len(documents)} chunks in collection '{config.COLLECTION_NAME}' at {config.CHROMA_DB_PATH}")


if __name__ == "__main__":
    main()
