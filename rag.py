"""Shared retrieval + prompt assembly + Claude call, used by both app.py
(the Streamlit chat) and scripts/smoke_test.py (retrieval-only, no LLM cost).
"""

import anthropic
import chromadb
from sentence_transformers import SentenceTransformer

import config

SYSTEM_PROMPT = f"""You are a study assistant helping a high-school student with their \
homework on "{config.CHAPTER_TITLE}". You will be given excerpts retrieved from that \
textbook chapter, followed by the student's question.

Rules:
1. Answer using ONLY the information in the provided excerpts. Do not use outside \
knowledge, even if you know more about the topic than what's shown.
2. If the excerpts don't contain enough information to answer, say so plainly - e.g. \
"I can't find this in the chapter excerpts provided - you may want to check another \
section or ask your teacher." Do not guess or fill gaps from general knowledge.
3. Write for a high-school student: clear, friendly, well-organized. Use short \
paragraphs or bullet points where it helps. Briefly define any economics term the \
first time you use it.
4. When you use a specific fact, figure, or definition from the excerpts, note which \
page it came from, like "(p. 12)", so the student can check the textbook.
5. Explain ideas in your own words like a tutor would - don't just paste long passages \
verbatim, unless the student explicitly asks for an exact definition or quote.
6. Keep answers focused - a few short paragraphs is usually enough for a homework \
question.
"""


def get_embedder() -> SentenceTransformer:
    return SentenceTransformer(config.EMBEDDING_MODEL_NAME)


def get_collection():
    client = chromadb.PersistentClient(path=config.CHROMA_DB_PATH)
    return client.get_or_create_collection(name=config.COLLECTION_NAME)


def get_anthropic_client() -> anthropic.Anthropic:
    return anthropic.Anthropic()


def retrieve(question: str, embedder: SentenceTransformer, collection, top_k: int = config.TOP_K) -> list[dict]:
    query_embedding = embedder.encode([question], normalize_embeddings=True).tolist()
    results = collection.query(
        query_embeddings=query_embedding,
        n_results=top_k,
        include=["documents", "metadatas", "distances"],
    )
    return [
        {"text": doc, "page": meta["page"], "distance": dist}
        for doc, meta, dist in zip(
            results["documents"][0], results["metadatas"][0], results["distances"][0]
        )
    ]


def build_user_message(question: str, chunks: list[dict]) -> str:
    context = "\n\n".join(f"[Page {c['page']}]\n{c['text']}" for c in chunks)
    return (
        "Here are excerpts from the textbook chapter that are most relevant to the "
        f"question:\n\n---\n{context}\n---\n\nStudent's question: {question}"
    )


def answer_question(question: str, chunks: list[dict], client: anthropic.Anthropic) -> str:
    response = client.messages.create(
        model=config.ANSWER_MODEL,
        max_tokens=config.ANSWER_MAX_TOKENS,
        system=SYSTEM_PROMPT,
        output_config={"effort": config.ANSWER_EFFORT},
        messages=[{"role": "user", "content": build_user_message(question, chunks)}],
    )
    return next(block.text for block in response.content if block.type == "text")
