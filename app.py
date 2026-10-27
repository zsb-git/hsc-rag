import os

import streamlit as st
from dotenv import load_dotenv

import config
import ingest
import rag

load_dotenv()

# On Streamlit Community Cloud the key lives in st.secrets (set via the app's
# "Secrets" panel); locally it comes from .env via load_dotenv() above.
if not os.environ.get("ANTHROPIC_API_KEY") and "ANTHROPIC_API_KEY" in st.secrets:
    os.environ["ANTHROPIC_API_KEY"] = st.secrets["ANTHROPIC_API_KEY"]

st.set_page_config(page_title="HSC Economics — Topic 1 Chat", page_icon="📘")


@st.cache_resource
def cached_embedder():
    return rag.get_embedder()


@st.cache_resource
def cached_collection():
    return rag.get_collection()


@st.cache_resource
def cached_anthropic_client():
    return rag.get_anthropic_client()


st.title("📘 HSC Economics — Topic 1 Chat")
st.caption(f"Ask questions about: {config.CHAPTER_TITLE}")

collection = cached_collection()

if collection.count() == 0:
    st.info("No study material loaded yet. Upload the chapter PDF below to get started.")
    uploaded_pdf = st.file_uploader("Upload PDF", type="pdf")
    if uploaded_pdf is not None:
        with st.spinner("Reading and indexing the PDF... this happens once."):
            pages = ingest.extract_pages_from_bytes(uploaded_pdf.getvalue())
            result = ingest.ingest_pages(pages, uploaded_pdf.name, embedder=cached_embedder())
        st.success(f"Indexed {result['chunks']} chunks from {result['pages']} pages. Loading chat...")
        st.rerun()
    st.stop()

if "messages" not in st.session_state:
    st.session_state.messages = []

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])
        if message.get("sources"):
            with st.expander("Sources"):
                for source in message["sources"]:
                    st.caption(f"Page {source['page']} (distance {source['distance']:.3f})")

if prompt := st.chat_input("Ask a question about Topic 1..."):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            try:
                chunks = rag.retrieve(prompt, cached_embedder(), cached_collection())
                answer = rag.answer_question(prompt, chunks, cached_anthropic_client())
            except Exception as exc:  # noqa: BLE001 - surface any failure to the chat UI
                answer = f"Something went wrong answering that: {exc}"
                chunks = []

        st.markdown(answer)
        if chunks:
            with st.expander("Sources"):
                for chunk in chunks:
                    st.caption(f"Page {chunk['page']} (distance {chunk['distance']:.3f})")

    st.session_state.messages.append({"role": "assistant", "content": answer, "sources": chunks})
