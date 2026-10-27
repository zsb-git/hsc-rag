import streamlit as st
from dotenv import load_dotenv

import config
import rag

load_dotenv()

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
