"""
PC RAG Chatbot — Streamlit UI (ChromaDB version)
=======================================================

PURPOSE
-------
This is the full-featured version of the chatbot UI.
It uses ChromaDB for semantic/vector search — the "proper" RAG approach
where the search understands meaning, not just keywords.

WHEN TO USE THIS FILE
---------------------
- Local development and demonstration
- When you have enough RAM (2GB+) — ChromaDB needs ~300MB after loading
- When you want to show semantic search (understands synonyms and meaning)

DO NOT USE FOR FREE CLOUD DEPLOYMENT
-------------------------------------
ChromaDB downloads a 79MB ONNX model at runtime and requires ~300MB RAM
after loading. Free cloud tiers (512MB total) crash in a loop.
Use streamlit_app_tfidf.py for free deployment instead.

HOW TO RUN LOCALLY
------------------
    streamlit run streamlit_app.py
    Open: http://localhost:8501

HOW STREAMLIT WORKS
-------------------
Streamlit re-runs the entire script from top to bottom on every user interaction
(every time someone types a message, clicks a button, etc.).

This means:
- Variables defined at the top level are recreated on every re-run
- @st.cache_resource prevents load_rag() from re-running on every re-run
- st.session_state persists data across re-runs within the same browser session

ARCHITECTURE
------------
    User types question
          │
          ▼
    streamlit_app.py
          │
          ├── load_rag() [cached] ──► reads _data/*.txt
          │                       ──► chunks into paragraphs
          │                       ──► indexes in ChromaDB (in memory)
          │
          ├── ask_rag(question)
          │       │
          │       ├── ChromaDB.query() ──► top 3 most relevant chunks
          │       └── Groq API call    ──► LLaMA 3.3 70B generates answer
          │
          └── Streamlit renders answer + sources + chunk dropdown

DEPENDENCIES
------------
- streamlit : web UI framework — builds interactive apps in pure Python
- chromadb  : in-memory vector database for semantic document search
- openai    : client library used to call Groq's OpenAI-compatible API
- python-dotenv : loads .env file for local API key management
"""

import os
import streamlit as st
from pypdf import PdfReader
from dotenv import load_dotenv
from openai import OpenAI
import chromadb
import json


# ──────────────────────────────────────
# CONFIG
# ──────────────────────────────────────

# Load .env file into environment variables.
# Locally: reads groq_api_key from .env
# On Render: reads from environment variables set in the dashboard
# python-dotenv silently does nothing if .env doesn't exist (safe in production)
load_dotenv()
GROQ_API_KEY = os.getenv("GROQ_API_KEY")

# Absolute path to the _data/ folder, resolved relative to this file.
# Using __file__ ensures this works regardless of what directory you
# run streamlit from.
DATA_FOLDER = os.path.join(os.path.dirname(__file__), "_data")

#Create a PDF extraction function
def extract_pdf_text(pdf_path):
    reader = PdfReader(pdf_path)

    text = ""

    for page in reader.pages:
        page_text = page.extract_text()

        if page_text:
            text += page_text + "\n"

    return text

# ──────────────────────────────────────
# PAGE SETUP
# ──────────────────────────────────────

# st.set_page_config() must be the FIRST Streamlit call in the script.
# It sets the browser tab title, favicon, and layout mode.
st.set_page_config(
    page_title="PC Assistant",
    page_icon="💬",
    layout="centered"   # "centered" = fixed-width column | "wide" = full width
)

st.title("PC Internal Assistant")
st.caption("Ask anything about company policies, products, or procedures.")
st.divider()   # horizontal rule for visual separation

# ──────────────────────────────────────
# RAG SETUP
# ──────────────────────────────────────

@st.cache_resource
def load_rag():
    """
    Read all .txt files from _data/, chunk by paragraph, and index in ChromaDB.

    @st.cache_resource decorator:
        - Runs this function ONCE when the app first loads
        - Caches the return value in memory
        - Returns the cached value on all subsequent Streamlit re-runs
        - Without this, load_rag() would re-run on EVERY user message,
          re-indexing 120 chunks each time — extremely slow

    ChromaDB semantic search:
        When you call collection.query(query_texts=["WFH policy"]):
        1. ChromaDB converts "WFH policy" to a 384-dimensional vector
           using the all-MiniLM-L6-v2 sentence transformer model
        2. It computes cosine similarity between that vector and all
           stored chunk vectors
        3. Returns the top N chunks with the smallest angular distance
        This finds semantically similar chunks even if they use different
        words — e.g. "remote work" would match a "work from home" query.

    Returns:
        collection   : ChromaDB collection (ready to query)
        client       : Groq API client
        chunk_id     : total number of chunks indexed (for display)
    """

    if not os.path.exists(DATA_FOLDER):
        st.error(f"_data/ folder not found at: {DATA_FOLDER}")
        st.stop()   # halt the app — nothing works without documents

    # Groq client using the OpenAI SDK.
    # Groq provides an OpenAI-compatible REST API, so we just point
    # the standard openai library at Groq's base_url instead.
    client = OpenAI(
        api_key=GROQ_API_KEY,
        base_url="https://api.groq.com/openai/v1"
    )

    # EphemeralClient = in-memory only, no disk persistence.
    # The collection is rebuilt on every app startup, which is fine
    # for a small document set.
    chroma = chromadb.EphemeralClient()

    # Clean slate — delete existing collection if app was reloaded
    try:
        chroma.delete_collection("docs")
    except Exception:
        pass

    collection = chroma.create_collection("docs")

    all_chunks    = []
    all_ids       = []
    all_metadatas = []
    chunk_id      = 0

    for filename in sorted(os.listdir(DATA_FOLDER)):
        if not filename.endswith(".txt"):
            continue

        with open(os.path.join(DATA_FOLDER, filename), "r") as f:
            text = f.read()

        # Paragraph chunking: split on blank lines (\n\n)
        # This is the simplest chunking strategy — each paragraph becomes
        # one searchable unit. More sophisticated approaches (sliding window,
        # sentence-level chunking) exist but paragraph chunking works well
        # for structured company documents.
        for para in text.strip().split("\n\n"):
            para = para.strip()

            # Skip fragments too short to be meaningful (headers, single lines)
            if len(para) < 50:
                continue

            # Skip visual dividers like "==============================="
            if para.startswith("===="):
                continue

            all_chunks.append(para)
            all_ids.append(f"chunk_{chunk_id}")

            # metadata dict stored alongside each chunk in ChromaDB
            # "source" lets us tell users which document the answer came from
            # "text" is a convenience copy (accessible from metadata directly)
            all_metadatas.append({"source": filename, "text": para})
            chunk_id += 1

    # Add all chunks in one batch — ChromaDB embeds them all here
    collection.add(
        documents=all_chunks,
        ids=all_ids,
        metadatas=all_metadatas
    )

    return collection, client, chunk_id


# Run load_rag() with a visible spinner.
# The spinner shows "Loading and indexing..." while ChromaDB downloads
# its model and processes the documents.
with st.spinner("Loading and indexing company documents..."):
    collection, groq_client, total_chunks = load_rag()

st.success(f"Ready — {total_chunks} document chunks indexed.", icon="✅")
st.divider()

# ──────────────────────────────────────
# CHAT HISTORY
# ──────────────────────────────────────

# st.session_state is a dict-like object that persists across Streamlit re-runs
# within the same browser session (until the tab is closed or the page is refreshed).
# We use it to store the full conversation so previous messages can be re-rendered.
#
# Each message is a dict:
#   {"role": "user",      "content": "What is the WFH policy?"}
#   {"role": "assistant", "content": "...", "sources": [...], "chunks": [...]}
if "messages" not in st.session_state:
    st.session_state.messages = []

# ──────────────────────────────────────
# RAG FUNCTION
# ──────────────────────────────────────

def ask_rag(question: str) -> dict:
    """
    Execute the full RAG pipeline: retrieve → augment → generate.

    Args:
        question: the user's natural language question

    Returns:
        dict with keys:
            "answer"  : str — the LLM-generated answer
            "sources" : list[str] — deduplicated source filenames
            "chunks"  : list[tuple] — [(chunk_text, source_file), ...]

    Retrieve:
        ChromaDB.query() finds the 3 most semantically similar chunks.
        It uses the same embedding model that was used to index the documents,
        so the question and chunk vectors are in the same vector space.

    Augment:
        The 3 chunks are concatenated (separated by "---") and injected
        into the prompt as "Context". The system message instructs the LLM
        to answer ONLY from this context — not from its training data.
        This is what makes it a RAG system rather than a plain chatbot.

    Generate:
        Groq runs LLaMA 3.3 70B with temperature=0.2.
        Low temperature = more deterministic, factual answers.
        High temperature (0.8+) = more creative but less accurate.
    """

    # Retrieve top 3 most semantically relevant chunks
    results = collection.query(query_texts=[question], n_results=3)

    chunks  = results["documents"][0]                           # list of 3 strings
    sources = [m["source"] for m in results["metadatas"][0]]   # list of 3 filenames

    # Build the augmented prompt
    context = "\n\n---\n\n".join(chunks)

    messages = [
        {
            "role": "system",
            "content": (
                "You are a helpful PC company assistant. "
                "Answer questions using ONLY the provided context. "
                "If the context does not contain enough information, "
                "say 'I don't have enough information to answer this.' "
                "Be concise and direct."
            )
        },
        {
            "role": "user",
            "content": f"Context:\n{context}\n\nQuestion: {question}"
        }
    ]

    # Call Groq LLM
    response = groq_client.chat.completions.create(
        model="llama-3.3-70b-versatile",
        messages=messages,
        temperature=0.2
    )

    answer = response.choices[0].message.content

    return {
        "answer":  answer,
        "sources": list(set(sources)),        # set() removes duplicates
        "chunks":  list(zip(chunks, sources)) # pair each chunk with its source file
    }

# ──────────────────────────────────────
# RENDER CHAT HISTORY
# ──────────────────────────────────────

# On every Streamlit re-run, redraw all previous messages from session_state.
# This gives the illusion of a persistent conversation even though the script
# re-runs from scratch each time.
for msg in st.session_state.messages:

    # st.chat_message() renders a chat bubble with a user or robot icon
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

        # Assistant messages also show sources and retrieved chunks
        if msg["role"] == "assistant" and msg.get("sources"):

            # Small grey text listing which documents were used
            st.caption(f"Sources: {', '.join(msg['sources'])}")

            # Collapsible section showing the raw retrieved chunks.
            # Collapsed by default so they don't clutter the conversation.
            # Students can expand to see exactly what the LLM was given.
            with st.expander("View retrieved document chunks"):
                for i, (chunk_text, source_file) in enumerate(msg["chunks"], 1):
                    st.markdown(f"**Chunk {i} — `{source_file}`**")
                    st.info(chunk_text)   # st.info() renders a blue info box

# ──────────────────────────────────────
# CHAT INPUT
# ──────────────────────────────────────

# st.chat_input() renders the text box pinned to the bottom of the page.
# It returns the submitted text (or None if the user hasn't submitted yet).
# On submit, Streamlit re-runs the script and this variable has the question.
question = st.chat_input("Ask a question about PC policies or products...")

if question:

    # Immediately render the user's message so the UI feels responsive
    with st.chat_message("user"):
        st.markdown(question)

    # Persist the user message to session_state so it shows on future re-runs
    st.session_state.messages.append({"role": "user", "content": question})

    # Render the assistant response
    with st.chat_message("assistant"):
        with st.spinner("Searching documents and generating answer..."):
            result = ask_rag(question)

        st.markdown(result["answer"])
        st.caption(f"Sources: {', '.join(result['sources'])}")

        with st.expander("View retrieved document chunks"):
            for i, (chunk_text, source_file) in enumerate(result["chunks"], 1):
                st.markdown(f"**Chunk {i} — `{source_file}`**")
                st.info(chunk_text)

    # Persist the assistant response (including sources and chunks for re-render)
    st.session_state.messages.append({
        "role":    "assistant",
        "content": result["answer"],
        "sources": result["sources"],
        "chunks":  result["chunks"]
    })