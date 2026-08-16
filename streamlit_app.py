"""
PC RAG Chatbot — Streamlit UI (ChromaDB + PDF version)
=======================================================

Reads PDF documents from the _data folder, extracts their text using
pypdf, chunks the text, stores the chunks in ChromaDB, and uses Groq
to generate answers from the retrieved context.

Project structure:

PC_RAG/
│
├── streamlit_app.py
├── requirements.txt
├── .env
├── .gitignore
│
├── _data/
│   ├── HR_Handbook_v2026July01.pdf
│   └── Holiday List_2026-Bangalore.pdf
│
└── .venv/
"""

import os

import streamlit as st
from pypdf import PdfReader
from dotenv import load_dotenv
from openai import OpenAI
import chromadb


# ============================================================
# CONFIGURATION
# ============================================================

# Load variables from .env
load_dotenv()

GROQ_API_KEY = os.getenv("GROQ_API_KEY")

# Folder containing PDF documents
DATA_FOLDER = os.path.join(
    os.path.dirname(__file__),
    "_data"
)


# ============================================================
# PAGE CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="PC Assistant",
    page_icon="💬",
    layout="centered"
)

st.title("PC Internal Assistant")
st.caption("Ask anything about company policies, products, or procedures.")
st.divider()


# ============================================================
# VALIDATE CONFIGURATION
# ============================================================

if not GROQ_API_KEY:
    st.error(
        "GROQ_API_KEY is not configured. "
        "Please add GROQ_API_KEY to your .env file."
    )
    st.stop()


if not os.path.exists(DATA_FOLDER):
    st.error(
        f"_data folder not found.\n\n"
        f"Expected location:\n{DATA_FOLDER}"
    )
    st.stop()


# ============================================================
# PDF TEXT EXTRACTION
# ============================================================

def extract_pdf_text(pdf_path):
    """
    Extract text from all pages of a PDF.

    Args:
        pdf_path: Full path to the PDF file.

    Returns:
        Extracted text as a single string.
    """

    reader = PdfReader(pdf_path)

    text_parts = []

    for page_number, page in enumerate(reader.pages, start=1):

        try:
            page_text = page.extract_text()

            if page_text:
                text_parts.append(page_text)

        except Exception as e:
            print(
                f"Could not extract page {page_number} "
                f"from {pdf_path}: {e}"
            )

    return "\n\n".join(text_parts)


# ============================================================
# CHUNKING
# ============================================================

def chunk_text(text, chunk_size=1500, overlap=200):
    """
    Split extracted PDF text into overlapping chunks.

    Args:
        text: Extracted PDF text.
        chunk_size: Maximum approximate characters per chunk.
        overlap: Number of overlapping characters.

    Returns:
        List of text chunks.
    """

    text = text.strip()

    if not text:
        return []

    chunks = []

    start = 0
    text_length = len(text)

    while start < text_length:

        end = start + chunk_size

        chunk = text[start:end].strip()

        if chunk:
            chunks.append(chunk)

        # Prevent infinite loop
        if end >= text_length:
            break

        start = end - overlap

    return chunks


# ============================================================
# RAG SETUP
# ============================================================

@st.cache_resource
def load_rag():
    """
    Load PDFs, extract text, create chunks, and index them
    in an in-memory ChromaDB collection.

    Returns:
        collection
        Groq/OpenAI-compatible client
        total_chunks
    """

    # --------------------------------------------------------
    # Create Groq client
    # --------------------------------------------------------

    client = OpenAI(
        api_key=GROQ_API_KEY,
        base_url="https://api.groq.com/openai/v1"
    )

    # --------------------------------------------------------
    # Create ChromaDB in-memory client
    # --------------------------------------------------------

    chroma = chromadb.EphemeralClient()

    # --------------------------------------------------------
    # Create collection
    # --------------------------------------------------------

    collection = chroma.create_collection(
        name="docs"
    )

    # --------------------------------------------------------
    # Lists used for ChromaDB
    # --------------------------------------------------------

    all_chunks = []
    all_ids = []
    all_metadatas = []

    chunk_id = 0

    # --------------------------------------------------------
    # Read files from _data
    # --------------------------------------------------------

    files = sorted(os.listdir(DATA_FOLDER))

    pdf_files = [
        filename
        for filename in files
        if filename.lower().endswith(".pdf")
    ]

    if not pdf_files:
        st.error(
            f"No PDF files found in:\n{DATA_FOLDER}"
        )
        st.stop()

    # --------------------------------------------------------
    # Process every PDF
    # --------------------------------------------------------

    for filename in pdf_files:

        pdf_path = os.path.join(
            DATA_FOLDER,
            filename
        )

        try:

            # Extract text from PDF
            text = extract_pdf_text(pdf_path)

        except Exception as e:

            st.warning(
                f"Could not read {filename}: {e}"
            )

            continue

        # Check if PDF contains extractable text
        if not text.strip():

            st.warning(
                f"No extractable text found in {filename}. "
                f"If this is a scanned PDF, OCR may be required."
            )

            continue

        # ----------------------------------------------------
        # Chunk extracted text
        # ----------------------------------------------------

        chunks = chunk_text(
            text,
            chunk_size=1500,
            overlap=200
        )

        # ----------------------------------------------------
        # Add chunks to ChromaDB lists
        # ----------------------------------------------------

        for chunk in chunks:

            # Skip very small chunks
            if len(chunk.strip()) < 50:
                continue

            all_chunks.append(chunk)

            all_ids.append(
                f"chunk_{chunk_id}"
            )

            all_metadatas.append({
                "source": filename
            })

            chunk_id += 1

    # --------------------------------------------------------
    # Make sure we have data before collection.add()
    # --------------------------------------------------------

    if not all_chunks:

        st.error(
            "No usable text chunks were extracted from the PDFs."
        )

        st.stop()

    # --------------------------------------------------------
    # Add documents to ChromaDB
    # --------------------------------------------------------

    collection.add(
        documents=all_chunks,
        ids=all_ids,
        metadatas=all_metadatas
    )

    return collection, client, chunk_id


# ============================================================
# LOAD RAG
# ============================================================

with st.spinner(
    "Loading PDFs and indexing documents..."
):

    collection, groq_client, total_chunks = load_rag()


st.success(
    f"Ready — {total_chunks} document chunks indexed.",
    icon="✅"
)

st.divider()


# ============================================================
# CHAT HISTORY
# ============================================================

if "messages" not in st.session_state:

    st.session_state.messages = []


# ============================================================
# RAG QUESTION FUNCTION
# ============================================================

def ask_rag(question: str) -> dict:
    """
    Retrieve relevant chunks from ChromaDB and ask Groq
    to generate an answer using only those chunks.
    """

    # --------------------------------------------------------
    # Retrieve relevant chunks
    # --------------------------------------------------------

    results = collection.query(
        query_texts=[question],
        n_results=3
    )

    # --------------------------------------------------------
    # Check results
    # --------------------------------------------------------

    if not results["documents"]:

        return {
            "answer": (
                "I don't have enough information to answer this."
            ),
            "sources": [],
            "chunks": []
        }

    chunks = results["documents"][0]

    metadatas = results["metadatas"][0]

    sources = [
        metadata["source"]
        for metadata in metadatas
    ]

    # --------------------------------------------------------
    # Build context
    # --------------------------------------------------------

    context_parts = []

    for i, (chunk, source) in enumerate(
        zip(chunks, sources),
        start=1
    ):

        context_parts.append(
            f"Source: {source}\n"
            f"Content:\n{chunk}"
        )

    context = "\n\n---\n\n".join(
        context_parts
    )

    # --------------------------------------------------------
    # Prompt
    # --------------------------------------------------------

    messages = [

        {
            "role": "system",
            "content": (
                "You are a helpful internal company assistant. "
                "Answer the user's question using ONLY the "
                "provided document context. "
                "Do not use outside knowledge. "
                "If the answer is not available in the context, "
                "say exactly: "
                "'I don't have enough information to answer this.' "
                "Be concise and direct."
            )
        },

        {
            "role": "user",
            "content": (
                f"Document Context:\n\n"
                f"{context}\n\n"
                f"Question:\n{question}"
            )
        }
    ]

    # --------------------------------------------------------
    # Call Groq
    # --------------------------------------------------------

    response = groq_client.chat.completions.create(
        model="llama-3.3-70b-versatile",
        messages=messages,
        temperature=0.2
    )

    answer = response.choices[0].message.content

    # --------------------------------------------------------
    # Return result
    # --------------------------------------------------------

    return {
        "answer": answer,
        "sources": list(set(sources)),
        "chunks": list(
            zip(chunks, sources)
        )
    }


# ============================================================
# DISPLAY CHAT HISTORY
# ============================================================

for msg in st.session_state.messages:

    with st.chat_message(msg["role"]):

        st.markdown(
            msg["content"]
        )

        if (
            msg["role"] == "assistant"
            and msg.get("sources")
        ):

            st.caption(
                f"Sources: {', '.join(msg['sources'])}"
            )

            with st.expander(
                "View retrieved document chunks"
            ):

                for i, (
                    chunk_text,
                    source_file
                ) in enumerate(
                    msg["chunks"],
                    start=1
                ):

                    st.markdown(
                        f"**Chunk {i} — `{source_file}`**"
                    )

                    st.info(
                        chunk_text
                    )


# ============================================================
# CHAT INPUT
# ============================================================

question = st.chat_input(
    "Ask a question about PC policies or products..."
)


if question:

    # --------------------------------------------------------
    # Display user question
    # --------------------------------------------------------

    with st.chat_message("user"):

        st.markdown(question)

    st.session_state.messages.append({
        "role": "user",
        "content": question
    })

    # --------------------------------------------------------
    # Generate assistant answer
    # --------------------------------------------------------

    with st.chat_message("assistant"):

        with st.spinner(
            "Searching documents and generating answer..."
        ):

            result = ask_rag(question)

        st.markdown(
            result["answer"]
        )

        if result["sources"]:

            st.caption(
                f"Sources: {', '.join(result['sources'])}"
            )

        # ----------------------------------------------------
        # Show retrieved chunks
        # ----------------------------------------------------

        with st.expander(
            "View retrieved document chunks"
        ):

            for i, (
                chunk_text,
                source_file
            ) in enumerate(
                result["chunks"],
                start=1
            ):

                st.markdown(
                    f"**Chunk {i} — `{source_file}`**"
                )

                st.info(
                    chunk_text
                )

    # --------------------------------------------------------
    # Save assistant response
    # --------------------------------------------------------

    st.session_state.messages.append({

        "role": "assistant",

        "content": result["answer"],

        "sources": result["sources"],

        "chunks": result["chunks"]
    })