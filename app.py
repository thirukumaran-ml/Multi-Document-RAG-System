import streamlit as st
import tempfile
import os

from utils.rag_pipeline import RAGPipeline


# ==========================================
# PAGE CONFIG
# ==========================================

st.set_page_config(
    page_title="PDF Question Answering System (RAG)",
    page_icon="🤖",
    layout="wide"
)

# ==========================================
# SESSION STATE
# ==========================================

if "messages" not in st.session_state:
    st.session_state.messages = []

# ==========================================
# HEADER
# ==========================================

st.title("🤖 PDF Question Answering System (RAG)")

st.markdown("""
Ask questions across multiple PDF documents using:

- Gemini 2.5 Flash
- FAISS Vector Search
- Sentence Transformers
- Retrieval Augmented Generation (RAG)
""")

# ==========================================
# SIDEBAR
# ==========================================

with st.sidebar:

    st.header("⚙️ Controls")

    if st.button("🗑 Clear Chat"):

        st.session_state.messages = []

        st.rerun()

    if os.path.exists(
        "vectorstore/faiss_index.bin"
    ):

        if st.button(
            "⚡ Load Saved Vector DB"
        ):

            rag = RAGPipeline()

            rag.load_vectorstore()

            st.session_state.rag = rag

            st.success(
                "Vector Database Loaded!"
            )

    st.divider()

    st.header("📊 System Information")

    if "rag" in st.session_state:

        st.metric(
            "Chunks Loaded",
            len(st.session_state.rag.chunks)
        )

        st.success(
            "Vector DB Ready"
        )

        st.write(
            "Embedding Model"
        )

        st.code(
            "all-MiniLM-L6-v2"
        )

    else:

        st.warning(
            "No Vector DB Loaded"
        )

# ==========================================
# MULTI PDF UPLOAD
# ==========================================

uploaded_files = st.file_uploader(
    "Upload PDF Files",
    type=["pdf"],
    accept_multiple_files=True
)

if uploaded_files:

    st.success(
        f"{len(uploaded_files)} PDF(s) uploaded"
    )

    for pdf in uploaded_files:

        st.write(
            f"📄 {pdf.name}"
        )

    if st.button(
        "Process PDFs"
    ):

        pdf_paths = []

        for uploaded_file in uploaded_files:

            with tempfile.NamedTemporaryFile(
                delete=False,
                suffix=".pdf"
            ) as tmp_file:

                tmp_file.write(
                    uploaded_file.read()
                )

                pdf_paths.append(
                    tmp_file.name
                )

        with st.spinner(
            "Processing PDF files..."
        ):

            rag = RAGPipeline()

            rag.process_pdfs(
                pdf_paths
            )

            st.session_state.rag = rag

            st.session_state.messages = []

        st.success(
            "PDFs processed successfully!"
        )

# ==========================================
# CHAT INTERFACE
# ==========================================

if "rag" in st.session_state:

    st.divider()

    st.subheader(
        "💬 Chat with your PDFs"
    )

    # Display chat history

    for message in st.session_state.messages:

        with st.chat_message(
            message["role"]
        ):

            st.markdown(
                message["content"]
            )

    # User Input

    question = st.chat_input(
        "Ask a question about your PDFs..."
    )

    if question:

        # Store User Message

        st.session_state.messages.append(
            {
                "role": "user",
                "content": question
            }
        )

        with st.chat_message(
            "user"
        ):

            st.markdown(
                question
            )

        # Generate Answer

        with st.spinner(
            "Generating Answer..."
        ):

            result = (
                st.session_state.rag.answer_question(
                    question
                )
            )

            answer = result["answer"]

            sources = result["sources"]

        # Store Assistant Message

        st.session_state.messages.append(
            {
                "role": "assistant",
                "content": answer
            }
        )

        # Display Assistant Message

        with st.chat_message(
            "assistant"
        ):

            st.markdown(
                answer
            )

            with st.expander(
                "📚 Sources Used"
            ):

                for i, source in enumerate(
                    sources,
                    start=1
                ):

                    st.markdown(
                        f"### Source {i}"
                    )

                    preview = source[:500]

                    if len(source) > 500:
                        preview += "..."

                    st.write(
                        preview
                    )

# ==========================================
# FOOTER
# ==========================================

st.divider()

st.caption(
    "Built with Streamlit • Gemini • FAISS • Sentence Transformers"
)