import os

from utils.pdf_loader import extract_text_from_pdf
from utils.text_splitter import split_text
from utils.embeddings import load_embedding_model
from utils.retriever import retrieve_relevant_chunks
from utils.llm import load_llm
from utils.prompts import build_prompt

from utils.vector_store import (
    create_faiss_index,
    save_faiss_index,
    load_faiss_index,
    save_chunks,
    load_chunks
)


class RAGPipeline:

    def __init__(self):

        self.embedding_model = load_embedding_model()
        self.llm = load_llm()

        self.index = None
        self.chunks = None

    # ==========================================
    # PROCESS MULTIPLE PDFS
    # ==========================================

    def process_pdfs(self, pdf_paths):

        all_chunks = []

        for pdf_path in pdf_paths:

            text = extract_text_from_pdf(
                pdf_path
            )

            chunks = split_text(
                text
            )

            all_chunks.extend(
                chunks
            )

        self.chunks = all_chunks

        embeddings = (
            self.embedding_model.encode(
                self.chunks
            )
        )

        self.index = create_faiss_index(
            embeddings
        )

        # Save Vector Store

        os.makedirs(
            "vectorstore",
            exist_ok=True
        )

        save_faiss_index(
            self.index,
            "vectorstore/faiss_index.bin"
        )

        save_chunks(
            self.chunks,
            "vectorstore/chunks.pkl"
        )

    # ==========================================
    # LOAD SAVED VECTOR STORE
    # ==========================================

    def load_vectorstore(self):

        self.index = load_faiss_index(
            "vectorstore/faiss_index.bin"
        )

        self.chunks = load_chunks(
            "vectorstore/chunks.pkl"
        )

    # ==========================================
    # QUESTION ANSWERING
    # ==========================================

    def answer_question(self, question):

        if self.index is None:

            return {
                "answer": "Please upload and process PDF files first.",
                "sources": []
            }

        # Retrieve Relevant Chunks

        retrieved_chunks = retrieve_relevant_chunks(
            query=question,
            model=self.embedding_model,
            index=self.index,
            chunks=self.chunks,
            top_k=5
        )

        # Build Context

        context = "\n\n".join(
            retrieved_chunks
        )

        # Create Prompt

        prompt = build_prompt(
            context=context,
            question=question
        )

        try:

            response = self.llm.invoke(
                prompt
            )

            answer = response.content

        except Exception as e:

            error_message = str(e)

            # Gemini Free Tier Limit

            if "RESOURCE_EXHAUSTED" in error_message:

                answer = (
                    "⚠️ Gemini API quota exceeded.\n\n"
                    "Showing the most relevant retrieved content instead:\n\n"
                    f"{retrieved_chunks[0]}"
                )

            else:

                answer = (
                    "⚠️ Error generating answer.\n\n"
                    f"{error_message}"
                )

        return {
            "answer": answer,
            "sources": retrieved_chunks
        }