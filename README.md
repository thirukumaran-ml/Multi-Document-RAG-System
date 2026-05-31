# 🤖 AI Document Assistant (RAG)

A Retrieval-Augmented Generation (RAG) application that allows users to upload multiple PDF documents and ask questions using natural language.

The application combines semantic search, vector databases, and Google's Gemini LLM to provide context-aware answers directly from uploaded documents.

---

## 🚀 Features

* 📄 Multi-PDF Upload
* ✂️ Intelligent Text Chunking
* 🔍 Semantic Search using Sentence Transformers
* 🗄️ FAISS Vector Database
* 💬 ChatGPT-style Interface
* 📚 Source Citations
* ⚡ Persistent Vector Database
* 🤖 Gemini 2.5 Flash Integration
* 🎨 Streamlit User Interface

---

## 🏗️ Architecture

PDF Upload

↓

Text Extraction

↓

Text Chunking

↓

Sentence Embeddings

(all-MiniLM-L6-v2)

↓

FAISS Vector Store

↓

Retriever

↓

Gemini 2.5 Flash

↓

Answer + Sources

---

## 🛠️ Tech Stack

### Frontend

* Streamlit

### NLP & RAG

* LangChain
* Sentence Transformers
* FAISS

### LLM

* Google Gemini 2.5 Flash

### PDF Processing

* PyPDF

---

## 📂 Project Structure

```text
PDF Question Answering System (RAG)

├── app.py
├── requirements.txt
├── README.md

├── utils
│   ├── pdf_loader.py
│   ├── text_splitter.py
│   ├── embeddings.py
│   ├── vector_store.py
│   ├── retriever.py
│   ├── prompts.py
│   ├── llm.py
│   └── rag_pipeline.py

├── data
└── vectorstore
```

## ⚙️ Installation

```bash
git clone <repository-url>

cd PDF-Question-Answering-System-RAG

pip install -r requirements.txt
```

## 🔑 Environment Variables

Create a `.env` file:

```env
GOOGLE_API_KEY=your_api_key_here
```

## ▶️ Run Application

```bash
streamlit run app.py
```

---

## 📸 Application Features

* Upload multiple PDF documents
* Build a semantic knowledge base
* Ask questions in natural language
* View retrieved source chunks
* Persistent FAISS storage
* Chat history support

---

## 🎯 Future Improvements

* Hybrid Search (BM25 + Vector Search)
* Conversational Memory
* PDF Page References
* Cloud Storage Integration
* Advanced RAG Evaluation

---

## 👨‍💻 Author

Thiru Kumaran

B.Sc Statistics | M.Sc.Applied Data Science 

Built using Streamlit, Gemini, FAISS and Sentence Transformers.
