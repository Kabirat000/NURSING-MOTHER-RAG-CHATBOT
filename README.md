# 🍼 Nursing Mothers RAG Chatbot

A Retrieval-Augmented Generation (RAG) chatbot that provides evidence-based breastfeeding and infant care guidance using semantic search and Groq LLMs.

---

## 🚀 Overview

This project implements a full RAG pipeline:

1. Documents are chunked and embedded using HuggingFace embeddings.
2. FAISS stores semantic vectors locally.
3. User questions trigger similarity search over embedded knowledge.
4. Retrieved context is passed to a Groq LLM.
5. The model generates grounded, context-aware responses.

The system ensures responses are based only on trusted source material.

---

## 🧠 Architecture

User Question  
→ FAISS Similarity Search  
→ Retrieve Top Chunks  
→ Prompt Construction  
→ Groq LLM (Llama 3.1)  
→ Grounded Response  

---

## 🛠 Tech Stack

- Python
- Streamlit
- LangChain (v1)
- FAISS (Vector Database)
- HuggingFace Embeddings (`all-MiniLM-L6-v2`)
- Groq API (Llama 3.1)

---

## ⚙️ Setup Instructions

### 1️⃣ Clone the Repository

```bash
git clone https://github.com/Kabirat000/NURSING-MOTHER-RAG-CHATBOT.git
cd NURSING-MOTHER-RAG-CHATBOT
