# ============================
# Streamlit RAG Chat App (Groq Version - Cloud Ready)
# ============================

import streamlit as st
import os
import json
from dotenv import load_dotenv
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.chat_models import ChatOpenAI

# Load environment variables (works locally)
load_dotenv()

# -- GET GROQ API KEY (works locally + on Streamlit Cloud)
api_key = os.getenv("GROQ_API_KEY") or st.secrets.get("GROQ_API_KEY")

if not api_key:
    raise ValueError("Please set GROQ_API_KEY in your environment or Streamlit secrets.")

# -- PAGE CONFIG --
st.set_page_config(
    page_title="Nursing Mothers Chatbot",
    page_icon="🍼",
    layout="centered",
)

accent_color = "#5bc0be"

# -- SIDEBAR --
with st.sidebar:
    st.markdown("## About")
    st.write(
        "AI assistant providing evidence-based breastfeeding and infant care guidance."
    )
    st.write("---")
    st.markdown(
        "**Disclaimer:** This chatbot provides general information, not medical advice."
    )

# -- HEADER --
st.markdown(
    f"<h1 style='text-align:center;color:{accent_color};'>Nursing Mothers Chatbot</h1>",
    unsafe_allow_html=True,
)

# ============================
# LOAD MODELS & BUILD FAISS
# ============================

@st.cache_resource
def load_models_and_db():
    embedding_model = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

    # Build FAISS dynamically from JSON files
    chunk_folder = "index/data/chunks"
    all_chunks = []

    for filename in os.listdir(chunk_folder):
        if filename.endswith(".json"):
            with open(os.path.join(chunk_folder, filename), "r", encoding="utf-8") as f:
                chunks = json.load(f)
                all_chunks.extend(chunks)

    if not all_chunks:
        raise ValueError("No chunks found. Check your JSON files in index/data/chunks/")

    vector_db = FAISS.from_texts(all_chunks, embedding_model)

    llm = ChatOpenAI(
        model="llama-3.1-8b-instant",   # Groq supported model
        temperature=0.2,
        openai_api_key=api_key,
        openai_api_base="https://api.groq.com/openai/v1",
    )

    return vector_db, llm


vector_db, llm = load_models_and_db()

# ============================
# SESSION STATE
# ============================

if "history" not in st.session_state:
    st.session_state.history = []

# ============================
# PROMPT TEMPLATE
# ============================

def rag_prompt(context, question):
    return f"""
You are a breastfeeding expert. Use ONLY the provided sources.
Be clear, concise, and kind.
If unsure, say so.

Sources:
{context}

Question:
{question}

Answer:
""".strip()

# ============================
# USER INPUT
# ============================

with st.form("chat-form"):
    question = st.text_area(
        "Ask your breastfeeding or infant care question:",
        height=80,
    )
    submitted = st.form_submit_button("Ask AI")

if submitted and question:
    with st.spinner("Generating answer..."):
        docs = vector_db.similarity_search(question, k=5)
        context = "\n".join({doc.page_content.strip() for doc in docs})
        prompt = rag_prompt(context, question)

        response = llm.invoke(prompt)
        answer = response.content.strip()

        st.session_state.history.append(
            {"question": question, "answer": answer}
        )

# ============================
# DISPLAY CHAT HISTORY
# ============================

if st.session_state.history:
    st.markdown("---")
    for entry in reversed(st.session_state.history):
        st.markdown(f"**You:** {entry['question']}")
        st.markdown(f"**AI:** {entry['answer']}")
