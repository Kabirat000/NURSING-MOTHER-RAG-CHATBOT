# ============================
# Streamlit RAG Chat App (Groq Version)
# ============================

import streamlit as st
import os
from dotenv import load_dotenv
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.chat_models import ChatOpenAI

# Load environment variables
load_dotenv()

# -- CONFIG --
index_folder = "embeddings"
index_name = "breastfeeding_index"

# -- GET GROQ API KEY --
api_key = os.environ.get("GROQ_API_KEY")
if not api_key:
    raise ValueError("Please set GROQ_API_KEY in your environment or .env file.")

# -- PAGE SETUP --
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

# -- LOAD MODELS --
@st.cache_resource
def load_models_and_db():
    embedding_model = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

    vector_db = FAISS.load_local(
        index_folder,
        embeddings=embedding_model,
        index_name=index_name,
        allow_dangerous_deserialization=True,
    )

    llm = ChatOpenAI(
        model="llama-3.1-8b-instant",
        temperature=0.2,
        openai_api_key=api_key,
        openai_api_base="https://api.groq.com/openai/v1",
    )

    return vector_db, llm


vector_db, llm = load_models_and_db()

# -- SESSION HISTORY --
if "history" not in st.session_state:
    st.session_state.history = []


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


# -- UI --
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

# -- DISPLAY HISTORY --
if st.session_state.history:
    st.markdown("---")
    for entry in reversed(st.session_state.history):
        st.markdown(f"**You:** {entry['question']}")
        st.markdown(f"**AI:** {entry['answer']}")
