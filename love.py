import streamlit as st
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langgraph.graph import StateGraph, END
from transformers import pipeline
from duckduckgo_search import DDGS
from groq import Groq
import requests
from bs4 import BeautifulSoup
import re
import uuid
from typing import TypedDict

# =====================
# 🔑 Sidebar Configuration
# =====================
st.set_page_config(page_title="LoveBot", page_icon="💖", layout="wide")

with st.sidebar:
    st.header("🔐 API Configuration")
    groq_key = st.text_input("Enter Groq API Key:", type="password")
    st.markdown("[Get Groq Key](https://console.groq.com/keys)")
    
if not groq_key:
    st.error("Please provide the Groq API key in the sidebar to proceed.")
    st.stop()

# =====================
# 1. Enhanced Book Summary Gathering
# =====================
def get_book_summary(title, author):
    """Fetch book summaries with improved parsing and fallback"""
    try:
        with DDGS() as ddgs:
            results = ddgs.text(f"{title} by {author} book summary", max_results=3)
            
        summary = ""
        for result in results:
            summary += f"{result['title']}: {result['body']}\n\n"
            if len(summary) > 1500:
                break

        return re.sub(r'\s+', ' ', summary).strip()[:1500] or f"Key concepts from {title} about relationships"
    
    except Exception as e:
        st.error(f"Error fetching summary: {str(e)}")
        return f"General information about {title}"

# =====================
# 2. Vector Store Setup
# =====================
books = [
    ("Nonviolent Communication", "Marshall B. Rosenberg"),
    ("The Seven Principles for Making Marriage Work", "John Gottman"),
    ("Attached", "Amir Levine"),
    ("The 5 Love Languages", "Gary Chapman"),
    ("Hold Me Tight", "Sue Johnson")
]

text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=500,
    chunk_overlap=100,
    separators=["\n\n", "\n", ". ", "! ", "? ", ", "]
)

docs = []
metadatas = []

for title, author in books:
    summary = get_book_summary(title, author)
    if summary:
        chunks = text_splitter.split_text(summary)
        docs.extend(chunks)
        metadatas.extend([{
            "source": title,
            "author": author,
            "type": "book_summary"
        } for _ in chunks])

embedding_function = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2",
    model_kwargs={'device': 'cpu'}
)

vector_store = Chroma.from_texts(
    texts=docs,
    metadatas=metadatas,
    embedding=embedding_function,
    persist_directory="./chroma_db"
)

# =====================
# 3. Safety System
# =====================
safety_classifier = pipeline(
    "text-classification", 
    model="Hate-speech-CNERG/dehatebert-mono-english"
)

def safety_check(text: str) -> bool:
    """Enhanced safety check with error handling"""
    try:
        result = safety_classifier(text[:512])[0]
        return result["label"] != "LABEL_1" or result["score"] < 0.85
    except Exception as e:
        st.error(f"Safety check error: {str(e)}")
        return "harm" not in text.lower()

# =====================
# 4. Groq Integration
# =====================
class LoveBot:
    def __init__(self, api_key):
        self.client = Groq(api_key=api_key)
    
    def generate(self, prompt: str, context: str) -> str:
        """Generate response with Groq's API"""
        try:
            response = self.client.chat.completions.create(
                model="mixtral-8x7b-32768",
                messages=[{
                    "role": "system",
                    "content": f"You're a relationship expert. Use this context:\n{context}"
                }, {
                    "role": "user",
                    "content": prompt
                }],
                temperature=0.7,
                max_tokens=500
            )
            return response.choices[0].message.content
        except Exception as e:
            st.error(f"Groq API error: {str(e)}")
            return "I'm having trouble responding right now."

bot = LoveBot(groq_key)

# =====================
# 5. LangGraph Workflow
# =====================
class BotState(TypedDict):
    messages: list
    book_context: str
    web_context: str

def retrieve_book_context(state: BotState):
    """Retrieve context from vector store"""
    try:
        docs = vector_store.similarity_search(state["messages"][-1], k=3)
        return {"book_context": "\n".join([d.page_content for d in docs])}
    except Exception as e:
        st.error(f"Vector search error: {str(e)}")
        return {"book_context": ""}

def retrieve_web_context(state: BotState):
    """Retrieve web context with DuckDuckGo"""
    try:
        with DDGS() as ddgs:
            results = ddgs.text(state["messages"][-1], max_results=3)
        return {"web_context": "\n".join([f"{r['title']}: {r['body']}" for r in results])}
    except Exception as e:
        st.error(f"Web search error: {str(e)}")
        return {"web_context": ""}

def generate_response(state: BotState):
    """Generate final response"""
    context = f"""BOOK KNOWLEDGE:
    {state['book_context']}
    
    WEB CONTEXT:
    {state['web_context']}
    """
    
    response = bot.generate(state["messages"][-1], context)
    
    if not safety_check(response):
        return {"response": "I cannot provide advice on that topic."}
    
    return {"response": response}

workflow = StateGraph(BotState)
workflow.add_node("get_books", retrieve_book_context)
workflow.add_node("get_web", retrieve_web_context)
workflow.add_node("generate", generate_response)

workflow.set_entry_point("get_books")
workflow.add_edge("get_books", "get_web")
workflow.add_edge("get_web", "generate")
workflow.add_edge("generate", END)

app = workflow.compile()

# =====================
# 💖 Streamlit UI
# =====================
st.title("💞 OpenLoveBot - Relationship Advisor")

if "messages" not in st.session_state:
    st.session_state.messages = []

for msg in st.session_state.messages:
    role_icon = "💬" if msg["role"] == "user" else "💖"
    st.chat_message(msg["role"], avatar=role_icon).write(msg["content"])

if prompt := st.chat_input("Ask about relationships..."):
    st.session_state.messages.append({"role": "user", "content": prompt})
    
    result = app.invoke({
        "messages": [prompt],
        "book_context": "",
        "web_context": ""
    })
    
    response = result.get("response", "I'm having trouble responding right now.")
    st.session_state.messages.append({"role": "assistant", "content": response})
    st.rerun()

# =====================
# 📚 Requirements
# =====================
# Requirements.txt:
# streamlit
# langchain-chroma
# sentence-transformers
# duckduckgo-search
# groq
# transformers
# beautifulsoup4
# requests
# langgraph
