import streamlit as st
from langgraph.graph import StateGraph, END
from qdrant_client import QdrantClient
from qdrant_client.http.models import PointStruct, VectorParams, Distance
from langchain_community.embeddings.huggingface import HuggingFaceEmbeddings
from groq import Groq
import uuid
from typing import TypedDict
from duckduckgo_search import DDGS
from transformers import pipeline
import sqlite3
import pickle
from functools import lru_cache
from collections import defaultdict
import time
from pathlib import Path
import re
import logging

# Configure logging
logging.basicConfig(filename='app.log', level=logging.ERROR)

# =====================
# 🔑 User Configuration
# =====================
st.set_page_config(page_title="LoveBot", page_icon="💖", layout="wide")

with st.sidebar:
    st.header("🔐 Configuration")
    groq_key = st.text_input("Enter Groq API Key:", type="password")
    st.markdown("[Get Groq Key](https://console.groq.com/keys)")

if not groq_key:
    st.error("Please provide the Groq API key in the sidebar to proceed.")
    st.stop()

# =====================
# 📚 Enhanced Knowledge Base
# =====================
class KnowledgeManager:
    def __init__(self, storage_dir="./storage"):
        self.storage_dir = Path(storage_dir)
        self.storage_dir.mkdir(exist_ok=True)
        
        self.embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
        self.client = QdrantClient(":memory:")
        self.collection_name = "lovebot_knowledge"
        
        self._init_vector_db()
        self._init_sqlite()
        self._load_persistent_data()

    def _init_vector_db(self):
        if self.client.collection_exists(self.collection_name):
            self.client.delete_collection(self.collection_name)
            
        self.client.create_collection(
            collection_name=self.collection_name,
            vectors_config=VectorParams(size=384, distance=Distance.COSINE),
        )

    def _init_sqlite(self):
        with sqlite3.connect(self.storage_dir / "knowledge.db") as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS knowledge_entries (
                    id TEXT PRIMARY KEY,
                    text TEXT,
                    source_type TEXT,
                    vector BLOB,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            """)

    def _load_persistent_data(self):
        with sqlite3.connect(self.storage_dir / "knowledge.db") as conn:
            rows = conn.execute("SELECT id, text, vector FROM knowledge_entries").fetchall()
            for id_, text, vector_blob in rows:
                self.client.upsert(
                    collection_name=self.collection_name,
                    points=[PointStruct(
                        id=id_,
                        vector=pickle.loads(vector_blob),
                        payload={"text": text}
                    )]
                )

    def add_knowledge(self, text: str, source_type: str):
        try:
            embedding = self.embeddings.embed_query(text)
            point_id = str(uuid.uuid4())
            
            # Add to vector DB
            self.client.upsert(
                collection_name=self.collection_name,
                points=[PointStruct(
                    id=point_id,
                    vector=embedding,
                    payload={"text": text, "type": source_type}
                )]
            )
            
            # Add to SQLite
            with sqlite3.connect(self.storage_dir / "knowledge.db") as conn:
                conn.execute(
                    "INSERT INTO knowledge_entries (id, text, source_type, vector) VALUES (?, ?, ?, ?)",
                    (point_id, text, source_type, pickle.dumps(embedding))
                )
        except Exception as e:
            logging.error(f"Error adding knowledge: {str(e)}")
            st.error("Failed to add knowledge to database")

    def search_knowledge(self, query: str, limit=3):
        try:
            embedding = self.embeddings.embed_query(query)
            results = self.client.search(
                collection_name=self.collection_name,
                query_vector=embedding,
                limit=limit,
                with_payload=True,
            )
            return [r.payload["text"] for r in results if "text" in r.payload]
        except Exception as e:
            logging.error(f"Search error: {str(e)}")
            return []

# =====================
# 🧠 AI Core Components
# =====================
class AIService:
    def __init__(self):
        self.groq_client = Groq(api_key=groq_key)
        self.safety_checker = pipeline(
            "text-classification", 
            model="Hate-speech-CNERG/dehatebert-mono-english"
        )
        self.rate_limiter = defaultdict(list)
        self.RATE_LIMIT = 50  # Requests per hour

    def generate_response(self, prompt: str, context: str, user_id: str):
        if not self._check_rate_limit(user_id):
            return "Rate limit exceeded. Please try again later."
        
        try:
            response = self.groq_client.chat.completions.create(
                model="mixtral-8x7b-32768",
                messages=[{
                    "role": "system",
                    "content": f"Context:\n{context}\nRespond as a relationship expert."
                }, {
                    "role": "user",
                    "content": prompt
                }],
                temperature=0.7,
                max_tokens=500
            )
            
            output = response.choices[0].message.content
            return output if self._is_safe(output) else "I cannot provide advice on that topic."
            
        except Exception as e:
            logging.error(f"Generation error: {str(e)}")
            return "I'm having trouble responding right now."

    def _is_safe(self, text: str):
        try:
            result = self.safety_checker(text[:512])[0]
            return result["label"] != "LABEL_1" or result["score"] < 0.85
        except Exception as e:
            logging.error(f"Safety check error: {str(e)}")
            return "harm" not in text.lower()

    def _check_rate_limit(self, user_id: str):
        current_time = time.time()
        self.rate_limiter[user_id] = [
            t for t in self.rate_limiter[user_id] 
            if current_time - t < 3600
        ]
        if len(self.rate_limiter[user_id]) >= self.RATE_LIMIT:
            return False
        self.rate_limiter[user_id].append(current_time)
        return True

# =====================
# 🤖 Application Workflow
# =====================
class BotState(TypedDict):
    messages: list[str]
    knowledge_context: str
    web_context: str
    user_id: str

def create_workflow():
    workflow = StateGraph(BotState)
    
    def retrieve_knowledge(state: BotState):
        try:
            query = state["messages"][-1]
            return {"knowledge_context": "\n".join(knowledge_manager.search_knowledge(query))}
        except Exception as e:
            logging.error(f"Knowledge retrieval error: {str(e)}")
            return {"knowledge_context": ""}

    def retrieve_web(state: BotState):
        try:
            with DDGS() as ddgs:
                results = ddgs.text(state["messages"][-1], max_results=2)
            return {"web_context": "\n".join(f"• {r['body']}" for r in results)}
        except Exception as e:
            logging.error(f"Web retrieval error: {str(e)}")
            return {"web_context": ""}

    def generate_response(state: BotState):
        full_context = f"""
        KNOWLEDGE BASE:\n{state['knowledge_context']}
        WEB CONTEXT:\n{state['web_context']}
        """
        response = ai_service.generate_response(
            prompt=state["messages"][-1],
            context=full_context,
            user_id=state["user_id"]
        )
        return {"response": response}

    workflow.add_node("retrieve_knowledge", retrieve_knowledge)
    workflow.add_node("retrieve_web", retrieve_web)
    workflow.add_node("generate", generate_response)

    workflow.set_entry_point("retrieve_knowledge")
    workflow.add_edge("retrieve_knowledge", "retrieve_web")
    workflow.add_edge("retrieve_web", "generate")
    workflow.add_edge("generate", END)

    return workflow.compile()

# =====================
# 💻 Streamlit Interface
# =====================
knowledge_manager = KnowledgeManager()
ai_service = AIService()
workflow_app = create_workflow()

# Initialize session state
if "user_id" not in st.session_state:
    st.session_state.user_id = str(uuid.uuid4())
if "messages" not in st.session_state:
    st.session_state.messages = []

# Sidebar Controls
with st.sidebar:
    st.header("📊 System Status")
    remaining_calls = ai_service.RATE_LIMIT - len(ai_service.rate_limiter.get(st.session_state.user_id, []))
    st.metric("Remaining Requests", remaining_calls)

# Main Chat Interface
st.title("💞 LoveBot - Relationship Expert")

for msg in st.session_state.messages:
    role_icon = "💬" if msg["role"] == "user" else "💖"
    st.chat_message(msg["role"], avatar=role_icon).write(msg["content"])

if prompt := st.chat_input("Ask about relationships..."):
    st.session_state.messages.append({"role": "user", "content": prompt})
    
    result = workflow_app.invoke({
        "messages": [prompt],
        "knowledge_context": "",
        "web_context": "",
        "user_id": st.session_state.user_id
    })
    
    if response := result.get("response"):
        st.session_state.messages.append({"role": "assistant", "content": response})
    st.rerun()

# Additional Features
with st.expander("📖 Story Assistance"):
    story_prompt = st.text_area("Start your relationship story:")
    if st.button("Continue Story"):
        response = ai_service.generate_response(
            prompt=f"Continue this story positively: {story_prompt}",
            context="",
            user_id=st.session_state.user_id
        )
        st.write(response)

with st.expander("🔍 Research Assistant"):
    research_query = st.text_input("Enter research topic:")
    if st.button("Learn About This"):
        with st.spinner("Researching..."):
            try:
                with DDGS() as ddgs:
                    results = ddgs.text(research_query, max_results=3)
                for result in results:
                    knowledge_manager.add_knowledge(
                        f"{result['title']}: {result['body']}",
                        "web_research"
                    )
                st.success(f"Added {len(results)} entries about {research_query}")
            except Exception as e:
                st.error("Failed to complete research")

# Requirements
st.sidebar.markdown("""
**Required Packages:**
