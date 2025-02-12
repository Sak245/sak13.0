import warnings
import sys
import os
import tempfile
import streamlit as st
from langgraph.graph import StateGraph, END
from qdrant_client import QdrantClient
from qdrant_client.http.models import PointStruct, VectorParams, Distance
from langchain_huggingface import HuggingFaceEmbeddings
from groq import Groq
import uuid
from typing import TypedDict
from duckduckgo_search import DDGS
from transformers import pipeline
import sqlite3
import pickle
from functools import lru_cache
import time
from pathlib import Path
import logging

# Suppress warnings and configure environment
warnings.filterwarnings("ignore")
os.environ["TOKENIZERS_PARALLELISM"] = "false"

# Initialize Torch first
import torch
torch._C._set_default_dtype(torch.float32)

# =====================
# 🛠️ Configuration Setup
# =====================
class Config:
    def __init__(self):
        # Streamlit Cloud compatibility
        if 'HOSTNAME' in os.environ and 'streamlit' in os.environ['HOSTNAME']:
            base_dir = Path(tempfile.mkdtemp())
            self.qdrant_path = base_dir / "qdrant_storage"
            self.storage_path = base_dir / "knowledge_storage"
        else:
            self.qdrant_path = Path("qdrant_storage")
            self.storage_path = Path("knowledge_storage")
        
        self.qdrant_path.mkdir(parents=True, exist_ok=True)
        self.storage_path.mkdir(parents=True, exist_ok=True)
        
        self.device = "cpu"
        self.embedding_model = "sentence-transformers/all-MiniLM-L6-v2"
        self.safety_model = "Hate-speech-CNERG/dehatebert-mono-english"
        self.rate_limit = 50

config = Config()

# =====================
# 🔐 Streamlit Configuration
# =====================
st.set_page_config(page_title="LoveBot", page_icon="💖", layout="wide")

with st.sidebar:
    st.header("🔐 Configuration")
    groq_key = st.text_input("Enter Groq API Key:", type="password")
    st.markdown("[Get Groq Key](https://console.groq.com/keys)")
    
    st.header("📊 System Status")
    st.write(f"**Processing Device:** {config.device.upper()}")
    st.write(f"**Storage Location:** {config.storage_path}")

if not groq_key:
    st.error("Please provide the Groq API key to proceed.")
    st.stop()

# =====================
# 📚 Knowledge Management
# =====================
class KnowledgeManager:
    def __init__(self):
        self.embeddings = HuggingFaceEmbeddings(
            model_name=config.embedding_model,
            encode_kwargs={'normalize_embeddings': True}
        )
        
        self.client = QdrantClient(location=str(config.qdrant_path))
        self._init_collection()
        self._init_sqlite()

    def _init_collection(self):
        if not self.client.collection_exists("lovebot_knowledge"):
            self.client.create_collection(
                collection_name="lovebot_knowledge",
                vectors_config=VectorParams(size=384, distance=Distance.COSINE),
            )

    def _init_sqlite(self):
        with sqlite3.connect(config.storage_path / "knowledge.db") as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS knowledge_entries (
                    id TEXT PRIMARY KEY,
                    text TEXT,
                    source_type TEXT,
                    vector BLOB,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            """)

    def add_knowledge(self, text: str, source_type: str):
        try:
            embedding = self.embeddings.embed_query(text)
            point_id = str(uuid.uuid4())
            
            self.client.upsert(
                collection_name="lovebot_knowledge",
                points=[PointStruct(
                    id=point_id,
                    vector=embedding,
                    payload={"text": text}
                )]
            )
            
            with sqlite3.connect(config.storage_path / "knowledge.db") as conn:
                conn.execute(
                    "INSERT INTO knowledge_entries VALUES (?, ?, ?, ?, CURRENT_TIMESTAMP)",
                    (point_id, text, source_type, pickle.dumps(embedding))
                )
        except Exception as e:
            logging.error(f"Knowledge addition error: {str(e)}")
            st.error("Failed to add knowledge")

    def search_knowledge(self, query: str, limit=3):
        try:
            embedding = self.embeddings.embed_query(query)
            results = self.client.search(
                collection_name="lovebot_knowledge",
                query_vector=embedding,
                limit=limit,
                with_payload=True
            )
            return [r.payload["text"] for r in results]
        except Exception as e:
            logging.error(f"Search error: {str(e)}")
            return []

# =====================
# 🔍 Search Management
# =====================
class SearchManager:
    @lru_cache(maxsize=100)
    def cached_search(self, query: str):
        try:
            with DDGS() as ddgs:
                return ddgs.text(query, max_results=2)
        except Exception as e:
            logging.error(f"Search error: {str(e)}")
            return []

# =====================
# 🧠 AI Service
# =====================
class AIService:
    def __init__(self):
        self.groq_client = Groq(api_key=groq_key)
        self.safety_checker = pipeline(
            "text-classification", 
            model=config.safety_model
        )
        self.searcher = SearchManager()
        self.rate_limits = dict()

    def generate_response(self, prompt: str, context: str, user_id: str):
        if self.rate_limits.get(user_id, 0) >= config.rate_limit:
            return "Rate limit exceeded"
        
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
            self.rate_limits[user_id] = self.rate_limits.get(user_id, 0) + 1
            return output if "harm" not in output.lower() else "Content blocked"
            
        except Exception as e:
            logging.error(f"Error: {str(e)}")
            return "I'm having trouble responding"

# =====================
# 💻 Streamlit Interface
# =====================
class WorkflowManager:
    def __init__(self):
        self.knowledge = KnowledgeManager()
        self.ai = AIService()

    def process_query(self, prompt: str, user_id: str):
        knowledge = "\n".join(self.knowledge.search_knowledge(prompt))
        results = self.ai.searcher.cached_search(prompt)
        web_context = "\n".join(f"• {r['body']}" for r in results)
        
        return self.ai.generate_response(
            prompt=prompt,
            context=f"Knowledge:\n{knowledge}\nWeb Results:\n{web_context}",
            user_id=user_id
        )

workflow = WorkflowManager()

if "user_id" not in st.session_state:
    st.session_state.user_id = str(uuid.uuid4())
if "messages" not in st.session_state:
    st.session_state.messages = []

st.title("💖 LoveBot: Relationship Expert")

for role, text in st.session_state.messages:
    with st.chat_message(role, avatar="💬" if role == "user" else "💖"):
        st.write(text)

if prompt := st.chat_input("Ask about relationships..."):
    st.session_state.messages.append(("user", prompt))
    response = workflow.process_query(prompt, st.session_state.user_id)
    st.session_state.messages.append(("assistant", response))
    st.rerun()
