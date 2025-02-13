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
import sqlite3
import pickle
from functools import lru_cache
import time
from pathlib import Path
import logging
import torch
import numpy as np
from collections import defaultdict
import traceback
import sys
import re

# Configuration setup
class Config:
    def __init__(self):
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
        self.rate_limit = 50
        self.max_text_length = 1000
        self.retry_attempts = 3
        self.retry_delay = 2

config = Config()

# Streamlit configuration
st.set_page_config(page_title="LoveBot", page_icon="💖", layout="wide")
st.write("""
<style>
    [data-testid="stSidebar"] {
        background: linear-gradient(135deg, #ff758c, #ff7eb3) !important;
    }
    [data-testid="stHeader"] {
        background-color: rgba(0,0,0,0) !important;
    }
</style>
""", unsafe_allow_html=True)

# Enhanced Knowledge Management
class KnowledgeManager:
    def __init__(self):
        self.embeddings = HuggingFaceEmbeddings(
            model_name=config.embedding_model,
            encode_kwargs={'normalize_embeddings': True}
        )
        self.client = QdrantClient(path=str(config.qdrant_path), prefer_grpc=False)
        self._init_collection()
        self._init_sqlite()
        self._ensure_persistence()

    def _init_collection(self):
        try:
            if not self.client.collection_exists("lovebot_knowledge"):
                self.client.create_collection(
                    collection_name="lovebot_knowledge",
                    vectors_config=VectorParams(size=384, distance=Distance.COSINE),
                )
        except Exception as e:
            logging.error(f"Collection init error: {str(e)}")
            raise RuntimeError("Knowledge base initialization failed")

    def _init_sqlite(self):
        try:
            with sqlite3.connect(config.storage_path / "knowledge.db") as conn:
                conn.execute("""
                    CREATE TABLE IF NOT EXISTS knowledge_entries (
                        id TEXT PRIMARY KEY,
                        text TEXT UNIQUE,
                        source_type TEXT,
                        vector BLOB,
                        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                    )
                """)
                conn.commit()
        except Exception as e:
            logging.error(f"Database init error: {str(e)}")
            raise RuntimeError("Database initialization failed")

    def add_knowledge(self, text: str, source_type: str) -> bool:
        try:
            text = text.strip()
            if not text:
                raise ValueError("Empty text input")
                
            embedding = self.embeddings.embed_query(text)
            if not embedding or len(embedding) == 0:
                raise ValueError("Empty embedding generated")
                
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
                    "INSERT OR IGNORE INTO knowledge_entries VALUES (?, ?, ?, ?, CURRENT_TIMESTAMP)",
                    (point_id, text, source_type, pickle.dumps(embedding))
                )
                conn.commit()
            return True
        except Exception as e:
            logging.error(f"Add knowledge error: {str(e)}")
            return False

    def search_knowledge(self, query: str, limit: int = 3) -> list[str]:
        try:
            embedding = self.embeddings.embed_query(query)
            results = self.client.search(
                collection_name="lovebot_knowledge",
                query_vector=embedding,
                limit=limit,
                score_threshold=0.3
            )
            return [r.payload.get("text", "") for r in results if r.payload]
        except Exception as e:
            logging.error(f"Search error: {str(e)}")
            return []

# Enhanced Search Manager
class SearchManager:
    @lru_cache(maxsize=100)
    def cached_search(self, query: str, max_results: int = 2) -> list[dict]:
        try:
            with DDGS() as ddgs:
                return list(ddgs.text(query + " relationship advice", max_results=max_results))
        except Exception as e:
            logging.error(f"Search error: {str(e)}")
            return []

# AI Service with Error Handling
class AIService:
    def __init__(self):
        self.groq_client = Groq(api_key=st.session_state.groq_key)
        self.searcher = SearchManager()
        self.rate_limits = defaultdict(list)

    def check_rate_limit(self, user_id: str) -> bool:
        current_time = time.time()
        self.rate_limits[user_id] = [t for t in self.rate_limits[user_id] if current_time - t < 3600]
        return len(self.rate_limits[user_id]) < config.rate_limit

    def generate_response(self, prompt: str, context: str, user_id: str) -> str:
        if not self.check_rate_limit(user_id):
            return "⏳ Please wait before asking more questions"
        
        system_prompt = """You are a compassionate relationship expert. Provide advice using:
        1. Knowledge base context when available
        2. Web research when needed
        3. Clear source attribution
        4. Supportive, professional tone"""
        
        for attempt in range(config.retry_attempts):
            try:
                response = self.groq_client.chat.completions.create(
                    model="mixtral-8x7b-32768",
                    messages=[
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": f"Context:\n{context}\n\nQuestion: {prompt}"}
                    ],
                    temperature=0.7,
                    max_tokens=500
                )
                
                if response.choices:
                    output = response.choices[0].message.content.strip()
                    if output:
                        self.rate_limits[user_id].append(time.time())
                        return output
                
                if attempt < config.retry_attempts - 1:
                    time.sleep(config.retry_delay)
                    
            except Exception as e:
                logging.error(f"Generation error: {str(e)}")
                time.sleep(config.retry_delay * (attempt + 1))
        
        return "⚠️ Please try your question again"

# Workflow Management
class BotState(TypedDict):
    messages: list[str]
    knowledge_context: str
    web_context: str
    user_id: str
    knowledge_found: bool

class WorkflowManager:
    def __init__(self):
        self.knowledge = KnowledgeManager()
        self.ai = AIService()
        self.workflow = self._build_workflow()

    def _build_workflow(self):
        workflow = StateGraph(BotState)
        workflow.add_node("retrieve_knowledge", self.retrieve_knowledge)
        workflow.add_node("retrieve_web", self.retrieve_web)
        workflow.add_node("generate", self.generate)
        
        workflow.set_entry_point("retrieve_knowledge")
        workflow.add_conditional_edges(
            "retrieve_knowledge",
            self.decide_web_fallback,
            {"web_fallback": "retrieve_web", "direct_response": "generate"}
        )
        workflow.add_edge("retrieve_web", "generate")
        workflow.add_edge("generate", END)
        
        return workflow.compile()

    def decide_web_fallback(self, state: BotState) -> str:
        return "web_fallback" if not state.get("knowledge_found") else "direct_response"

    def retrieve_knowledge(self, state: BotState) -> dict:
        try:
            query = state["messages"][-1]
            context = self.knowledge.search_knowledge(query)
            return {"knowledge_context": "\n".join(context), "knowledge_found": bool(context)}
        except Exception as e:
            logging.error(f"Knowledge retrieval error: {str(e)}")
            return {"knowledge_context": "", "knowledge_found": False}

    def retrieve_web(self, state: BotState) -> dict:
        try:
            results = self.ai.searcher.cached_search(state["messages"][-1], max_results=5)
            return {"web_context": "\n".join(f"• {r['body']}" for r in results) if results else ""}
        except Exception as e:
            logging.error(f"Web retrieval error: {str(e)}")
            return {"web_context": ""}

    def generate(self, state: BotState) -> dict:
        context = []
        if state.get("knowledge_context"):
            context.append(f"KNOWLEDGE BASE:\n{state['knowledge_context']}")
        if state.get("web_context"):
            context.append(f"WEB CONTEXT:\n{state['web_context']}")
            
        response = self.ai.generate_response(
            state["messages"][-1],
            "\n\n".join(context) or "No specific context available",
            state["user_id"]
        )
        return {"response": response}

# Streamlit Interface
if "groq_key" not in st.session_state:
    st.session_state.groq_key = None

with st.sidebar:
    st.header("🔐 Configuration")
    groq_key = st.text_input("Groq API Key:", type="password")
    st.markdown("[Get Groq Key](https://console.groq.com/keys)")
    
    if groq_key:
        try:
            test_client = Groq(api_key=groq_key)
            test_client.chat.completions.create(
                model="mixtral-8x7b-32768",
                messages=[{"role": "user", "content": "test"}],
                max_tokens=1
            )
            st.session_state.groq_key = groq_key
        except Exception as e:
            st.error(f"❌ Invalid API key: {str(e)}")
            st.stop()

if not st.session_state.groq_key:
    st.error("Please provide a valid Groq API key")
    st.stop()

if "workflow_manager" not in st.session_state:
    st.session_state.workflow_manager = WorkflowManager()

if "user_id" not in st.session_state:
    st.session_state.user_id = str(uuid.uuid4())

if "messages" not in st.session_state:
    st.session_state.messages = []

st.title("💖 LoveBot: AI Relationship Assistant")

# Custom Knowledge Input
with st.expander("📥 Add Custom Knowledge"):
    custom_input = st.text_area(
        "Relationship insight:",
        help="Share your relationship wisdom (max 10,000 chars)",
        max_chars=10000,
        key="custom_input"
    )
    
    if st.button("💾 Save", key="custom_save_button", use_container_width=True):
        if custom_input.strip():
            try:
                success = st.session_state.workflow_manager.knowledge.add_knowledge(
                    text=custom_input,
                    source_type="user"
                )
                st.session_state.custom_input = ""
                st.success("✅ Knowledge saved!" if success else "❌ Save failed")
            except Exception as e:
                st.error(f"Error: {str(e)}")

# Chat Interface
for role, text in st.session_state.messages:
    st.chat_message(role, avatar="👤" if role == "user" else "💖").write(text)

if prompt := st.chat_input("Ask about relationships..."):
    st.session_state.messages.append(("user", prompt))
    
    with st.status("💞 Processing...", expanded=True) as status:
        try:
            result = st.session_state.workflow_manager.workflow.invoke({
                "messages": [m[1] for m in st.session_state.messages],
                "knowledge_context": "",
                "web_context": "",
                "user_id": st.session_state.user_id
            })
            
            if result.get("response"):
                st.session_state.messages.append(("assistant", result["response"]))
            else:
                st.error("Failed to generate response")
                
        except Exception as e:
            st.error(f"Error: {str(e)}")
            logging.error(traceback.format_exc())
            
    st.rerun()
