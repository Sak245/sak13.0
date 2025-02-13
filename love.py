import sys
sys.modules['torch.classes'] = None  # Critical Torch workaround (MUST BE FIRST)

import warnings
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
import re

# Configure environment
warnings.filterwarnings("ignore")
os.environ["TOKENIZERS_PARALLELISM"] = "false"
torch.set_default_dtype(torch.float32)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler()]
)

class Config:
    def __init__(self):
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

# Streamlit Configuration
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

class KnowledgeManager:
    def __init__(self):
        self.embeddings = HuggingFaceEmbeddings(
            model_name=config.embedding_model,
            encode_kwargs={'normalize_embeddings': True}
        )
        self.client = QdrantClient(path=str(config.qdrant_path))
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
            logging.error(f"Collection initialization error: {str(e)}")
            raise RuntimeError("Failed to initialize knowledge base")

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
            logging.error(f"Database initialization error: {str(e)}")
            raise RuntimeError("Failed to initialize database")

    def _ensure_persistence(self):
        """Ensure initial data exists in database"""
        try:
            with sqlite3.connect(config.storage_path / "knowledge.db") as conn:
                cur = conn.execute("SELECT COUNT(*) FROM knowledge_entries")
                if cur.fetchone()[0] == 0:
                    self._seed_initial_data()
        except Exception as e:
            logging.error(f"Persistence check failed: {str(e)}")
            self._seed_initial_data()

    def _seed_initial_data(self):
        """Add default relationship knowledge"""
        initial_data = [
            ("Healthy relationships require trust and communication", "seed"),
            ("Setting boundaries is essential for relationship health", "seed"),
            ("Effective conflict resolution involves active listening", "seed")
        ]
        for text, source in initial_data:
            try:
                self.add_knowledge(text, source)
            except Exception as e:
                logging.error(f"Failed to seed entry: {text}. Error: {str(e)}")

    def add_knowledge(self, text: str, source_type: str) -> bool:
        try:
            text = text.strip()
            if not text:
                raise ValueError("Empty text input")
            
            embedding = self.embeddings.embed_query(text)
            if isinstance(embedding, list):
                embedding = np.array(embedding)

            point_id = str(uuid.uuid4())
            self.client.upsert(
                collection_name="lovebot_knowledge",
                points=[PointStruct(
                    id=point_id,
                    vector=embedding.tolist(),
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
            logging.error(f"Knowledge search error: {str(e)}")
            return []

class SearchManager:
    @lru_cache(maxsize=100)
    def cached_search(self, query: str, max_results: int = 2) -> list[dict]:
        try:
            with DDGS() as ddgs:
                results = list(ddgs.text(
                    query + " relationship advice",
                    max_results=max_results
                ))
            return results if results else []
        except Exception as e:
            logging.error(f"Web search error: {str(e)}")
            return []

class AIService:
    def __init__(self):
        self.groq_client = Groq(api_key=st.session_state.groq_key)
        self.searcher = SearchManager()
        self.rate_limits = defaultdict(list)

    def check_rate_limit(self, user_id: str) -> bool:
        current_time = time.time()
        self.rate_limits[user_id] = [
            t for t in self.rate_limits[user_id]
            if current_time - t < 3600
        ]
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
                
                if not response.choices:
                    raise ValueError("Empty API response")
                    
                output = response.choices[0].message.content.strip()
                if not output:
                    raise ValueError("Empty content in response")
                
                self.rate_limits[user_id].append(time.time())
                return output
                
            except Exception as e:
                logging.error(f"Generation attempt {attempt+1} failed: {str(e)}")
                if attempt < config.retry_attempts - 1:
                    time.sleep(config.retry_delay * (attempt + 1))
        
        return "⚠️ Please try your question again"

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
            return {
                "knowledge_context": "\n".join(context) if context else "",
                "knowledge_found": bool(context)
            }
        except Exception as e:
            logging.error(f"Knowledge retrieval error: {str(e)}")
            return {"knowledge_context": "", "knowledge_found": False}

    def retrieve_web(self, state: BotState) -> dict:
        try:
            search_limit = 5 if not state.get("knowledge_found") else 2
            results = self.ai.searcher.cached_search(
                state["messages"][-1], 
                max_results=search_limit
            )
            return {
                "web_context": "\n".join(f"• {r['body']}" for r in results) if results else ""
            }
        except Exception as e:
            logging.error(f"Web retrieval error: {str(e)}")
            return {"web_context": ""}

    def generate(self, state: BotState) -> dict:
        context_sources = []
        if state.get("knowledge_context"):
            context_sources.append(f"KNOWLEDGE BASE:\n{state['knowledge_context']}")
        if state.get("web_context"):
            context_sources.append(f"WEB CONTEXT:\n{state['web_context']}")
            
        full_context = "\n\n".join(context_sources) or "No specific context available"
        
        response = self.ai.generate_response(
            prompt=state["messages"][-1],
            context=full_context,
            user_id=state["user_id"]
        )
        return {"response": response}

if "groq_key" not in st.session_state:
    st.session_state.groq_key = None

with st.sidebar:
    st.header("🔐 Configuration")
    groq_key = st.text_input("Enter Groq API Key:", type="password", key="groq_key_input")
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
            st.success("API Key validated successfully!")
        except Exception as e:
            st.error(f"❌ Invalid API key: {str(e)}")
            st.session_state.groq_key = None
            st.stop()

if not st.session_state.groq_key:
    st.error("Please provide a valid Groq API key to proceed.")
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
        "Enter your relationship insight:",
        help="Share your relationship wisdom (max 10,000 characters)",
        max_chars=10000,
        key="custom_input_widget"
    )
    
    col1, col2 = st.columns([1, 4])
    with col1:
        if st.button("💾 Save", 
                   key="custom_save_button",
                   use_container_width=True):
            if custom_input.strip():
                try:
                    success = st.session_state.workflow_manager.knowledge.add_knowledge(
                        text=custom_input,
                        source_type="user"
                    )
                    if success:
                        st.success("✅ Knowledge saved successfully!")
                        st.session_state.custom_input_widget = ""
                    else:
                        st.error("Failed to save knowledge")
                except Exception as e:
                    st.error(f"Error saving knowledge: {str(e)}")
            else:
                st.warning("Please enter valid text to save")

# Chat Interface
chat_container = st.container()
with chat_container:
    for role, text in st.session_state.messages:
        avatar = "👤" if role == "user" else "💖"
        with st.chat_message(role, avatar=avatar):
            st.write(text)

    if prompt := st.chat_input("Ask about relationships...", key="chat_input"):
        st.session_state.messages.append(("user", prompt))
        
        with st.status("💞 Processing your question...", expanded=True) as status:
            try:
                status.write("🔍 Searching knowledge base...")
                result = st.session_state.workflow_manager.workflow.invoke({
                    "messages": [m[1] for m in st.session_state.messages],
                    "knowledge_context": "",
                    "web_context": "",
                    "user_id": st.session_state.user_id
                })
                
                if result.get("response"):
                    st.session_state.messages.append(("assistant", result["response"]))
                    status.update(label="✅ Response ready", state="complete")
                else:
                    status.update(label="❌ No response generated", state="error")
                    st.error("Failed to generate response. Please try again.")
                    
            except Exception as e:
                status.update(label="❌ Error processing request", state="error")
                st.error(f"An error occurred: {str(e)}")
                logging.error(traceback.format_exc())
        st.rerun()

st.markdown("---")
st.markdown("💝 **LoveBot** - Your AI Relationship Assistant")
