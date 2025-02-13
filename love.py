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
from transformers import pipeline
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

# Workaround for Streamlit watcher bug
sys.modules['torch.classes'] = None

# Configure environment
warnings.filterwarnings("ignore")
os.environ["TOKENIZERS_PARALLELISM"] = "false"]
torch.set_default_dtype(torch.float32)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler()]
)

# =====================
# 🛠️ Configuration Setup
# =====================
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
        self.safety_model = "Hate-speech-CNERG/dehatebert-mono-english"
        self.rate_limit = 50
        self.max_text_length = 1000

config = Config()

# =====================
# 🔐 Streamlit Configuration
# =====================
st.set_page_config(page_title="LoveBot", page_icon="💖", layout="wide")
st.write("""
<style>
    [data-testid="stStatusWidget"] {
        display: none;
    }
</style>
""", unsafe_allow_html=True)

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

# Validate API key
try:
    test_client = Groq(api_key=groq_key)
    test_client.chat.completions.create(
        model="mixtral-8x7b-32768",
        messages=[{"role": "user", "content": "test"}],
        max_tokens=1
    )
except Exception as e:
    st.error(f"❌ Invalid API key: {str(e)}")
    st.stop()

# =====================
# 📚 Enhanced Knowledge Management
# =====================
class KnowledgeManager:
    def __init__(self):
        self.embeddings = HuggingFaceEmbeddings(
            model_name=config.embedding_model,
            encode_kwargs={'normalize_embeddings': True}
        )
        
        self.client = QdrantClient(
            path=str(config.qdrant_path),
            prefer_grpc=False
        )
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
            logging.error(f"Collection error: {str(e)}")
            st.error("Failed to initialize knowledge base")

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
        except Exception as e:
            logging.error(f"Database error: {str(e)}")

    def _ensure_persistence(self):
        try:
            with sqlite3.connect(config.storage_path / "knowledge.db") as conn:
                cur = conn.execute("SELECT COUNT(*) FROM knowledge_entries")
                if cur.fetchone()[0] == 0:
                    self._seed_initial_data()
        except Exception as e:
            logging.error(f"Persistence check failed: {str(e)}")
            self._seed_initial_data()

    def _seed_initial_data(self):
        initial_data = [
            ("Love requires trust and communication", "seed"),
            ("Healthy relationships need boundaries", "seed"),
            ("Conflict resolution is key for lasting relationships", "seed")
        ]
        for text, source in initial_data:
            self.add_knowledge(text, source)

    def _chunk_text(self, text: str):
        paragraphs = [p.strip() for p in text.split('\n') if p.strip()]
        chunks = []
        current_chunk = []
        current_length = 0
        
        for para in paragraphs:
            if current_length + len(para) <= config.max_text_length:
                current_chunk.append(para)
                current_length += len(para)
            else:
                chunks.append("\n".join(current_chunk))
                current_chunk = [para]
                current_length = len(para)
        
        if current_chunk:
            chunks.append("\n".join(current_chunk))
            
        return chunks

    def add_knowledge(self, text: str, source_type: str):
        try:
            text = text.strip()
            if not text:
                raise ValueError("Empty text input")
                
            if len(text) > config.max_text_length * 5:
                for chunk in self._chunk_text(text):
                    self._add_single_knowledge(chunk, source_type)
            else:
                self._add_single_knowledge(text, source_type)
                
        except Exception as e:
            logging.error(f"Add knowledge error: {str(e)}")
            raise

    def _add_single_knowledge(self, text: str, source_type: str):
        embedding = self.embeddings.embed_query(text)
        point_id = str(uuid.uuid4())
        
        if isinstance(embedding, list):
            embedding = np.array(embedding)
        
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

    def search_knowledge(self, query: str, limit=3):
        try:
            embedding = self.embeddings.embed_query(query)
            if isinstance(embedding, list):
                embedding = np.array(embedding)
                
            results = self.client.search(
                collection_name="lovebot_knowledge",
                query_vector=embedding.tolist(),
                limit=limit,
                with_payload=True,
                score_threshold=0.3
            )
            return [r.payload.get("text", "") for r in results if r.payload]
        except Exception as e:
            logging.error(f"Search error: {str(e)}")
            return ["Knowledge base unavailable"]

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
            logging.error(f"Web search error: {str(e)}")
            return []

# =====================
# 🧠 Enhanced AI Service
# =====================
class AIService:
    def __init__(self):
        self.groq_client = Groq(api_key=groq_key)
        self.safety_checker = pipeline(
            "text-classification", 
            model=config.safety_model
        )
        self.searcher = SearchManager()
        self.rate_limits = defaultdict(list)
        self.max_retries = 5
        self.retry_delay = 2

    def check_rate_limit(self, user_id: str):
        current_time = time.time()
        self.rate_limits[user_id] = [t for t in self.rate_limits[user_id] if current_time - t < 3600]
        return len(self.rate_limits[user_id]) < config.rate_limit

    def generate_response(self, prompt: str, context: str, user_id: str):
        if not self.check_rate_limit(user_id):
            return "⏳ Please wait before asking more questions"
        
        for attempt in range(self.max_retries):
            try:
                response = self.groq_client.chat.completions.create(
                    model="mixtral-8x7b-32768",
                    messages=[{
                        "role": "system",
                        "content": f"Context:\n{context}\nRespond as a compassionate relationship expert."
                    }, {
                        "role": "user",
                        "content": prompt
                    }],
                    temperature=0.7,
                    max_tokens=500,
                    timeout=30
                )
                
                if not response.choices:
                    raise ValueError("Empty response from API")
                    
                output = response.choices[0].message.content
                if not output.strip() or len(output) < 50:
                    raise ValueError("Insufficient response content")
                    
                valid_prefixes = ("I", "You", "We", "Love", "Relationships", "It's")
                if not any(output.strip().startswith(prefix) for prefix in valid_prefixes):
                    raise ValueError("Unstructured response format")
                    
                return output if self._is_safe(output) else "🚫 Response blocked by safety filters"
            
            except Exception as e:
                logging.error(f"Generation attempt {attempt+1} failed: {str(e)}")
                time.sleep(self.retry_delay * (attempt + 1))
                
        return "⚠️ Failed to generate response after multiple attempts"

    def _is_safe(self, text: str):
        try:
            result = self.safety_checker(text[:512])[0]
            return result["label"] != "LABEL_1" and result["score"] < 0.85
        except Exception as e:
            logging.error(f"Safety error: {str(e)}")
            return False

# =====================
# 🤖 Workflow Management
# =====================
class BotState(TypedDict):
    messages: list[str]
    knowledge_context: str
    web_context: str
    user_id: str

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
        workflow.add_edge("retrieve_knowledge", "retrieve_web")
        workflow.add_edge("retrieve_web", "generate")
        workflow.add_edge("generate", END)
        
        return workflow.compile()

    def retrieve_knowledge(self, state: BotState):
        try:
            query = state["messages"][-1]
            context = self.knowledge.search_knowledge(query)
            return {"knowledge_context": "\n".join(context) if context else ""}
        except Exception as e:
            logging.error(f"Knowledge error: {str(e)}")
            return {"knowledge_context": ""}

    def retrieve_web(self, state: BotState):
        try:
            results = self.ai.searcher.cached_search(state["messages"][-1])
            return {"web_context": "\n".join(f"• {r['body']}" for r in results) if results else ""}
        except Exception as e:
            logging.error(f"Web error: {str(e)}")
            return {"web_context": ""}

    def generate(self, state: BotState):
        full_context = f"""
        KNOWLEDGE BASE:\n{state['knowledge_context']}
        WEB CONTEXT:\n{state['web_context']}
        """.strip()
        
        if not full_context:
            full_context = "No relevant context found"
            
        return {"response": self.ai.generate_response(
            prompt=state["messages"][-1],
            context=full_context,
            user_id=state["user_id"]
        )}

# =====================
# 💻 Streamlit Interface
# =====================
if "workflow_manager" not in st.session_state:
    st.session_state.workflow_manager = WorkflowManager()

if "user_id" not in st.session_state:
    st.session_state.user_id = str(uuid.uuid4())
if "messages" not in st.session_state:
    st.session_state.messages = []

st.title("💖 LoveBot: AI Relationship Assistant")

# Custom Knowledge Input
with st.expander("📥 Add Custom Knowledge"):
    custom_input = st.text_area("Enter your relationship insight:", 
                              help="Maximum 10,000 characters", 
                              key="custom_input",
                              height=200)
    if st.button("💾 Save Permanently"):
        if custom_input.strip():
            try:
                st.session_state.workflow_manager.knowledge.add_knowledge(
                    text=custom_input,
                    source_type="user"
                )
                st.success("Knowledge added to permanent storage!")
            except Exception as e:
                st.error(f"Failed to save: {str(e)}")
        else:
            st.warning("Please enter valid text to save")

# Chat Interface
for role, text in st.session_state.messages:
    avatar = "💬" if role == "user" else "💖"
    if role == "system":
        avatar = "ℹ️"
    with st.chat_message(role, avatar=avatar):
        st.write(text)

if prompt := st.chat_input("Ask about relationships..."):
    st.session_state.messages.append(("user", prompt))
    
    try:
        with st.status("💞 Analyzing relationship patterns...", expanded=True) as status:
            try:
                st.write("🔍 Searching knowledge base...")
                result = st.session_state.workflow_manager.workflow.invoke({
                    "messages": [m[1] for m in st.session_state.messages],
                    "knowledge_context": "",
                    "web_context": "",
                    "user_id": st.session_state.user_id
                })
                
                if not result.get("response") or not result["response"].strip():
                    raise ValueError("Empty AI response")
                    
                st.session_state.messages.append(("assistant", result["response"]))
                status.update(label="✅ Analysis complete", state="complete")
                
            except Exception as e:
                status.update(label="❌ Analysis failed", state="error")
                st.error(f"Processing error: {str(e)}")
                logging.error(traceback.format_exc())
                
    except Exception as fatal_error:
        st.error(f"Critical system failure: {str(fatal_error)}")
        logging.critical(traceback.format_exc())
    
    st.rerun()

# Additional Features
with st.expander("📖 Story Assistance"):
    story_prompt = st.text_area("Start your relationship story:")
    if st.button("Continue Story"):
        if story_prompt.strip():
            st.session_state.messages.append(("user", f"Continue this story: {story_prompt}"))
            try:
                with st.status("📖 Continuing your story...") as status:
                    result = st.session_state.workflow_manager.workflow.invoke({
                        "messages": [m[1] for m in st.session_state.messages],
                        "knowledge_context": "",
                        "web_context": "",
                        "user_id": st.session_state.user_id
                    })
                    st.session_state.messages.append(("assistant", result["response"]))
                    st.rerun()
            except Exception as e:
                st.error(f"Error continuing story: {str(e)}")
        else:
            st.warning("Please enter a story beginning")

with st.expander("🔍 Research Assistant"):
    research_query = st.text_input("Enter research topic:")
    if st.button("Learn About This"):
        if research_query.strip():
            try:
                with st.status("🔍 Researching..."):
                    results = st.session_state.workflow_manager.ai.searcher.cached_search(research_query)
                    if results:
                        count = len(results)
                        st.session_state.messages.append(("system", 
                            f"Researched '{research_query}', found {count} sources. Updating knowledge base."))
                        for result in results:
                            text = f"{result['title']}: {result['body'][:500]}"
                            st.session_state.workflow_manager.knowledge.add_knowledge(
                                text, "web_research"
                            )
                        st.rerun()
                    else:
                        st.session_state.messages.append(("system",
                            f"No results found for '{research_query}'"))
                        st.rerun()
            except Exception as e:
                st.error("Research failed. Please try again later.")
        else:
            st.warning("Please enter a research topic")
