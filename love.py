# Add at the very top before other imports
import sys
sys.modules['torch.classes'] = None  # Fix Streamlit watcher bug

# =====================
# 🛠️ Initial Configuration
# =====================
import warnings
import os
import tempfile
import torch
import transformers
import fitz  # PyMuPDF for PDF processing

# Workaround for Streamlit watcher bug
sys.modules['torch.classes'] = None
torch._dynamo.config.suppress_errors = True

# Configure transformers logging
transformers.logging.set_verbosity_error()

# =====================
# 📦 Package Imports
# =====================
import streamlit as st
from langgraph.graph import StateGraph, END
from qdrant_client import QdrantClient
from qdrant_client.http.models import PointStruct, VectorParams, Distance, Filter
from langchain_huggingface import HuggingFaceEmbeddings
from groq import Groq
import uuid
from typing import TypedDict, List
from duckduckgo_search import DDGS
from transformers import pipeline as transformers_pipeline
import sqlite3
import time
from pathlib import Path
import logging
import numpy as np
from collections import defaultdict
import traceback
from cachetools import TTLCache
import re

# =====================
# ⚙️ Configuration Setup
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
        
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.embedding_model = "sentence-transformers/all-MiniLM-L6-v2"
        self.safety_model = "Hate-speech-CNERG/dehatebert-mono-english"
        self.rate_limit = 50
        self.max_text_length = 1000
        self.pdf_chunk_size = 500  # Characters per chunk
        
        self._validate()
        
    def _validate(self):
        if self.max_text_length < 100:
            raise ValueError("max_text_length must be at least 100")

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
    .pdf-uploader section {padding: 1rem;}
</style>
""", unsafe_allow_html=True)

# =====================
# 📚 Knowledge Management (Enhanced with PDF Support)
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

    def _pdf_to_text(self, pdf_bytes: bytes) -> List[str]:
        """Extract and clean text from PDF bytes"""
        try:
            doc = fitz.open(stream=pdf_bytes, filetype="pdf")
            text = ""
            for page in doc:
                text += page.get_text("text") + "\n"
            
            # Clean text
            text = re.sub(r'\s+', ' ', text)  # Remove excessive whitespace
            text = re.sub(r'\n{3,}', '\n\n', text)  # Normalize newlines
            return self._chunk_text(text)
        except Exception as e:
            logging.error(f"PDF processing error: {str(e)}")
            return []

    def _chunk_text(self, text: str) -> List[str]:
        """Smart text chunking with overlap and structure preservation"""
        paragraphs = re.split(r'\n\s*\n', text)  # Split on paragraph breaks
        chunks = []
        current_chunk = []
        current_length = 0
        
        for para in paragraphs:
            para = para.strip()
            if not para:
                continue
                
            if current_length + len(para) <= config.max_text_length:
                current_chunk.append(para)
                current_length += len(para)
            else:
                if current_chunk:
                    chunks.append(" ".join(current_chunk))
                    current_chunk = [para]
                    current_length = len(para)
                else:
                    # Handle very long paragraphs
                    sub_paras = re.split(r'(?<=[.!?]) +', para)
                    for sub_para in sub_paras:
                        if len(sub_para) > config.max_text_length:
                            chunks.extend([sub_para[i:i+config.max_text_length] 
                                        for i in range(0, len(sub_para), config.max_text_length)])
                        else:
                            chunks.append(sub_para)
        if current_chunk:
            chunks.append(" ".join(current_chunk))
        return chunks

    def add_knowledge(self, text: str, source_type: str):
        try:
            text = text.strip()
            if not text:
                return
                
            chunks = self._chunk_text(text)
            for chunk in chunks:
                self._add_single_knowledge(chunk, source_type)
        except Exception as e:
            logging.error(f"Add knowledge error: {str(e)}")
            raise

    def add_pdf_knowledge(self, pdf_bytes: bytes, source_name: str):
        try:
            chunks = self._pdf_to_text(pdf_bytes)
            for chunk in chunks:
                self._add_single_knowledge(chunk, f"pdf:{source_name}")
            return len(chunks)
        except Exception as e:
            logging.error(f"PDF upload error: {str(e)}")
            return 0

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
                payload={"text": text, "source": source_type}
            )]
        )
        
        with sqlite3.connect(config.storage_path / "knowledge.db") as conn:
            conn.execute(
                "INSERT OR IGNORE INTO knowledge_entries VALUES (?, ?, ?, CURRENT_TIMESTAMP)",
                (point_id, text, source_type)
            )

    def search_knowledge(self, query: str, limit=3) -> List[str]:
        try:
            embedding = self.embeddings.embed_query(query)
            if isinstance(embedding, list):
                embedding = np.array(embedding)
                
            results = self.client.query_points(
                collection_name="lovebot_knowledge",
                query_vector=embedding.tolist(),
                limit=limit,
                query_filter=Filter(
                    must=[{"key": "source", "match": {"any": ["seed", "user", "pdf"]}}]
                ),
                score_threshold=0.3
            )
            return [r.payload.get("text", "") for r in results if r.payload]
        except Exception as e:
            logging.error(f"Search error: {str(e)}")
            return []

# =====================
# 🔍 Search Management
# =====================
class SearchManager:
    @lru_cache(maxsize=100)
    def cached_search(self, query: str) -> List[dict]:
        try:
            with DDGS() as ddgs:
                results = ddgs.text(query, max_results=3)
                return [{"title": r["title"], "content": r["body"]} for r in results]
        except Exception as e:
            logging.error(f"Web search error: {str(e)}")
            return []

# =====================
# 🧠 AI Service (Enhanced Safety and Reliability)
# =====================
class AIService:
    def __init__(self):
        self.groq_client = Groq(api_key=st.secrets.get("GROQ_API_KEY", ""))
        self.safety_checker = transformers_pipeline(
            "text-classification", 
            model=config.safety_model,
            device=0 if torch.cuda.is_available() else -1
        )
        self.searcher = SearchManager()
        self.rate_limits = TTLCache(maxsize=1000, ttl=3600)
        self.max_retries = 3
        self.retry_delay = 1.5

    def check_rate_limit(self, user_id: str) -> bool:
        return self.rate_limits.get(user_id, 0) < config.rate_limit

    def generate_response(self, prompt: str, context: str, user_id: str) -> str:
        if not self.check_rate_limit(user_id):
            return "⏳ Please wait before asking more questions"
        
        self.rate_limits[user_id] = self.rate_limits.get(user_id, 0) + 1
        
        for attempt in range(self.max_retries):
            try:
                response = self.groq_client.chat.completions.create(
                    model="mixtral-8x7b-32768",
                    messages=[{
                        "role": "system",
                        "content": f"""You're a compassionate relationship expert. Use this context if relevant:
                        {context}
                        If no context matches, use your general knowledge."""
                    }, {
                        "role": "user",
                        "content": prompt
                    }],
                    temperature=0.7,
                    max_tokens=500,
                    timeout=15
                )
                
                output = response.choices[0].message.content.strip()
                if not output or len(output) < 20:
                    raise ValueError("Empty response")
                    
                return output if self._is_safe(output) else "🚫 Response blocked by safety filters"
            
            except Exception as e:
                logging.error(f"Attempt {attempt+1} failed: {str(e)}")
                time.sleep(self.retry_delay * (attempt + 1))
                
        return self._fallback_response(prompt)

    def _is_safe(self, text: str) -> bool:
        try:
            chunks = [text[i:i+512] for i in range(0, len(text), 512)]
            for chunk in chunks:
                result = self.safety_checker(chunk)[0]
                if result["label"] == "LABEL_1" and result["score"] >= 0.85:
                    return False
            return True
        except Exception as e:
            logging.error(f"Safety error: {str(e)}")
            return False

    def _fallback_response(self, prompt: str) -> str:
        """Generate response using web search when LLM fails"""
        try:
            results = self.searcher.cached_search(prompt)
            if results:
                return f"I found these resources that might help:\n" + "\n".join(
                    f"- {r['title']}: {r['content'][:200]}" for r in results
                )
            return "I'm having trouble finding information. Could you rephrase your question?"
        except Exception as e:
            logging.error(f"Fallback failed: {str(e)}")
            return "I'm currently unable to respond. Please try again later."

# =====================
# 🤖 Workflow Management
# =====================
class BotState(TypedDict):
    messages: List[str]
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

    def retrieve_knowledge(self, state: BotState) -> dict:
        try:
            query = state["messages"][-1]
            context = self.knowledge.search_knowledge(query)
            return {"knowledge_context": "\n".join(context) if context else ""}
        except Exception as e:
            logging.error(f"Knowledge error: {str(e)}")
            return {"knowledge_context": ""}

    def retrieve_web(self, state: BotState) -> dict:
        try:
            results = self.ai.searcher.cached_search(state["messages"][-1])
            return {"web_context": "\n".join(
                f"• {r['title']}: {r['content'][:200]}" for r in results
            ) if results else ""}
        except Exception as e:
            logging.error(f"Web error: {str(e)}")
            return {"web_context": ""}

    def generate(self, state: BotState) -> dict:
        full_context = f"""
        KNOWLEDGE BASE:\n{state['knowledge_context'] or 'No relevant knowledge found'}
        WEB RESULTS:\n{state['web_context'] or 'No web results available'}
        """.strip()
            
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

# PDF Upload Sidebar
with st.sidebar:
    st.header("📥 Knowledge Upload")
    uploaded_file = st.file_uploader(
        "Upload relationship PDF guide",
        type=["pdf"],
        key="pdf_uploader",
        help="Max 10MB per file"
    )
    
    if uploaded_file:
        with st.spinner("Processing PDF..."):
            try:
                num_chunks = st.session_state.workflow_manager.knowledge.add_pdf_knowledge(
                    uploaded_file.getvalue(),
                    uploaded_file.name
                )
                st.success(f"Added {num_chunks} knowledge chunks from PDF!")
            except Exception as e:
                st.error(f"PDF processing failed: {str(e)}")

    st.header("🔍 Web Search Settings")
    st.checkbox("Enable real-time web search", value=True, key="web_search_enabled")

# Main Chat Interface
st.title("💖 LoveBot: AI Relationship Assistant")

for role, text in st.session_state.messages:
    avatar = "💬" if role == "user" else "💖"
    with st.chat_message(role, avatar=avatar):
        st.write(text)

if prompt := st.chat_input("Ask about relationships..."):
    st.session_state.messages.append(("user", prompt))
    
    try:
        with st.status("💞 Analyzing relationship patterns...", expanded=True) as status:
            try:
                # Execute workflow
                result = st.session_state.workflow_manager.workflow.invoke({
                    "messages": [m[1] for m in st.session_state.messages],
                    "knowledge_context": "",
                    "web_context": "",
                    "user_id": st.session_state.user_id
                })
                
                response = result.get("response", "I'm still learning about relationships. Could you ask something more specific?")
                
                # Fallback checks
                if not response.strip():
                    response = "I'm having trouble understanding. Could you clarify?"
                    
                st.session_state.messages.append(("assistant", response))
                status.update(label="✅ Analysis complete", state="complete")
                
            except Exception as e:
                st.session_state.messages.append(("assistant", "I'm having trouble responding right now. Please try again later."))
                status.update(label="❌ Analysis failed", state="error")
                logging.error(traceback.format_exc())
                
    except Exception as fatal_error:
        st.session_state.messages.append(("assistant", "Something went wrong. Let's try that again!"))
        logging.critical(traceback.format_exc())
    
    st.rerun()

# Additional Features
with st.expander("📖 Story Assistance"):
    story_prompt = st.text_area("Start your relationship story:", key="story_input")
    if st.button("Continue Story"):
        if story_prompt.strip():
            st.session_state.messages.append(("user", f"Continue this story: {story_prompt}"))
            st.rerun()

with st.expander("🔍 Research Assistant"):
    research_query = st.text_input("Enter research topic:", key="research_query")
    if st.button("Learn About This"):
        if research_query.strip():
            st.session_state.messages.append(("user", f"Research topic: {research_query}"))
            st.rerun()
