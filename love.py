import warnings
import sys
import os
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
import re
import logging
import torch
import redis
from collections import defaultdict

warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore", category=SyntaxWarning)

# =====================
# 🛠️ Configuration Setup
# =====================
class Config:
    def __init__(self):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.qdrant_path = Path("qdrant_storage")
        self.storage_path = Path("knowledge_storage")
        self.redis_conn = self._init_redis()
        
        self.embedding_model = "sentence-transformers/all-MiniLM-L6-v2"
        self.safety_model = "Hate-speech-CNERG/dehatebert-mono-english"
        self.rate_limit = 50  # Requests per hour

    def _init_redis(self):
        try:
            r = redis.Redis(host='localhost', port=6379, db=0, socket_timeout=1)
            r.ping()
            return r
        except redis.ConnectionError:
            return None

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
    if config.redis_conn:
        st.success("Redis-connected rate limiting")
    else:
        st.info("In-memory rate limiting")
    st.write(f"**Qdrant Storage:** {config.qdrant_path}")

if not groq_key:
    st.error("Please provide the Groq API key in the sidebar to proceed.")
    st.stop()

# =====================
# 📚 Knowledge Management
# =====================
class KnowledgeManager:
    def __init__(self):
        self.embeddings = HuggingFaceEmbeddings(
            model_name=config.embedding_model,
            model_kwargs={'device': config.device},
            encode_kwargs={'normalize_embeddings': True}
        )
        
        self.client = QdrantClient(
            path=str(config.qdrant_path),
            prefer_grpc=True,
            timeout=30
        )
        
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
            
            # Qdrant storage
            self.client.upsert(
                collection_name="lovebot_knowledge",
                points=[PointStruct(
                    id=point_id,
                    vector=embedding,
                    payload={"text": text, "type": source_type}
                )]
            )
            
            # SQLite storage
            with sqlite3.connect(config.storage_path / "knowledge.db") as conn:
                conn.execute(
                    "INSERT INTO knowledge_entries VALUES (?, ?, ?, ?, CURRENT_TIMESTAMP)",
                    (point_id, text, source_type, pickle.dumps(embedding)))
                
        except Exception as e:
            logging.error(f"Knowledge addition error: {str(e)}")
            st.error("Failed to add knowledge entry")

    def search_knowledge(self, query: str, limit=3):
        try:
            embedding = self.embeddings.embed_query(query)
            results = self.client.query_points(
                collection_name="lovebot_knowledge",
                query_vector=embedding,
                limit=limit,
                with_payload=True,
                with_vectors=False
            )
            return [r.payload["text"] for r in results if "text" in r.payload]
        except Exception as e:
            logging.error(f"Knowledge search error: {str(e)}")
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
            model=config.safety_model,
            device=0 if config.device == "cuda" else -1,
            torch_dtype=torch.float16 if config.device == "cuda" else torch.float32
        )
        self.searcher = SearchManager()
        self.rate_limits = defaultdict(list) if not config.redis_conn else None

    def check_rate_limit(self, user_id: str):
        if config.redis_conn:
            current = int(time.time())
            key = f"ratelimit:{user_id}"
            
            with config.redis_conn.pipeline() as pipe:
                pipe.zremrangebyscore(key, 0, current - 3600)
                pipe.zcard(key)
                pipe.zadd(key, {current: current})
                pipe.expire(key, 3600)
                results = pipe.execute()
                
            return results[1] < config.rate_limit
        else:
            current_time = time.time()
            self.rate_limits[user_id] = [
                t for t in self.rate_limits[user_id] 
                if current_time - t < 3600
            ]
            if len(self.rate_limits[user_id]) >= config.rate_limit:
                return False
            self.rate_limits[user_id].append(current_time)
            return True

    def generate_response(self, prompt: str, context: str, user_id: str):
        if not self.check_rate_limit(user_id):
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
            return output if self._is_safe(output) else "Content blocked by safety filter"
            
        except Exception as e:
            logging.error(f"Generation error: {str(e)}")
            return "I'm having trouble responding right now."

    def _is_safe(self, text: str):
        try:
            result = self.safety_checker(text[:512])[0]
            return result["label"] != "LABEL_1" or result["score"] < 0.85
        except Exception as e:
            logging.error(f"Safety check error: {str(e)}")
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
            return {"knowledge_context": "\n".join(self.knowledge.search_knowledge(query))}
        except Exception as e:
            logging.error(f"Knowledge retrieval error: {str(e)}")
            return {"knowledge_context": ""}

    def retrieve_web(self, state: BotState):
        try:
            results = self.ai.searcher.cached_search(state["messages"][-1])
            return {"web_context": "\n".join(f"• {r['body']}" for r in results)}
        except Exception as e:
            logging.error(f"Web retrieval error: {str(e)}")
            return {"web_context": ""}

    def generate(self, state: BotState):
        full_context = f"""
        KNOWLEDGE BASE:\n{state['knowledge_context']}
        WEB CONTEXT:\n{state['web_context']}
        """
        return {"response": self.ai.generate_response(
            prompt=state["messages"][-1],
            context=full_context,
            user_id=state["user_id"]
        )}

# =====================
# 💻 Streamlit Interface
# =====================
workflow_manager = WorkflowManager()

if "user_id" not in st.session_state:
    st.session_state.user_id = str(uuid.uuid4())
if "messages" not in st.session_state:
    st.session_state.messages = []

st.title("💖 LoveBot: AI Relationship Assistant")
st.write("Ask anything about love, relationships, and dating!")

# Display chat messages
for message in st.session_state.messages:
    role, text = message
    avatar = "💬" if role == "user" else "💖"
    with st.chat_message(role, avatar=avatar):
        st.write(text)

# Handle user input
user_input = st.chat_input("Type your message...")
if user_input:
    # Add user message to history
    st.session_state.messages.append(("user", user_input))
    
    # Process through workflow
    result = workflow_manager.workflow.invoke({
        "messages": [user_input],
        "knowledge_context": "",
        "web_context": "",
        "user_id": st.session_state.user_id
    })
    
    # Add AI response to history
    if response := result.get("response"):
        st.session_state.messages.append(("assistant", response))
    
    # Refresh the display
    st.rerun()

# Additional features remain the same
with st.expander("📖 Story Assistance"):
    story_prompt = st.text_area("Start your relationship story:")
    if st.button("Continue Story"):
        response = workflow_manager.ai.generate_response(
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
                results = workflow_manager.ai.searcher.cached_search(research_query)
                for result in results:
                    workflow_manager.knowledge.add_knowledge(
                        f"{result['title']}: {result['body']}",
                        "web_research"
                    )
                st.success(f"Added {len(results)} entries about {research_query}")
            except Exception as e:
                st.error("Research failed. Please try again later.")
