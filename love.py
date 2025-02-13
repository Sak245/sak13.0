import sys
sys.modules['torch.classes'] = None  # Critical Torch workaround - MUST BE FIRST

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
        self.rate_limit = 30  # Reduced rate limit
        self.max_text_length = 1000
        self.retry_attempts = 5  # Increased retry attempts
        self.retry_delay = 3

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
            logging.error(f"Collection init error: {str(e)}")
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
            logging.error(f"Database init error: {str(e)}")
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
            ("Healthy communication is key to resolving conflicts", "seed"),
            ("Respecting boundaries builds trust in relationships", "seed"),
            ("Active listening is essential for understanding partners", "seed")
        ]
        for text, source in initial_data:
            try:
                self.add_knowledge(text, source)
            except Exception as e:
                logging.error(f"Failed to seed entry: {text}. Error: {str(e)}")

    # Rest of KnowledgeManager methods remain the same

class AIService:
    def __init__(self):
        self.groq_client = Groq(api_key=st.session_state.groq_key)
        self.searcher = SearchManager()
        self.rate_limits = defaultdict(list)
        self.last_request = 0  # Rate limiting tracking

    def check_rate_limit(self, user_id: str) -> bool:
        current_time = time.time()
        # Clear old timestamps
        self.rate_limits[user_id] = [t for t in self.rate_limits[user_id] if current_time - t < 60]
        return len(self.rate_limits[user_id]) < config.rate_limit

    def generate_response(self, prompt: str, context: str, user_id: str) -> str:
        if not self.check_rate_limit(user_id):
            return "⏳ Please wait before asking more questions"

        # Rate limiting throttle
        time_since_last = time.time() - self.last_request
        if time_since_last < 1.0:
            time.sleep(1.0 - time_since_last)
            
        system_prompt = """You are a relationship expert. Provide advice using:
        1. Available knowledge base context
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
                    max_tokens=500,
                    timeout=30  # Added timeout
                )
                
                if not response.choices:
                    raise ValueError("Empty API response")
                    
                output = response.choices[0].message.content.strip()
                if not output:
                    raise ValueError("Empty content in response")
                
                self.rate_limits[user_id].append(time.time())
                self.last_request = time.time()
                return output
                
            except Exception as e:
                logging.error(f"Attempt {attempt+1} failed: {str(e)}")
                if attempt < config.retry_attempts - 1:
                    delay = config.retry_delay * (attempt + 1)
                    time.sleep(delay)
        
        return "⚠️ Please try your question again later"


