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
os.environ["TOKENIZERS_PARALLELISM"] = "false"]
torch.set_default_dtype(torch.float32)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler()]
)

class Config:
    def __init__(self):
        # Astra DB configuration (can be empty)
        self.astra_db_id = st.session_state.get("astra_db_id", "")
        self.astra_db_region = st.session_state.get("astra_db_region", "")
        self.astra_db_app_token = st.session_state.get("astra_db_app_token", "")
        self.astra_db_namespace = st.session_state.get("astra_db_namespace", "default")
        
        # Local storage fallback
        self.qdrant_path = Path("qdrant_storage")
        self.storage_path = Path("knowledge_storage")
        self.qdrant_path.mkdir(parents=True, exist_ok=True)
        self.storage_path.mkdir(parents=True, exist_ok=True)
        
        # Model configuration
        self.device = "cpu"
        self.embedding_model = "sentence-transformers/all-MiniLM-L6-v2"
        self.rate_limit = 30
        self.max_text_length = 1000
        self.retry_attempts = 5
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
        
        # Initialize vector DB client (Astra DB or local Qdrant)
        if all([config.astra_db_id, config.astra_db_region, config.astra_db_app_token]):
            self.client = QdrantClient(
                url=f"https://{config.astra_db_id}-{config.astra_db_region}.apps.astra.datastax.com",
                api_key=config.astra_db_app_token,
                prefer_grpc=True,
                timeout=60
            )
            self.collection_name = f"{config.astra_db_namespace}_lovebot_knowledge"
        else:
            self.client = QdrantClient(path=str(config.qdrant_path))
            self.collection_name = "lovebot_knowledge"

        self._init_collection()
        self._init_sqlite()
        self._ensure_persistence()

    def _init_collection(self):
        try:
            if not self.client.collection_exists(self.collection_name):
                self.client.create_collection(
                    collection_name=self.collection_name,
                    vectors_config=VectorParams(size=384, distance=Distance.COSINE),
                )
        except Exception as e:
            logging.error(f"Collection initialization error: {str(e)}")
            raise RuntimeError("Failed to initialize knowledge base")

    # Rest of the KnowledgeManager methods remain same as previous version
    # (including _init_sqlite, _ensure_persistence, _seed_initial_data, etc.)

class SearchManager:
    # Existing SearchManager implementation

class AIService:
    # Existing AIService implementation

class BotState(TypedDict):
    # Existing BotState definition

class WorkflowManager:
    # Existing WorkflowManager implementation

def initialize_session_state():
    required_keys = {
        'groq_key': None,
        'workflow_manager': None,
        'user_id': str(uuid.uuid4()),
        'messages': [],
        'custom_input': "",
        'astra_db_id': "",
        'astra_db_region': "",
        'astra_db_app_token': "",
        'astra_db_namespace': "default"
    }
    
    for key, default in required_keys.items():
        if key not in st.session_state:
            st.session_state[key] = default

def main():
    initialize_session_state()

    with st.sidebar:
        st.header("🔐 Configuration")
        
        # Groq API Key
        groq_key = st.text_input("Enter Groq API Key:", 
                                type="password", 
                                key="groq_key_input",
                                help="Get your API key from https://console.groq.com/keys")
        
        # Astra DB Configuration
        with st.expander("🚀 Astra DB Configuration (Optional)"):
            st.session_state.astra_db_id = st.text_input("Astra DB ID:")
            st.session_state.astra_db_region = st.text_input("Astra DB Region:")
            st.session_state.astra_db_app_token = st.text_input("Astra DB App Token:", type="password")
            st.session_state.astra_db_namespace = st.text_input("Namespace:", value="default")
        
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

    try:
        if not st.session_state.workflow_manager:
            st.session_state.workflow_manager = WorkflowManager()
    except Exception as e:
        st.error(f"Application initialization failed: {str(e)}")
        st.stop()

    # Rest of the Streamlit UI remains same

if __name__ == "__main__":
    main()
