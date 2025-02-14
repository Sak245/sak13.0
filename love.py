# Add at the very top before other imports
import sys
sys.modules['torch.classes'] = None
from functools import lru_cache

# =====================
# 🛠️ Initial Configuration
# =====================
import warnings
import os
import tempfile
import torch
import transformers
import fitz
from astrapy import DataAPIClient

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
from langchain_huggingface import HuggingFaceEmbeddings
from groq import Groq
import uuid
from typing import TypedDict, List
from duckduckgo_search import DDGS
from transformers import pipeline as transformers_pipeline
import time
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
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.embedding_model = "sentence-transformers/all-MiniLM-L6-v2"
        self.safety_model = "Hate-speech-CNERG/dehatebert-mono-english"
        self.rate_limit = 50
        self.max_text_length = 1000
        self.pdf_chunk_size = 500
        
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
# 🔑 Credential Handling
# =====================
with st.sidebar:
    st.header("🔐 Database & API Configuration")
    
    # Astra DB Credentials
    st.subheader("Astra DB Credentials")
    astra_db_token = st.text_input("Astra DB Token", type="password")
    astra_db_id = st.text_input("Astra DB ID", type="password")
    astra_db_region = st.text_input("Astra DB Region", type="password")
    
    # Groq API Key
    st.subheader("Groq API Configuration")
    groq_key = st.text_input("Groq API Key", type="password")
    st.markdown("[Get Groq Key](https://console.groq.com/keys)")
    
    st.header("📊 System Status")
    st.write(f"**Processing Device:** {config.device.upper()}")

# Validate credentials
if not all([astra_db_token, astra_db_id, astra_db_region]):
    st.error("Please provide all Astra DB credentials to proceed.")
    st.stop()

if not groq_key:
    st.error("Please provide the Groq API key to proceed.")
    st.stop()

# =====================
# 📚 Astra DB Knowledge Management
# =====================
class KnowledgeManager:
    def __init__(self, token: str, db_id: str, region: str):
        self.embeddings = HuggingFaceEmbeddings(
            model_name=config.embedding_model,
            encode_kwargs={'normalize_embeddings': True}
        )
        
        self.client = DataAPIClient(token)
        self.db = self.client.get_database_by_api_endpoint(
            f"https://{db_id}-{region}.apps.astra.datastax.com"
        )
        
        # Initialize collection
        if "lovebot_knowledge" not in self.db.list_collection_names():
            self.collection = self.db.create_collection("lovebot_knowledge")
        else:
            self.collection = self.db.get_collection("lovebot_knowledge")

    def _pdf_to_text(self, pdf_bytes: bytes) -> List[str]:
        try:
            doc = fitz.open(stream=pdf_bytes, filetype="pdf")
            text = ""
            for page in doc:
                text += page.get_text("text") + "\n"
            text = re.sub(r'\s+', ' ', text)
            text = re.sub(r'\n{3,}', '\n\n', text)
            return self._chunk_text(text)
        except Exception as e:
            logging.error(f"PDF error: {str(e)}")
            return []

    def _chunk_text(self, text: str) -> List[str]:
        paragraphs = re.split(r'\n\s*\n', text)
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
            logging.error(f"Add error: {str(e)}")

    def add_pdf_knowledge(self, pdf_bytes: bytes, source_name: str) -> int:
        try:
            chunks = self._pdf_to_text(pdf_bytes)
            for chunk in chunks:
                self._add_single_knowledge(chunk, f"pdf:{source_name}")
            return len(chunks)
        except Exception as e:
            logging.error(f"PDF add error: {str(e)}")
            return 0

    def _add_single_knowledge(self, text: str, source_type: str):
        embedding = self.embeddings.embed_query(text)
        
        self.collection.insert_one({
            "text": text,
            "embedding": embedding,
            "source_type": source_type
        })

    def search_knowledge(self, query: str, limit=3) -> List[str]:
        try:
            query_embedding = self.embeddings.embed_query(query)
            results = self.collection.find(
                {},
                vector=query_embedding,
                limit=limit
            )
            return [doc["text"] for doc in results["data"]["documents"]]
        except Exception as e:
            logging.error(f"Search error: {str(e)}")
            return []

# ... [Keep the rest of the code from previous implementation unchanged] ...

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
            logging.error(f"Search error: {str(e)}")
            return []

# =====================
# 🧠 AI Service
# =====================
class AIService:
    def __init__(self, api_key: str):
        self.groq_client = Groq(api_key=api_key)
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
                        "content": f"""You are a compassionate relationship expert. Context:
                        {context}
                        Provide thoughtful, empathetic advice."""
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
                    
                return output if self._is_safe(output) else "🚫 Response blocked"
            
            except Exception as e:
                logging.error(f"Attempt {attempt+1} failed: {str(e)}")
                time.sleep(self.retry_delay * (attempt + 1))
                
        return "Open communication and mutual understanding are key. Could you elaborate?"

# =====================
# 🤖 Workflow Management
# =====================
class BotState(TypedDict):
    messages: List[str]
    knowledge_context: str
    web_context: str
    user_id: str

class WorkflowManager:
    def __init__(self, db_creds: dict, api_key: str):
        self.knowledge = KnowledgeManager(db_creds)
        self.ai = AIService(api_key)
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
            return {"knowledge_context": "\n".join(context) if context else "No relevant knowledge found"}
        except Exception as e:
            return {"knowledge_context": "Knowledge base unavailable"}

    def retrieve_web(self, state: BotState) -> dict:
        try:
            results = self.ai.searcher.cached_search(state["messages"][-1])
            return {"web_context": "\n".join(
                f"• {r['title']}: {r['content'][:200]}" for r in results
            ) if results else "No web results found"}
        except Exception as e:
            return {"web_context": "Web search unavailable"}

    def generate(self, state: BotState) -> dict:
        full_context = f"""
        KNOWLEDGE BASE:\n{state['knowledge_context']}
        WEB RESULTS:\n{state['web_context']}
        """.strip()
            
        return {"response": self.ai.generate_response(
            prompt=state["messages"][-1],
            context=full_context,
            user_id=state["user_id"]
        )}

# =====================
# 💻 Streamlit Interface
# =====================
if "workflow_manager" not in st.session_state or st.session_state.groq_key != groq_key:
    st.session_state.groq_key = groq_key
    st.session_state.workflow_manager = WorkflowManager({
        "id": astra_db_id,
        "region": astra_db_region,
        "app_token": astra_db_app_token,
        "client_id": astra_db_client_id,
        "client_secret": astra_db_client_secret
    }, groq_key)

if "user_id" not in st.session_state:
    st.session_state.user_id = str(uuid.uuid4())
if "messages" not in st.session_state:
    st.session_state.messages = []

# PDF Upload
with st.sidebar:
    st.header("📥 Knowledge Upload")
    uploaded_file = st.file_uploader("Upload relationship PDF", type=["pdf"])
    
    if uploaded_file:
        with st.spinner("Processing PDF..."):
            try:
                num_chunks = st.session_state.workflow_manager.knowledge.add_pdf_knowledge(
                    uploaded_file.getvalue(),
                    uploaded_file.name
                )
                st.success(f"Added {num_chunks} knowledge chunks!")
            except Exception as e:
                st.error(f"PDF error: {str(e)}")

# Chat Interface
st.title("💖 LoveBot: AI Relationship Assistant")

for role, text in st.session_state.messages:
    avatar = "💬" if role == "user" else "💖"
    with st.chat_message(role, avatar=avatar):
        st.write(text)

if prompt := st.chat_input("Ask about relationships..."):
    st.session_state.messages.append(("user", prompt))
    
    try:
        with st.status("💞 Analyzing...", expanded=True) as status:
            try:
                result = st.session_state.workflow_manager.workflow.invoke({
                    "messages": [m[1] for m in st.session_state.messages],
                    "knowledge_context": "",
                    "web_context": "",
                    "user_id": st.session_state.user_id
                })
                
                response = result.get("response", "Could you clarify?")                
                st.session_state.messages.append(("assistant", response))
                status.update(label="✅ Done", state="complete")
                
            except Exception as e:
                st.session_state.messages.append(("assistant", "Let's focus on understanding each other. Could you rephrase?"))
                status.update(label="❌ Failed", state="error")
                logging.error(traceback.format_exc())
                
    except Exception as fatal_error:
        st.session_state.messages.append(("assistant", "Let's try again! Please share your thoughts."))
        logging.critical(traceback.format_exc())
    
    st.rerun()

# Additional Features
with st.expander("📖 Story Assistance"):
    story_prompt = st.text_area("Start your relationship story:")
    if st.button("Continue Story"):
        if story_prompt.strip():
            st.session_state.messages.append(("user", f"Continue: {story_prompt}"))
            st.rerun()

with st.expander("🔍 Research Assistant"):
    research_query = st.text_input("Research topic:")
    if st.button("Learn More"):
        if research_query.strip():
            st.session_state.messages.append(("user", f"Research: {research_query}"))
            st.rerun()
