# =====================
# 🛠️ Initial Configuration
# =====================
import sys
sys.modules['torch.classes'] = None
import re
import uuid
import time
import logging
import traceback
import torch
import fitz
import numpy as np
from typing import TypedDict, List
from cachetools import TTLCache
import transformers
from astrapy import DataAPIClient
from duckduckgo_search import DDGS
from transformers import pipeline as transformers_pipeline
from langgraph.graph import StateGraph, END
from langchain_huggingface import HuggingFaceEmbeddings
from groq import Groq
import streamlit as st

# =====================
# 🧠 Core AI Components
# =====================
class Config:
    def __init__(self):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.embedding_model = "sentence-transformers/all-MiniLM-L6-v2"
        self.safety_model = "Hate-speech-CNERG/dehatebert-mono-english"
        self.rate_limit = 100
        self.max_text_length = 1500
        self.pdf_chunk_size = 750
        self.search_depth = 5

config = Config()

class NeuroState(TypedDict):
    dialog: List[dict]
    memories: List[str]
    web_context: str
    user_id: str

class QuantumKnowledgeManager:
    def __init__(self, token: str, db_id: str, region: str):
        self.endpoint = f"https://{db_id}-{region}.apps.astra.datastax.com"
        try:
            self.client = DataAPIClient(token)
            self.db = self.client.get_database_by_api_endpoint(self.endpoint)
            
            # SAK Collection Configuration
            if "sak" not in self.db.list_collection_names():
                try:
                    self.collection = self.db.create_collection(
                        "sak",
                        options={"vector": {
                            "dimension": 384,
                            "metric": "cosine"
                        }}
                    )
                    logging.info("Created new SAK collection")
                except Exception as create_error:
                    raise RuntimeError(f"""🚨 Collection Creation Failed
                    Ensure your token has 'Database Administrator' privileges
                    Raw error: {str(create_error)}""")
            else:
                self.collection = self.db.get_collection("sak")
                
            self.embeddings = HuggingFaceEmbeddings(
                model_name=config.embedding_model,
                encode_kwargs={'normalize_embeddings': True},
                model_kwargs={'device': config.device}
            )
            
        except Exception as e:
            raise RuntimeError(f"""🔌 Quantum Connection Failure
            Endpoint: {self.endpoint}
            Token: {token[:25]}... (Full: {token[:15]}...{token[-10:]})
            Diagnostic Steps:
            1. IP Whitelisting: https://dtsx.io/ip-whitelist-guide
            2. Token Verification: https://astra.datastax.com/org/tokens
            3. Connection Test:
               ```python
               from astrapy import DataAPIClient
               try:
                   client = DataAPIClient("{token}")
                   db = client.get_database_by_api_endpoint("{self.endpoint}")
                   print("Collections:", db.list_collection_names())
               except Exception as e:
                   print("Connection failed:", str(e))
               ```""")

    def _process_content(self, content: str) -> List[str]:
        return [content[i:i+config.pdf_chunk_size] 
                for i in range(0, len(content), config.pdf_chunk_size]

    def add_knowledge(self, content: str, source: str):
        try:
            chunks = self._process_content(content)
            for chunk in chunks:
                embedding = self.embeddings.embed_query(chunk)
                self.collection.insert_one({
                    "text": chunk,
                    "embedding": embedding.tolist(),
                    "source": source,
                    "timestamp": time.time()
                })
        except Exception as e:
            logging.error(f"Knowledge injection error: {str(e)}")

    def retrieve_memory(self, query: str, limit=5) -> List[str]:
        try:
            query_embed = self.embeddings.embed_query(query).tolist()
            results = self.collection.aggregate([
                {"$vectorSearch": {
                    "queryVector": query_embed,
                    "path": "embedding",
                    "numCandidates": 150,
                    "limit": limit,
                    "index": "vector_index"
                }},
                {"$project": {"text": 1, "_id": 0}}
            ])
            return [doc["text"] for doc in results]
        except Exception as e:
            logging.error(f"Memory retrieval error: {str(e)}")
            return []

class NeuroLoveAI:
    def __init__(self, api_key: str):
        self.groq = Groq(api_key=api_key)
        self.safety = transformers_pipeline(
            "text-classification", 
            model=config.safety_model,
            device=0 if torch.cuda.is_available() else -1
        )
        self.rate_limits = TTLCache(maxsize=10000, ttl=3600)

    def generate_empathy(self, prompt: str, context: str, user_id: str) -> str:
        if self.rate_limits.get(user_id, 0) >= config.rate_limit:
            return "💔 Let's take a breath and continue later..."
            
        self.rate_limits[user_id] = self.rate_limits.get(user_id, 0) + 1
        
        try:
            response = self.groq.chat.completions.create(
                model="mixtral-8x7b-32768",
                messages=[{
                    "role": "system",
                    "content": f"""As an empathy AI, integrate this context:
                    {context}
                    Respond with compassion and understanding."""
                }, {
                    "role": "user",
                    "content": prompt
                }],
                temperature=0.7,
                max_tokens=500
            )
            output = response.choices[0].message.content
            return output if self._safety_check(output) else "🚫 Response filtered"
        except Exception as e:
            logging.error(f"Generation error: {str(e)}")
            return "🌈 Every challenge is growth. Could you share more?"

    def _safety_check(self, text: str) -> bool:
        try:
            result = self.safety(text[:512])
            return result[0]['label'] == 'SAFE'
        except Exception as e:
            logging.error(f"Safety check error: {str(e)}")
            return False

class LoveFlow2025:
    def __init__(self, db_creds: dict, api_key: str):
        self.knowledge = QuantumKnowledgeManager(**db_creds)
        self.ai = NeuroLoveAI(api_key)
        self.flow = self._build_neural_graph()

    def _build_neural_graph(self):
        workflow = StateGraph(NeuroState)
        workflow.add_node("retrieve_memories", self._remember)
        workflow.add_node("search_web", self._search)
        workflow.add_node("synthesize", self._synthesize)
        workflow.set_entry_point("retrieve_memories")
        workflow.add_edge("retrieve_memories", "search_web")
        workflow.add_edge("search_web", "synthesize")
        workflow.add_edge("synthesize", END)
        return workflow.compile()

    def _remember(self, state: NeuroState) -> dict:
        return {"memories": self.knowledge.retrieve_memory(state["dialog"][-1]["content"])}

    def _search(self, state: NeuroState) -> dict:
        with DDGS() as ddgs:
            results = ddgs.text(state["dialog"][-1]["content"], max_results=config.search_depth)
            return {"web_context": "\n".join(f"🌐 {r['title']}: {r['body'][:200]}" for r in results)}

    def _synthesize(self, state: NeuroState) -> dict:
        context = f"KNOWLEDGE:\n{state['memories']}\nWEB:\n{state['web_context']}"
        return {"response": self.ai.generate_empathy(
            state["dialog"][-1]["content"],
            context,
            state["user_id"]
        )}

# =====================
# 💻 Streamlit Interface
# =====================
def validate_credentials(db_id: str, region: str) -> bool:
    uuid_pattern = re.compile(r"^[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}$", re.IGNORECASE)
    region_pattern = re.compile(r"^[a-z]{2}-[a-z]+-\d+$")
    
    if not uuid_pattern.match(db_id):
        st.error("❌ Invalid DB Cluster ID! Must be UUID format")
        return False
        
    if not region_pattern.match(region):
        st.error("❌ Region must be format 'us-east-1'")
        return False
        
    return True

st.set_page_config(page_title="LoveBot 2025", page_icon="💞", layout="wide")
st.write("""
<style>
    .stAlert {padding: 20px!important; border-radius: 10px!important;}
    .stTextInput input {border: 1px solid #ff69b4!important;}
</style>
""", unsafe_allow_html=True)

# Security credentials
ASTRA_TOKEN = "AstraCS:YzTkrbyuwALNUbsxXYYEOlHi:e7d4e9f2ad71f198b962813cd535f3a2a6f6bf9ea88420d286aca5803e280a89"
DB_ID = "40e5db47-786f-4907-acf1-17e1628e48ac"
REGION = "us-east-1"
GROQ_KEY = "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"

with st.sidebar:
    st.header("🔐 2025 Security Configuration")
    
    with st.expander("Astra DB Quantum Security", expanded=True):
        astra_db_token = st.text_input("Quantum Token", value=ASTRA_TOKEN, type="password")
        astra_db_id = st.text_input("DB Cluster ID", value=DB_ID)
        astra_db_region = st.text_input("Neural Region", value=REGION)

    with st.expander("Groq API v3"):
        groq_key = st.text_input("NeuroKey", value=GROQ_KEY, type="password")

    if st.button("🚀 Initialize Quantum Connection", type="primary"):
        if validate_credentials(astra_db_id, astra_db_region):
            try:
                st.session_state.neuro_flow = LoveFlow2025(
                    db_creds={
                        "token": astra_db_token,
                        "db_id": astra_db_id,
                        "region": astra_db_region
                    },
                    api_key=groq_key
                )
                st.success("✅ Quantum Neural Link Established!")
            except Exception as e:
                endpoint = f"https://{astra_db_id}-{astra_db_region}.apps.astra.datastax.com"
                st.error(f"""
                ❗ Critical Connection Failure
                Endpoint: {endpoint}
                Token: {astra_db_token[:25]}... (Full: {astra_db_token[:15]}...{astra_db_token[-10:]})
                
                🔥 REQUIRED ACTIONS:
                1. Verify IP Whitelisting: https://dtsx.io/ip-whitelist-guide
                2. Check Token Permissions: https://astra.datastax.com/org/tokens
                3. Test Connection Manually:
                ```python
                from astrapy import DataAPIClient
                try:
                    client = DataAPIClient("{astra_db_token}")
                    db = client.get_database_by_api_endpoint("{endpoint}")
                    print("Collections:", db.list_collection_names())
                except Exception as e:
                    print("ERROR:", str(e))
                ```""")
        else:
            st.error("Fix validation errors first")

    st.header("📊 System Health")
    if 'neuro_flow' in st.session_state:
        st.success("🟢 Neural Network Operational")
        st.metric("Processing Power", f"{config.device.upper()} ・ Torch {torch.__version__}")
    else:
        st.warning("🔴 System Offline")

# =====================
# 💞 Main Interface
# =====================
if "dialog" not in st.session_state:
    st.session_state.dialog = []
if "user_id" not in st.session_state:
    st.session_state.user_id = str(uuid.uuid4())

st.title("💞 LoveBot 2025 - Compassionate AI")

# Knowledge Upload
with st.expander("🧠 Upload Relationship Knowledge"):
    uploaded_files = st.file_uploader("Add PDF/text files", 
                                    type=["pdf", "txt", "md"],
                                    accept_multiple_files=True)
    if uploaded_files and 'neuro_flow' in st.session_state:
        for file in uploaded_files:
            try:
                if file.type == "application/pdf":
                    doc = fitz.open(stream=file.read(), filetype="pdf")
                    content = "".join(page.get_text() for page in doc)
                else:
                    content = file.read().decode()
                st.session_state.neuro_flow.knowledge.add_knowledge(content, file.name)
                st.toast(f"📥 Learned from {file.name}")
            except Exception as e:
                st.error(f"Upload failed: {str(e)}")

# Chat Interface
for msg in st.session_state.dialog:
    avatar = "💬" if msg["role"] == "user" else "💞"
    with st.chat_message(msg["role"], avatar=avatar):
        st.write(msg["content"])

if prompt := st.chat_input("Share your relationship thoughts..."):
    st.session_state.dialog.append({"role": "user", "content": prompt})
    
    if 'neuro_flow' not in st.session_state:
        st.error("Initialize quantum connection first!")
    else:
        try:
            with st.status("💭 Processing with Compassion...", expanded=True):
                result = st.session_state.neuro_flow.flow.invoke({
                    "dialog": st.session_state.dialog,
                    "memories": [],
                    "web_context": "",
                    "user_id": st.session_state.user_id
                })
                response = result.get("response", "Let's explore this together...")
                st.session_state.dialog.append({"role": "assistant", "content": response})
        except Exception as e:
            st.error("Quantum connection unstable - try again")
            logging.error(f"Main flow error: {traceback.format_exc()}")
    
    st.rerun()
