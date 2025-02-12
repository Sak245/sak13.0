import streamlit as st
from langgraph.graph import StateGraph, END
from qdrant_client import QdrantClient
from qdrant_client.http.models import PointStruct, VectorParams, Distance
from langchain_community.embeddings.huggingface import HuggingFaceEmbeddings
from groq import Groq
import uuid
from typing import TypedDict
from duckduckgo_search import DDGS
from transformers import pipeline
import re

# =====================
# 🔑 User Configuration
# =====================
st.set_page_config(page_title="LoveBot", page_icon="💖", layout="wide")

with st.sidebar:
    st.header("🔐 Configuration")
    groq_key = st.text_input("Enter Groq API Key:", type="password")
    st.markdown("[Get Groq Key](https://console.groq.com/keys)")

if not groq_key:
    st.error("Please provide the Groq API key in the sidebar to proceed.")
    st.stop()

# =====================
# 📚 Enhanced Knowledge Base (Qdrant)
# =====================
class KnowledgeBase:
    def __init__(self):
        self.client = QdrantClient(":memory:")
        self.embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
        self.collection_name = "lovebot_knowledge"
        
        # Initialize or reset collection
        if self.client.collection_exists(self.collection_name):
            self.client.delete_collection(self.collection_name)
            
        self.client.create_collection(
            collection_name=self.collection_name,
            vectors_config=VectorParams(size=384, distance=Distance.COSINE),
        )
        
        # Initialize with book summaries
        self.initialize_with_books()

    def initialize_with_books(self):
        """Initialize with relationship book summaries"""
        books = [
            ("Nonviolent Communication", "Marshall B. Rosenberg"),
            ("The Seven Principles for Making Marriage Work", "John Gottman"),
            ("Attached", "Amir Levine")
        ]
        
        for title, author in books:
            summary = self.get_book_summary(title, author)
            if summary:
                self._add_to_collection(summary, "book_summary")

    def get_book_summary(self, title: str, author: str) -> str:
        """Fetch book summary from web"""
        try:
            with DDGS() as ddgs:
                results = ddgs.text(f"{title} by {author} summary", max_results=1)
                if results:
                    return f"{title} by {author}: {results[0]['body']}"
        except Exception as e:
            st.error(f"Error fetching {title} summary: {str(e)}")
        return ""

    def _add_to_collection(self, text: str, source_type: str):
        """Helper to add documents to collection"""
        embedding = self.embeddings.embed_query(text)
        point = PointStruct(
            id=str(uuid.uuid4()),
            vector=embedding,
            payload={"text": text, "type": source_type}
        )
        self.client.upsert(collection_name=self.collection_name, points=[point])

    def add_from_web(self, query: str):
        """Add web results to knowledge base"""
        try:
            with DDGS() as ddgs:
                results = ddgs.text(query, max_results=3)
            for result in results:
                text = f"{result['title']}: {result['body']}"
                self._add_to_collection(text, "web_result")
        except Exception as e:
            st.error(f"Web search error: {str(e)}")

    def search(self, query: str, limit: int = 3):
        """Search knowledge base with safety checks"""
        try:
            embedding = self.embeddings.embed_query(query)
            results = self.client.search(
                collection_name=self.collection_name,
                query_vector=embedding,
                limit=limit,
                with_payload=True,
            )
            return [r.payload["text"] for r in results if "text" in r.payload]
        except Exception as e:
            st.error(f"Search error: {str(e)}")
            return []

kb = KnowledgeBase()

# =====================
# 🧠 Enhanced AI Core
# =====================
class LoveBot:
    def __init__(self):
        self.client = Groq(api_key=groq_key)
        self.safety_checker = pipeline(
            "text-classification", 
            model="Hate-speech-CNERG/dehatebert-mono-english"
        )

    def generate_response(self, prompt: str, context: str):
        """Generate safe response with context"""
        messages = [
            {"role": "system", "content": f"Context:\n{context}\n\nRespond helpfully as a relationship expert."},
            {"role": "user", "content": prompt}
        ]
        
        try:
            response = self.client.chat.completions.create(
                model="mixtral-8x7b-32768",
                messages=messages,
                temperature=0.7,
                max_tokens=500
            )
            output = response.choices[0].message.content
            
            if not self._is_safe(output):
                return "I cannot provide advice on that topic."
                
            return output
            
        except Exception as e:
            st.error(f"Generation error: {str(e)}")
            return "I'm having trouble responding right now."

    def _is_safe(self, text: str) -> bool:
        """Safety check using DeHateBERT"""
        try:
            result = self.safety_checker(text[:512])[0]
            return result["label"] != "LABEL_1" or result["score"] < 0.85
        except Exception as e:
            st.error(f"Safety check error: {str(e)}")
            return "harm" not in text.lower()

bot = LoveBot()

# =====================
# 🤖 Enhanced Workflow
# =====================
class BotState(TypedDict):
    messages: list[str]
    knowledge_context: str
    web_context: str

def retrieve_knowledge(state: BotState):
    """Retrieve from knowledge base"""
    try:
        query = state["messages"][-1]
        return {"knowledge_context": "\n".join(kb.search(query))}
    except Exception as e:
        st.error(f"Knowledge retrieval error: {str(e)}")
        return {"knowledge_context": ""}

def retrieve_web(state: BotState):
    """Retrieve fresh web context"""
    try:
        with DDGS() as ddgs:
            results = ddgs.text(state["messages"][-1], max_results=2)
        return {"web_context": "\n".join(f"• {r['body']}" for r in results)}
    except Exception as e:
        st.error(f"Web retrieval error: {str(e)}")
        return {"web_context": ""}

def generate_response(state: BotState):
    """Generate final response"""
    full_context = f"""
    KNOWLEDGE BASE CONTEXT:
    {state['knowledge_context']}
    
    WEB CONTEXT:
    {state['web_context']}
    """
    return {"response": bot.generate_response(state["messages"][-1], full_context)}

workflow = StateGraph(BotState)
workflow.add_node("retrieve_knowledge", retrieve_knowledge)
workflow.add_node("retrieve_web", retrieve_web)
workflow.add_node("generate", generate_response)

workflow.set_entry_point("retrieve_knowledge")
workflow.add_edge("retrieve_knowledge", "retrieve_web")
workflow.add_edge("retrieve_web", "generate")
workflow.add_edge("generate", END)

app = workflow.compile()

# =====================
# 💖 Streamlit UI
# =====================
st.title("💞 LoveBot - Relationship Expert")

# Chat Interface
if "messages" not in st.session_state:
    st.session_state.messages = []

for msg in st.session_state.messages:
    role_icon = "💬" if msg["role"] == "user" else "💖"
    st.chat_message(msg["role"], avatar=role_icon).write(msg["content"])

if prompt := st.chat_input("Ask about relationships..."):
    st.session_state.messages.append({"role": "user", "content": prompt})
    
    result = app.invoke({
        "messages": [prompt],
        "knowledge_context": "",
        "web_context": ""
    })
    
    if response := result.get("response"):
        st.session_state.messages.append({"role": "assistant", "content": response})
    st.rerun()

# Additional Features
with st.expander("📖 Story Completion"):
    story_input = st.text_area("Start a relationship story:")
    if st.button("Continue Story"):
        st.write(bot.generate_response(f"Continue this story positively: {story_input}", ""))

with st.expander("🔍 Research Assistant"):
    research_query = st.text_input("Research topic:")
    if st.button("Learn & Store"):
        kb.add_from_web(research_query)
        st.success(f"Learned about {research_query}!")

# Requirements
st.sidebar.markdown("""
**Required Packages:**)
