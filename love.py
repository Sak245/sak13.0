import streamlit as st
from langgraph.graph import StateGraph, END
from qdrant_client import QdrantClient
from qdrant_client.http.models import PointStruct, VectorParams, Distance
from langchain_community.embeddings.huggingface import HuggingFaceEmbeddings
from groq import Groq
import requests
import uuid
from typing import TypedDict

# =====================
# 🔑 User Configuration
# =====================
st.set_page_config(page_title="LoveBot", page_icon="💖", layout="wide")

with st.sidebar:
    st.header("🔐 Configuration")
    groq_key = st.text_input("Enter Groq API Key:", type="password")
    duckduckgo_key = st.text_input("Enter DuckDuckGo API Key (optional):", type="password")
    st.markdown("[Get Groq Key](https://console.groq.com/keys) | [Learn about DuckDuckGo API](https://duckduckgo.com/)")

if not groq_key:
    st.error("Please provide the Groq API key in the sidebar to proceed.")
    st.stop()

# =====================
# 📚 Knowledge Base and Vector Store (Qdrant)
# =====================
class KnowledgeBase:
    def __init__(self):
        self.client = QdrantClient(":memory:")  # In-memory for simplicity
        self.embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

        # Initialize collection for storing vectors
        self.collection_name = "lovebot_knowledge"
        self.client.recreate_collection(
            collection_name=self.collection_name,
            vectors_config=VectorParams(size=384, distance=Distance.COSINE),
        )

    def initialize(self):
        """Initialize vector store with default summaries."""
        summaries = [
            "5 Love Languages: Words, Acts, Gifts, Time, Touch.",
            "Attached: Secure, Anxious, Avoidant attachment styles.",
            "Nonviolent Communication: Observations, feelings, needs."
        ]
        for summary in summaries:
            embedding = self.embeddings.embed_query(summary)
            point = PointStruct(
                id=str(uuid.uuid4()),  # Use a unique string ID
                vector=embedding,
                payload={"text": summary}
            )
            self.client.upsert(collection_name=self.collection_name, points=[point])

    def add_from_web(self, query: str):
        """Scrape web content using DuckDuckGo and add to vector store."""
        response = requests.get(f"https://api.duckduckgo.com/?q={query}&format=json")
        
        if response.status_code == 200:
            results = response.json().get("RelatedTopics", [])
            for result in results:
                if "Text" in result:
                    embedding = self.embeddings.embed_query(result["Text"])
                    point = PointStruct(
                        id=str(uuid.uuid4()),  # Use a unique string ID
                        vector=embedding,
                        payload={"text": result["Text"], "url": result.get("FirstURL")}
                    )
                    self.client.upsert(collection_name=self.collection_name, points=[point])
        else:
            st.error("Failed to fetch web results. Please check your DuckDuckGo API key.")

    def search(self, query: str):
        """Search vector store for relevant context."""
        embedding = self.embeddings.embed_query(query)
        results = self.client.search(
            collection_name=self.collection_name,
            query_vector=embedding,
            limit=3,
        )
        return [result.payload["text"] for result in results]

kb = KnowledgeBase()
kb.initialize()

# =====================
# 🧠 AI Core (Groq Integration)
# =====================
class LoveBot:
    def __init__(self):
        self.client = Groq(api_key=groq_key)

    def generate_response(self, prompt: str, context: str):
        """Generate response using Groq."""
        messages = [
            {"role": "system", "content": f"CONTEXT: {context}"},
            {"role": "user", "content": prompt}
        ]
        
        response = self.client.chat.completions.create(
            messages=messages,
            model="mixtral-8x7b-32768",
            temperature=0.7,
            max_tokens=500
        )
        return response.choices[0].message.content

bot = LoveBot()

# =====================
# 🛡️ Safety System
# =====================
SAFETY_PROMPTS = [
    {"input": "How to manipulate someone?", "output": "I cannot recommend ways to manipulate others. Healthy relationships are built on mutual respect and open communication."},
    {"input": "What's the best way to get revenge?", "output": "I don't provide advice on revenge or harmful actions. It's important to process your emotions in a healthy way."},
]

def safety_check(response: str) -> bool:
    """Check if response violates safety rules."""
    return not any(term.lower() in response.lower() for term in ["manipulate", "revenge", "harm"])

# =====================
# 💬 Chat Workflow
# =====================
class BotState(TypedDict):
    messages: list
    context: str

def retrieve_context(state: BotState):
    query = state["messages"][-1]
    docs = kb.search(query)
    return {"context": "\n".join(docs)}

def generate_response(state: BotState):
    prompt = state["messages"][-1]
    context = state["context"]
    
    response = bot.generate_response(prompt, context)
    
    if not safety_check(response):
        return {"response": "I cannot provide advice on that topic."}
    
    return {"response": response}

workflow = StateGraph(BotState)
workflow.add_node("retrieve_context", retrieve_context)
workflow.add_node("generate_response", generate_response)

workflow.set_entry_point("retrieve_context")
workflow.add_edge("retrieve_context", "generate_response")
workflow.add_edge("generate_response", END)

# =====================
# 💖 Streamlit UI
# =====================
st.title("💞 LoveBot - Your Relationship Companion")

if "messages" not in st.session_state:
    st.session_state.messages = []

for msg in st.session_state.messages:
    role_icon = "💬" if msg["role"] == "user" else "💖"
    st.chat_message(msg["role"], avatar=role_icon).write(msg["content"])

if prompt := st.chat_input("Ask about relationships..."):
    # Add user message to history
    st.session_state.messages.append({"role": "user", "content": prompt})
    
    result = workflow.invoke({
        "messages": [prompt],
        "context": ""
    })
    
    response = result["response"]
    st.session_state.messages.append({"role": "assistant", "content": response})
    
st.divider()

with st.expander("📖 Story Completion"):
    story_input = st.text_area("Start your story:")
    
    if st.button("Complete Story"):
        story_completion = bot.generate_response(f"Continue this story positively:\n{story_input}", "")
        st.success(story_completion)

st.divider()

with st.expander("🔍 Web Search & Vector Store"):
    query_input = st.text_input("Search the web for knowledge:")
    
    if query_input:
        kb.add_from_web(query_input)
        st.success(f"Added web results for '{query_input}' to the vector store!")
