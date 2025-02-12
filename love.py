import streamlit as st
from langgraph.graph import StateGraph, END
from qdrant_client import QdrantClient
from qdrant_client.http.models import PointStruct, VectorParams, Distance
from langchain_community.embeddings.huggingface import HuggingFaceEmbeddings
from groq import Groq
import requests
import uuid
from typing import TypedDict

st.set_page_config(page_title="LoveBot", page_icon="💖", layout="wide")

with st.sidebar:
    st.header("🔐 Configuration")
    groq_key = st.text_input("Enter Groq API Key:", type="password")
    duckduckgo_key = st.text_input("Enter DuckDuckGo API Key (optional):", type="password")

if not groq_key:
    st.error("Please provide the Groq API key.")
    st.stop()

# ========== 📚 Knowledge Base Setup ==========
class KnowledgeBase:
    def __init__(self):
        self.client = QdrantClient(":memory:")
        self.embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
        self.collection_name = "lovebot_knowledge"
        if self.client.collection_exists(self.collection_name):
            self.client.delete_collection(self.collection_name)
        self.client.create_collection(
            collection_name=self.collection_name,
            vectors_config=VectorParams(size=384, distance=Distance.COSINE),
        )

    def initialize(self):
        summaries = [
            "5 Love Languages: Words, Acts, Gifts, Time, Touch.",
            "Attached: Secure, Anxious, Avoidant attachment styles.",
            "Nonviolent Communication: Observations, feelings, needs."
        ]
        for summary in summaries:
            embedding = self.embeddings.embed_query(summary)
            point = PointStruct(id=str(uuid.uuid4()), vector=embedding, payload={"text": summary})
            self.client.upsert(collection_name=self.collection_name, points=[point])

    def search(self, query: str):
        embedding = self.embeddings.embed_query(query)
        results = self.client.query_points(
            collection_name=self.collection_name,
            query_vector=embedding,
            limit=3,
        )
        return [result.payload["text"] for result in results if "text" in result.payload]

kb = KnowledgeBase()
kb.initialize()

# ========== 🧠 AI Core ==========
class LoveBot:
    def __init__(self):
        self.client = Groq(api_key=groq_key)

    def generate_response(self, prompt: str, context: str):
        messages = [{"role": "system", "content": f"CONTEXT: {context}"}, {"role": "user", "content": prompt}]
        
        try:
            response = self.client.chat.completions.create(
                messages=messages, model="mixtral-8x7b-32768", temperature=0.7, max_tokens=500
            )
            return response.choices[0].message.content
        except Exception as e:
            return f"Error generating response: {str(e)}"

bot = LoveBot()

# ========== 💬 Chat Workflow ==========
class BotState(TypedDict):
    messages: list[str]
    context: str

def retrieve_context(state: BotState):
    query = state["messages"][-1] if state["messages"] else ""
    docs = kb.search(query)
    return {"context": "\n".join(docs)}

def generate_response(state: BotState):
    prompt = state["messages"][-1] if state["messages"] else ""
    context = state["context"]
    response = bot.generate_response(prompt, context)
    return {"response": response if response else "[No response generated]"}

workflow = StateGraph(BotState)
workflow.add_node("retrieve_context", retrieve_context)
workflow.add_node("generate_response", generate_response)

workflow.set_entry_point("retrieve_context")
workflow.add_edge("retrieve_context", "generate_response")
workflow.add_edge("generate_response", END)

app = workflow.compile()

# ========== 💞 Streamlit UI ==========
st.title("💞 LoveBot - Your Relationship Companion")

if "messages" not in st.session_state:
    st.session_state.messages = []

for msg in st.session_state.messages:
    role_icon = "💬" if msg["role"] == "user" else "💖"
    st.chat_message(msg["role"], avatar=role_icon).write(msg["content"])

if prompt := st.chat_input("Ask about relationships..."):
    st.session_state.messages.append({"role": "user", "content": prompt})
    
    try:
        result = app.invoke({"messages": [prompt], "context": ""})
        response = result.get("response", "[No response generated]")  # Ensure key exists
    except Exception as e:
        response = f"Error processing request: {str(e)}"
    
    st.session_state.messages.append({"role": "assistant", "content": response})

st.divider()

with st.expander("📖 Story Completion"):
    story_input = st.text_area("Start your story:")
    if st.button("Complete Story"):
        story_completion = bot.generate_response(f"Continue this story positively:\n{story_input}", "")
        st.success(story_completion)
