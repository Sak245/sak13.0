import streamlit as st
from langgraph.graph import StateGraph, END
from qdrant_client import QdrantClient
from qdrant_client.http.models import PointStruct, VectorParams, Distance
from langchain_community.embeddings.huggingface import HuggingFaceEmbeddings
from groq import Groq
from serpapi import GoogleSearch
import uuid
from typing import TypedDict

# =====================
# 🔑 User Configuration
# =====================
st.set_page_config(page_title="LoveBot", page_icon="💖", layout="wide")

with st.sidebar:
    st.header("🔐 Configuration")
    groq_key = st.text_input("Enter Groq API Key:", type="password")
    serpapi_key = st.text_input("Enter SerpAPI Key:", type="password")
    st.markdown("[Get SerpAPI Key](https://serpapi.com/)")
    st.markdown("[Get Groq Key](https://console.groq.com/keys)")

if not groq_key or not serpapi_key:
    st.error("Please provide API keys in the sidebar to proceed.")
    st.stop()

# =====================
# 📚 Knowledge Base (Qdrant)
# =====================
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

    def search(self, query: str):
        embedding = self.embeddings.embed_query(query)
        results = self.client.search(
            collection_name=self.collection_name,
            query_vector=embedding,
            limit=3,
            with_payload=True,
        )
        return [r.payload.get("text", "") for r in results if "text" in r.payload]

    def add_book_summary(self, title: str, author: str):
        query = f"{title} by {author} book summary"
        params = {
            "engine": "google",
            "q": query,
            "api_key": serpapi_key
        }
        search = GoogleSearch(params)
        results = search.get_dict()
        snippets = [r["snippet"] for r in results.get("organic_results", []) if "snippet" in r]
        summary = " ".join(snippets)[:1000] if snippets else "Summary not found."
        embedding = self.embeddings.embed_query(summary)
        point = PointStruct(id=str(uuid.uuid4()), vector=embedding, payload={"text": summary})
        self.client.upsert(collection_name=self.collection_name, points=[point])
        return summary

kb = KnowledgeBase()

# =====================
# 🧠 AI Core (Groq Integration)
# =====================
class LoveBot:
    def __init__(self):
        self.client = Groq(api_key=groq_key)

    def generate_response(self, prompt: str, context: str):
        messages = [
            {"role": "system", "content": f"CONTEXT: {context}"},
            {"role": "user", "content": prompt}
        ]
        try:
            response = self.client.chat.completions.create(
                model="mixtral-8x7b-32768",
                messages=messages,
                temperature=0.7,
                max_tokens=500
            )
            return response.choices[0].message.content if response.choices else "[No response generated]"
        except Exception as e:
            return f"Error generating response: {e}"

bot = LoveBot()

# =====================
# 💬 Chat Workflow
# =====================
class BotState(TypedDict):
    messages: list[str]
    context: str

def retrieve_context(state: BotState):
    query = state["messages"][-1] if state["messages"] else ""
    docs = kb.search(query)
    return {"context": "\n".join(docs) if docs else "[No relevant context found]"}

def generate_response(state: BotState):
    response = bot.generate_response(state["messages"][-1], state["context"])
    return {"response": response}

workflow = StateGraph(BotState)
workflow.add_node("retrieve_context", retrieve_context)
workflow.add_node("generate_response", generate_response)
workflow.set_entry_point("retrieve_context")
workflow.add_edge("retrieve_context", "generate_response")
workflow.add_edge("generate_response", END)
app = workflow.compile()

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
    st.session_state.messages.append({"role": "user", "content": prompt})
    result = app.invoke({"messages": [prompt], "context": ""})
    response = result.get("response", "[No response generated]")
    st.session_state.messages.append({"role": "assistant", "content": response})

st.divider()

with st.expander("📖 Book Summaries"):
    title = st.text_input("Book Title:")
    author = st.text_input("Author:")
    if st.button("Fetch Summary"):
        summary = kb.add_book_summary(title, author)
        st.success(summary)

st.divider()

with st.expander("🔍 Web Search & Vector Store"):
    query_input = st.text_input("Search for knowledge:")
    if query_input and st.button("Search & Store"):
        kb.add_book_summary(query_input, "")
        st.success("Knowledge stored successfully! 🔍")
