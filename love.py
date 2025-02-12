import streamlit as st
from langgraph.graph import StateGraph, END
from qdrant_client import QdrantClient
from qdrant_client.http.models import PointStruct, VectorParams, Distance
from langchain_community.embeddings.huggingface import HuggingFaceEmbeddings
from groq import Groq
import uuid
from typing import TypedDict
from duckduckgo_search import DDGS  # ✅ Correct import

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

    def initialize(self):
        """Initialize vector store with predefined knowledge."""
        summaries = [
            "5 Love Languages: Words, Acts, Gifts, Time, Touch.",
            "Attachment Styles: Secure, Anxious, Avoidant.",
            "Nonviolent Communication: Observations, Feelings, Needs."
        ]
        for summary in summaries:
            embedding = self.embeddings.embed_query(summary)
            point = PointStruct(
                id=str(uuid.uuid4()), vector=embedding, payload={"text": summary}
            )
            self.client.upsert(collection_name=self.collection_name, points=[point])

    def add_from_web(self, query: str):
        """Fetch search results using DuckDuckGo and store in vector DB."""
        with DDGS() as ddgs:
            results = ddgs.text(query, max_results=3)  # ✅ Using duckduckgo-search API
        
        if not results:
            st.error("No search results found.")
            return
        
        for result in results:
            text = result["title"] + " - " + result["body"]
            embedding = self.embeddings.embed_query(text)
            point = PointStruct(
                id=str(uuid.uuid4()), vector=embedding, payload={"text": text, "url": result["href"]}
            )
            self.client.upsert(collection_name=self.collection_name, points=[point])

    def search(self, query: str):
        """Search the knowledge base for relevant context."""
        embedding = self.embeddings.embed_query(query)
        results = self.client.query_points(
            collection_name=self.collection_name,
            query_vectors=[embedding],
            limit=3,
            with_payload=True,
        )
        
        return [r.payload["text"] for r in results if "text" in r.payload]

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
        
        try:
            response = self.client.chat.completions.create(  # ✅ Fixed API call
                model="mixtral-8x7b-32768",
                messages=messages,
                temperature=0.7,
                max_tokens=500
            )
            
            if response.choices:
                return response.choices[0].message.content
            
        except Exception as e:
            st.error(f"Error generating response: {e}")
        
        return "[No response generated]"

bot = LoveBot()

# =====================
# 🛡️ Safety System
# =====================
def safety_check(response: str) -> bool:
    """Check if response violates safety rules."""
    return not any(term.lower() in response.lower() for term in ["manipulate", "revenge", "harm"])

# =====================
# 💬 Chat Workflow
# =====================
class BotState(TypedDict):
    messages: list[str]
    context: str

def retrieve_context(state: BotState):
    query = state["messages"][-1]
    docs = kb.search(query)
    
    if not docs:
        return {"context": "[No relevant context found]"}

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
    
    result = app.invoke({
        "messages": [prompt],
        "context": ""
    })
    
    response = result.get("response", "[No response generated]")
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
    
    if query_input and st.button("Search & Store"):
        kb.add_from_web(query_input)
        st.success("Knowledge stored successfully! 🔍")
