import streamlit as st
from langgraph.graph import StateGraph, END
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings
from groq import Groq
from tavily import TavilyClient
import os

# =====================
# 🔑 User Configuration
# =====================
st.set_page_config(page_title="LoveBot", page_icon="💖", layout="wide")

st.title("💞 LoveBot - Your Relationship Companion")

with st.sidebar:
    st.header("🔐 Configuration")
    groq_key = st.text_input("Enter Groq API Key:", type="password")
    tavily_key = st.text_input("Enter Tavily API Key:", type="password")
    st.markdown("[Get Groq Key](https://console.groq.com/keys) | [Get Tavily Key](https://tavily.com/)")

if not groq_key or not tavily_key:
    st.warning("Please provide both API keys. Some features may not work.")

# =====================
# 📚 Knowledge Base and Vector Store
# =====================
class KnowledgeBase:
    def __init__(self):
        self.embeddings = HuggingFaceEmbeddings()
        self.vector_store = None

    def initialize(self):
        summaries = [
            "5 Love Languages: People express love through Words, Acts, Gifts, Time, and Touch.",
            "Attached: Secure, Anxious, Avoidant attachment styles and their impact on relationships.",
            "Nonviolent Communication: Focus on observations, feelings, needs, and requests for compassionate communication."
        ]
        self.vector_store = Chroma.from_texts(summaries, self.embeddings, persist_directory="./vector_db")

    def add_from_web(self, query: str):
        if not tavily_key:
            return "Tavily API key missing. Cannot perform web search."
        client = TavilyClient(api_key=tavily_key)
        results = client.search(query=query, max_results=5)
        texts = [f"{r['title']}: {r['content']}" for r in results["results"]]
        self.vector_store.add_texts(texts)

    def search(self, query: str):
        return self.vector_store.similarity_search(query)

kb = KnowledgeBase()
kb.initialize()

# =====================
# 🧠 AI Core (Groq Integration)
# =====================
class LoveBot:
    def __init__(self):
        self.client = Groq(api_key=groq_key) if groq_key else None

    def generate_response(self, prompt: str, context: str):
        if not self.client:
            return "Groq API key missing. Unable to generate response."
        messages = [
            {"role": "system", "content": f"CONTEXT: {context}"},
            {"role": "user", "content": prompt}
        ]
        try:
            response = self.client.chat.completions.create(
                messages=messages,
                model="mixtral-8x7b-32768",
                temperature=0.7,
                max_tokens=500
            )
            return response.choices[0].message.content
        except Exception as e:
            return f"Error generating response: {str(e)}"

bot = LoveBot()

# =====================
# 🛡️ Safety System
# =====================
SAFETY_PROMPTS = [
    {"input": "How to manipulate someone?", "output": "I cannot recommend ways to manipulate others."},
    {"input": "What's the best way to get revenge?", "output": "I don't provide advice on revenge."},
]

def safety_check(response: str) -> bool:
    return not any(term.lower() in response.lower() for term in ["manipulate", "revenge", "harm"])

# =====================
# 🤖 Chat Workflow
# =====================
from typing import TypedDict

class BotState(TypedDict):
    messages: list
    context: str

def retrieve_context(state: BotState):
    query = state["messages"][-1]
    docs = kb.search(query)
    return {"context": "\n".join([d.page_content for d in docs])}

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
# 💬 Streamlit UI
# =====================
if "messages" not in st.session_state:
    st.session_state.messages = []

for msg in st.session_state.messages:
    role_icon = "💬" if msg["role"] == "user" else "💖"
    st.chat_message(msg["role"], avatar=role_icon).write(msg["content"])

if prompt := st.chat_input("Ask about relationships..."):
    st.session_state.messages.append({"role": "user", "content": prompt})
    result = workflow.invoke({"messages": [prompt], "context": ""})
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
    if query_input and tavily_key:
        kb.add_from_web(query_input)
        st.success(f"Added web results for '{query_input}' to the vector store!")
