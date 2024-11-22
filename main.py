import streamlit as st
from langchain_community.utilities import ArxivAPIWrapper, WikipediaAPIWrapper
from langchain_community.tools import ArxivQueryRun, WikipediaQueryRun
from langchain.schema import Document
from typing import List
from langchain.vectorstores.cassandra import Cassandra
from langchain.indexes.vectorstore import VectorStoreIndexWrapper
from langgraph.graph import END, StateGraph, START
from typing_extensions import TypedDict

# Define GraphState
class GraphState(TypedDict):
    question: str
    generation: str
    documents: List[str]

# Placeholder for retriever
def dummy_retrieve(question):
    return f"Retrieved results for question: {question}"

# Placeholder for Wikipedia search
def dummy_wiki_search(question):
    return f"Wiki search results for question: {question}"

# Graph functions
def retrieve(state):
    question = state["question"]
    documents = dummy_retrieve(question)
    return {"documents": [documents], "question": question}

def wiki_search(state):
    question = state["question"]
    documents = dummy_wiki_search(question)
    return {"documents": [documents], "question": question}

def route_question(state):
    question = state["question"]
    if "agent" in question.lower():
        return "vectorstore"
    return "wiki_search"

# Build the workflow graph
workflow = StateGraph(GraphState)
workflow.add_node("wiki_search", wiki_search)
workflow.add_node("retrieve", retrieve)
workflow.add_conditional_edges(
    START,
    route_question,
    {
        "wiki_search": "wiki_search",
        "vectorstore": "retrieve",
    },
)
workflow.add_edge("retrieve", END)
workflow.add_edge("wiki_search", END)
app = workflow.compile()

# Streamlit UI
st.title("LangGraph with Streamlit")
st.write("Ask your question, and we'll route it to the appropriate source.")

# Input from user
question = st.text_input("Enter your question:")

if question:
    st.write("Processing your query...")
    inputs = {"question": question}
    for output in app.stream(inputs):
        for key, value in output.items():
            st.write(f"Node: `{key}`")
            st.write("Response:")
            for doc in value.get("documents", []):
                st.write(f"- {doc}")
