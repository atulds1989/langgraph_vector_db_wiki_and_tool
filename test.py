import os
from dotenv import load_dotenv
import streamlit as st
from utils import *
from models import *
from langchain_groq import ChatGroq

# Load environment variables
load_dotenv()

# Constants
ASTRA_DB_APPLICATION_TOKEN = os.getenv("ASTRA_DB_APPLICATION_TOKEN")
ASTRA_DB_ID = os.getenv("ASTRA_DB_ID")
GROQ_API_KEY = os.getenv("GROQ_API_KEY")

import os
os.environ["USER_AGENT"] = "my-app/1.0"


urls = [
    "https://lilianweng.github.io/posts/2023-06-23-agent/",
    "https://lilianweng.github.io/posts/2023-03-15-prompt-engineering/",
    "https://lilianweng.github.io/posts/2023-10-25-adv-attack-llm/",
]

obj = Astra_db_init()


doc_splits = get_docs(urls=urls)
astra_vector_store, astra_vector_index, retriever = get_vector_db(doc_splits=doc_splits)

# print(retriever.invoke("What is agent",ConsistencyLevel="LOCAL_ONE"))


# Load environment variables from the .env file
load_dotenv()

# Set up API keys from environment variables
groq_api_key = os.getenv("GROQ_API_KEY")
langsmith_api_key = os.getenv("LANGCHAIN_API_KEY")


llm=ChatGroq(groq_api_key=groq_api_key,model_name="Gemma2-9b-It")
structured_llm_router = llm.with_structured_output(RouteQuery)


# Prompt
system = """You are an expert at routing a user question to a vectorstore or wikipedia.
The vectorstore contains documents related to agents, prompt engineering, and adversarial attacks.
Use the vectorstore for questions on these topics. Otherwise, use wiki-search."""
route_prompt = ChatPromptTemplate.from_messages(
    [
        ("system", system),
        ("human", "{question}"),
    ]
)


question_router = route_prompt | structured_llm_router

print(
    question_router.invoke(
        {"question": "who is Sharukh Khan?"}
    )
)

print(question_router.invoke(
        {"question": "What are the types of agent memory?"}
    )
)



### Working With Tools
from langchain_community.utilities import ArxivAPIWrapper,WikipediaAPIWrapper
from langchain_community.tools import ArxivQueryRun,WikipediaQueryRun

## Arxiv and wikipedia Tools
arxiv_wrapper=ArxivAPIWrapper(top_k_results=1, doc_content_chars_max=200)
arxiv=ArxivQueryRun(api_wrapper=arxiv_wrapper)

api_wrapper=WikipediaAPIWrapper(top_k_results=1,doc_content_chars_max=200)
wiki=WikipediaQueryRun(api_wrapper=api_wrapper)


from langchain.schema import Document
# def retrieve(state):
#     """
#     Retrieve documents

#     Args:
#         state (dict): The current graph state

#     Returns:
#         state (dict): New key added to state, documents, that contains retrieved documents
#     """
#     print("---RETRIEVE---")
#     question = state["question"]

#     # Retrieval
#     documents = retriever.invoke(question)
#     return {"documents": documents, "question": question}


# def wiki_search(state):
#     """
#     wiki search based on the re-phrased question.

#     Args:
#         state (dict): The current graph state

#     Returns:
#         state (dict): Updates documents key with appended web results
#     """

#     print("---wikipedia---")
#     print("---HELLO--")
#     question = state["question"]
#     # print(question)

#     # Wiki search
#     docs = wiki.invoke({"query": question})
#     #print(docs["summary"])
#     # wiki_results = docs
#     wiki_results = Document(page_content=docs)
#     return {"documents": wiki_results, "question": question}


def retrieve(state):
    print("---RETRIEVE---")
    question = state["question"]

    # Retrieval process
    documents = retriever.invoke(question)
    return {"documents": documents, "question": question}

def wiki_search(state):
    print("---WIKIPEDIA---")
    question = state["question"]

    # Wiki search process
    docs = wiki.invoke({"query": question})
    wiki_results = Document(page_content=docs)
    return {"documents": [wiki_results], "question": question}


### Edges ###
def route_question(state):
    """
    Route question to wiki search or RAG.

    Args:
        state (dict): The current graph state

    Returns:
        str: Next node to call
    """

    print("---ROUTE QUESTION---")
    question = state["question"]
    source = question_router.invoke({"question": question})
    if source.datasource == "wiki_search":
        print("---ROUTE QUESTION TO Wiki SEARCH---")
        return "wiki_search"
    elif source.datasource == "vectorstore":
        print("---ROUTE QUESTION TO RAG---")
        return "vectorstore"


from langgraph.graph import END, StateGraph, START

workflow = StateGraph(State)
# Define the nodes
workflow.add_node("wiki_search", wiki_search)  # web search
workflow.add_node("retrieve", retrieve)  # retrieve

# Build graph
workflow.add_conditional_edges(
    START,
    route_question,
    {
        "wiki_search": "wiki_search",
        "vectorstore": "retrieve",
    },
)
workflow.add_edge( "retrieve", END)
workflow.add_edge( "wiki_search", END)
# Compile
graph = workflow.compile()

# inputs = {"question": "What is agent?"}
# for output in graph.stream(inputs):
#     print(output)


# Initialize Streamlit app with a title and introduction
st.set_page_config(page_title="LangGraph Chatbot", page_icon="💬", layout="centered")
# st.title("💬 LangGraph Tool-Based Chatbot")

# Streamlit interface
st.markdown(
    "<h3 style='text-align: center; color: #333;'>💬 LangGraph Tool-Based Chatbot</h3>",
    unsafe_allow_html=True,
)


st.write("Ask a question below, and get responses from either an AI assistant or specific research tools.")

# Initialize session state for message tracking
if "messages" not in st.session_state:
    st.session_state["messages"] = []

# Sidebar information
with st.sidebar:
    st.subheader("🤖 Chatbot Information")
    st.write("This chatbot provides intelligent responses by either using an AI assistant or fetching data from research tools such as Arxiv and Wikipedia.")
    st.write("Sources used will be shown next to responses, enabling a clear understanding of where each answer originates from.")


# # Display chat history with bubble-style UI
# def display_chat():
#     # Display conversation history in a more interactive chat format
#     st.markdown("###### Chat History")
#     for message in st.session_state["messages"]:
#         role = message["type"]
#         content = message["content"]

#         if role == "user":
#             # Display user message in a speech bubble style
#             st.markdown(
#                 f'<div style="text-align:right;"><span style="display:inline-block; background-color:#DCF8C6; padding:10px; border-radius:10px;">'
#                 f'<strong>You:</strong> {content}</span></div>',
#                 unsafe_allow_html=True,
#             )
#         else:
#             # Display assistant or tool response with source labeling
#             role_label = "Assistant" if role == "assistant" else role.capitalize()
#             st.markdown(
#                 f'<div style="text-align:left;"><span style="display:inline-block; background-color:#E8E8E8; padding:10px; border-radius:10px;">'
#                 f'<strong>{role_label}:</strong> {content}</span></div>',
#                 unsafe_allow_html=True,
#             )

def display_chat():
    # Display conversation history in a more interactive chat format
    st.markdown("###### Chat History")
    for message in st.session_state["messages"]:
        role = message["type"]
        content = message["content"]

        if role == "user":
            # Display user message in a speech bubble style
            st.markdown(
                f"""
                <div style="text-align:right; margin-bottom:10px;">
                    <span style="
                        display:inline-block; 
                        background-color:#DCF8C6; 
                        padding:10px; 
                        border-radius:10px; 
                        max-width:70%; 
                        word-wrap:break-word;
                        box-shadow: 2px 2px 5px rgba(0,0,0,0.1);">
                        <strong>You:</strong> {content}
                    </span>
                </div>
                """,
                unsafe_allow_html=True,
            )
        else:
            # Display assistant or tool response with source labeling
            role_label = "Assistant" if role == "assistant" else role.capitalize()
            st.markdown(
                f"""
                <div style="text-align:left; margin-bottom:10px;">
                    <span style="
                        display:inline-block; 
                        background-color:#E8E8E8; 
                        padding:10px; 
                        border-radius:10px; 
                        max-width:70%; 
                        word-wrap:break-word;
                        box-shadow: 2px 2px 5px rgba(0,0,0,0.1);">
                        <strong>{role_label}:</strong> {content}
                    </span>
                </div>
                """,
                unsafe_allow_html=True,
            )




# Input section with an input field and an arrow button in the same row
with st.form(key="input_form", clear_on_submit=True):
    st.markdown("#### Type your query:")
    cols = st.columns([10, 1])  # Two columns for input and send button
    user_input = cols[0].text_input("Your Message", label_visibility="collapsed", placeholder="Ask me anything...")
    user_input_dict = {'question' : user_input}
    submit_button = cols[1].form_submit_button("➤")  # Arrow button


if submit_button and user_input:
    # Add user input to the conversation
    st.session_state['messages'].append({"type": "user", "content": user_input})

    # Run the graph and process the response
    for event in graph.stream({"question": user_input}):
        for value in event.values():
            # Handle the output
            if "documents" in value:
                response = value["documents"][0].page_content
            elif "generation" in value:
                response = value["generation"]
            else:
                response = "I'm not sure how to answer that."

            # Add assistant response to the conversation
            st.session_state['messages'].append({"type": "assistant", "content": response})

#     # Display updated chat history
#     display_chat()

# if submit_button and user_input:
#     # Add user input to the conversation
    # st.session_state['messages'].append({"type": "user", "content": user_input})

    # # Run the graph and process the response
    # for event in graph.stream({"question": user_input}):
#         for value in event.values():
#             response = value.get("documents", [{"page_content": "Sorry, I couldn't find an answer."}])[0].page_content
#             # Add assistant response to the conversation
#             st.session_state['messages'].append({"type": "assistant", "content": response})

    # Display the updated chat
    display_chat()







# # Process input and update chat
# if submit_button and user_input:
#     # Add user input to the conversation
#     st.session_state['messages'].append(("user", user_input_dict))
#     # st.session_state["messages"].append({"type": "user", "content": user_input})
#     # events = graph.stream({"messages": st.session_state["messages"]}, stream_mode="values")

#     # Run the chatbot state machine and get the response immediately
#     response = None
#     for event in graph.stream({'messages': st.session_state['messages']}):
#         for value in event.values():
#             response = value['messages'].content  # Store the response message
#             # response = value['documents'][0].dict()['metadata']['description']
#             st.session_state['messages'].append(("assistant", response))  # Add response to session state

#     # Display the updated chat history with new messages
#     display_chat()

# # Clear chat button below input section for better accessibility
# if st.button("Clear Chat", help="Clear the entire chat history"):
#     st.session_state['messages'] = []
#     st.experimental_rerun()  # Reload to refresh chat display


# Display footer
footer()

