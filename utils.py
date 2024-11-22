from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import WebBaseLoader
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain.indexes.vectorstore import VectorStoreIndexWrapper

def get_docs(urls):
    """
    Function to fetch and process documents from URLs.
    """
    # Docs to index
    # Load
    docs = [WebBaseLoader(url).load() for url in urls]
    docs_list = [item for sublist in docs for item in sublist]

    # Split
    text_splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(
        chunk_size=500, chunk_overlap=100
    )
    doc_splits = text_splitter.split_documents(docs_list)
    return doc_splits

def get_vector_db(doc_splits):
    """
    Function to create a FAISS-based vector store.
    """
    # Initialize embeddings
    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

    # Create FAISS vector store
    faiss_vector_store = FAISS.from_documents(doc_splits, embedding=embeddings)
    print(f"Inserted {len(doc_splits)} documents into FAISS.")

    # Create an index wrapper
    faiss_vector_index = VectorStoreIndexWrapper(vectorstore=faiss_vector_store)

    # Get retriever
    retriever = faiss_vector_store.as_retriever()
    return faiss_vector_store, faiss_vector_index, retriever


import streamlit as st
# Footer section
def footer():
    st.markdown("---")
    st.markdown("### About")
    st.markdown("This chatbot is built using Langsmith and LangGraph. Enjoy interactive conversations!")
    # st.markdown("Developed by **Atul Purohit**")























# ##########################################################
# # app.py 



# # Load environment variables
# load_dotenv()

# # Set up the Groq API key
# groq_api_key = os.getenv("GROQ_API_KEY")
# os.environ["GROQ_API_KEY"] = groq_api_key
# llm = ChatGroq(groq_api_key=groq_api_key, model_name="Gemma2-9b-It")

# # Define a route query model
# class RouteQuery(BaseModel):
#     """Route a user query to the most relevant datasource."""
#     datasource: str = Field(..., description="Route to vectorstore or wiki_search.")

# # Initialize the Astra DB
# init_astra_db()

# # URL List for indexing
# urls = [
#     "https://lilianweng.github.io/posts/2023-06-23-agent/",
#     "https://lilianweng.github.io/posts/2023-03-15-prompt-engineering/",
#     "https://lilianweng.github.io/posts/2023-10-25-adv-attack-llm/",
# ]

# # Build the index
# doc_splits = build_index(urls)
# embeddings = create_embeddings()
# astra_vector_store, astra_vector_index = create_vector_store(embeddings, doc_splits)

# # Define the prompt and router
# system = """You are an expert at routing a user question to a vectorstore or wikipedia."""
# route_prompt = ChatPromptTemplate.from_messages(
#     [
#         ("system", system),
#         ("human", "{question}"),
#     ]
# )

# # Define the Streamlit UI
# st.title("Query Routing System")

# # User input
# question = st.text_input("Enter your question:")

# if question:
#     # Define logic to route question to either Wikipedia or VectorStore
#     route_query = RouteQuery(datasource="vectorstore")  # Replace with actual logic
#     st.write(f"Routing question '{question}' to {route_query.datasource}")
    
#     # For now, display response from the vector store (stubbed out)
#     st.write("Fetching results from the Vector Store...")
    
#     # Display some dummy response
#     st.write("Result 1: Detailed info about agents.")
#     st.write("Result 2: Information on prompt engineering.")




# # utils.py 



# # Load environment variables from .env
# load_dotenv()

# ASTRA_DB_APPLICATION_TOKEN = os.getenv("ASTRA_DB_APPLICATION_TOKEN")
# ASTRA_DB_ID = os.getenv("ASTRA_DB_ID")
# GROQ_API_KEY = os.getenv("GROQ_API_KEY")

# def init_astra_db():
#     """Initialize the connection to Astra DB."""
#     cassio.init(token=ASTRA_DB_APPLICATION_TOKEN, database_id=ASTRA_DB_ID)

# def create_embeddings():
#     """Create embeddings using HuggingFace."""
#     return HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

# def build_index(urls):
#     """Build the index by loading and splitting documents."""
#     # Load documents
#     docs = [WebBaseLoader(url).load() for url in urls]
#     docs_list = [item for sublist in docs for item in sublist]
    
#     # Split documents
#     text_splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(
#         chunk_size=500, chunk_overlap=0
#     )
#     doc_splits = text_splitter.split_documents(docs_list)
    
#     return doc_splits

# def create_vector_store(embeddings, doc_splits):
#     """Create and populate a Cassandra vector store."""
#     astra_vector_store = Cassandra(
#         embedding=embeddings,
#         table_name="qa_mini_demo",
#         session=None,
#         keyspace=None
#     )
#     astra_vector_store.add_documents(doc_splits)
    
#     astra_vector_index = VectorStoreIndexWrapper(vectorstore=astra_vector_store)
#     return astra_vector_store, astra_vector_index

