### Router
from typing import Literal
from langchain_core.prompts import ChatPromptTemplate
# from langchain_core.pydantic_v1 import BaseModel, Field
from typing import List
from typing_extensions import TypedDict
import os, cassandra, cassio
from dotenv import load_dotenv
from langchain.schema import Document
from pydantic import BaseModel, Field


load_dotenv()
# Set up API keys from environment variables
ASTRA_DB_APPLICATION_TOKEN = os.getenv("ASTRA_DB_APPLICATION_TOKEN")
ASTRA_DB_ID = os.getenv("ASTRA_DB_ID")



class Astra_db_init:
    def __init__(self) -> None:
      try:
        cassio.init(token=ASTRA_DB_APPLICATION_TOKEN, database_id=ASTRA_DB_ID)
        print("Connection initialized successfully.")
      except Exception as e:
        print(f"Failed to initialize connection: {e}")


# obj = Astra_db_init()


# Data model
class RouteQuery(BaseModel):
    """Route a user query to the most relevant datasource."""

    datasource: Literal["vectorstore", "wiki_search"] = Field(
        ...,
        description="Given a user question choose to route it to wikipedia or a vectorstore.",
    )



## Graph
from typing import List
from typing_extensions import TypedDict

class State(TypedDict):
    """
    Represents the state of our graph.

    Attributes:
        question: question
        generation: LLM generation
        documents: list of documents
    """

    question: str
    generation: str
    documents: List[str]



if __name__=="__main__":

    obj = Astra_db_init()
