"""
config.py

Centralized configuration utilities for the generative AI project.
"""

import os
import logging
import streamlit as st
# import faiss
# import chromadb
# import json
# from langchain_openai import OpenAIEmbeddings
from langchain_core.messages import AnyMessage, SystemMessage, HumanMessage,ToolMessage, AIMessage

from dotenv import load_dotenv,find_dotenv
_ = load_dotenv(find_dotenv())

from langchain_openai import ChatOpenAI

# -----------------------------------------------------------------------------
# 1. ENVIRONMENT VARIABLES
# -----------------------------------------------------------------------------
# Load API key from environment variable
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")

# -----------------------------------------------------------------------------
# 2. LOGGING CONFIGURATION
# -----------------------------------------------------------------------------
def setup_logging():
    """
    Sets up logging configuration with file name, function name, and line number.
    Logs messages to the console at INFO level.
    """
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(filename)s:%(funcName)s:%(lineno)d - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S"
    )
    logging.info("Logging initialized.")


def display_saved_plot(plot_path: str,):

    """
    Loads and displays a saved plot from the given path in a Streamlit app with a highlighted background.

    Args:
        plot_path (str): Path to the saved plot image.
        bg_color (str): Background color for the image container.
        padding (str): Padding inside the image container.
        border_radius (str): Border radius for rounded corners.
    """

    bg_color: str = "#f0f2f6"
    padding: str = "5px"
    border_radius: str = "10px"
    if os.path.exists(plot_path):
        # Apply styling using markdown with HTML and CSS
        st.markdown(
            f"""
            <style>
                .image-container {{
                    background-color: {bg_color};
                    padding: {padding};
                    border-radius: {border_radius};
                    display: flex;
                    justify-content: center;
                }}
            </style>
            <div class="image-container">
                <img src="data:image/png;base64,{get_base64_image(plot_path)}" style="max-width:100%; height:auto;">
            </div>
            """,
            unsafe_allow_html=True
        )
    else:
        st.error(f"Plot not found at {plot_path}")

def get_base64_image(image_path: str) -> str:
    """
    Converts an image to a base64 string.

    Args:
        image_path (str): Path to the image.

    Returns:
        str: Base64-encoded image.
    """
    import base64
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode("utf-8")

#
# class SemanticCache:
#     def __init__(self, model_name="text-embedding-ada-002", chroma_db_path="chroma_db_data"):
#         """
#         Initializes the semantic cache with FAISS and ChromaDB.
#
#         :param model_name: Name of the sentence transformer model to use.
#         :param chroma_db_path: Path for storing ChromaDB persistent data.
#         """
#
#         self.model = OpenAIEmbeddings(model=model_name)
#         dimension = self.model.embed_query("Hello World !")
#
#         # Initialize FAISS index for embedding storage
#         self.faiss_index = faiss.IndexFlatL2(len(dimension))
#
#         # Initialize ChromaDB client
#         self.chroma_client = chromadb.PersistentClient(path=chroma_db_path)
#         # List all collections
#         # self.collections = self.chroma_client.list_collections()
#         # # Delete all collections
#         # for collection in self.collections:
#         #     self.chroma_client.delete_collection(collection.name)
#
#         # print("All collections deleted..")
#         self.collection = self.chroma_client.get_or_create_collection(name="semantic_cache10")
#
#     def generate_embedding(self, text):
#         """
#         Generates an embedding for a given text using the sentence transformer model.
#
#         :param text: The input text.
#         :return: The generated embedding as a NumPy array.
#         """
#         return self.model.embed_query(text)
#
#     def retrieve_from_cache(self, query, threshold=0.1):
#         """
#         Retrieves a cached response for a given query if a similar one exists.
#
#         :param query: The input query text.
#         :param threshold: The similarity threshold for retrieval.
#         :return: Cached response if found, otherwise None.
#         """
#         if self.faiss_index.ntotal == 0:
#             return None  # Cache is empty
#
#         query_embedding = self.generate_embedding(query)
#         query_vector = np.array([query_embedding], dtype=np.float32)
#
#         # Search for similar embeddings in FAISS index
#         distances, indices = self.faiss_index.search(query_vector, k=1)
#
#         # Check if the closest match is within the similarity threshold
#         if distances[0][0] < threshold:
#             doc_id = str(indices[0][0])
#             result = self.collection.get(ids=[doc_id])
#             if result and result['documents']:
#                 return result
#
#         return None
#
#     def update_cache(self, question, response):
#         """
#         Updates the cache by storing the new question-response pair.
#
#         :param question: The input question.
#         :param response: The generated response.
#         """
#         embedding = self.generate_embedding(question)
#         embedding = np.array([embedding], dtype=np.float32)
#
#         # Add embedding to FAISS index
#         self.faiss_index.add(embedding)
#
#         # Store question and response in ChromaDB
#         doc_id = str(self.faiss_index.ntotal - 1)  # ID is the index of the embedding
#         self.collection.add(
#             ids=[doc_id],
#             embeddings=embedding.tolist(),
#             documents=[question],
#             metadatas=[response]
#         )
#
#     def input_parser(self, data):
#         """
#         Converts unsupported data types to ChromaDB-compatible formats.
#
#             - None -> "None" (string)
#             - DataFrame -> JSON string
#             - list/tuple -> JSON string
#             - dict -> JSON string
#             - AIMessage -> Extracted text
#         """
#         cleaned_data = {}
#
#         for key, value in data.items():
#             if value is None:
#                 cleaned_data[key] = "None"  # Store None as a string
#             elif isinstance(value, pd.DataFrame):
#                 cleaned_data[key] = value.to_json()  # Store DataFrame as JSON
#             elif isinstance(value, (list, tuple, dict)):
#                 cleaned_data[key] = json.dumps(value)  # Convert lists, tuples, and dicts to JSON string
#             elif isinstance(value, AIMessage):
#                 cleaned_data[key] = value.content  # Extract text from AIMessage
#             else:
#                 cleaned_data[key] = value  # Keep valid types unchanged
#
#         return cleaned_data
#
#     def output_parser(self, data):
#         """
#         Converts stored ChromaDB-compatible formats back to original data types.
#
#             - "None" (string) -> None
#             - JSON string (from DataFrame) -> DataFrame (with correct dtypes)
#             - JSON string (from list/tuple) -> list/tuple
#             - Dict (stored as JSON) -> dict
#         """
#         parsed_data = {}
#
#         for key, value in data.items():
#             if value == "None":
#                 parsed_data[key] = None  # Convert "None" string back to None
#
#             elif isinstance(value, str):
#                 try:
#                     # Try parsing JSON (for DataFrames, lists, tuples, or dicts)
#                     parsed = json.loads(value)
#
#                     # If it's a dictionary with all string keys, check for DataFrame
#                     if isinstance(parsed, dict):
#                         try:
#                             df = pd.DataFrame(parsed)
#
#                             # Convert datetime-like columns back
#                             for col in df.columns:
#                                 if df[col].dtype == "object":  # Possible stringified datetime
#                                     try:
#                                         df[col] = pd.to_datetime(df[col], errors='ignore')
#                                     except Exception:
#                                         pass  # If conversion fails, keep original
#
#                             parsed_data[key] = df  # Convert back to DataFrame
#                         except Exception:
#                             parsed_data[key] = parsed  # Keep it as dict
#                     else:
#                         parsed_data[key] = parsed  # Convert back to list/tuple
#
#                 except (json.JSONDecodeError, TypeError):
#                     parsed_data[key] = value  # Keep it as a string if not JSON
#
#             else:
#                 parsed_data[key] = value  # Keep valid types unchanged
#
#         return parsed_data
