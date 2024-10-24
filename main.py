import streamlit as st
from components.chat_interface import chat_interface
from utils.model_handlers import get_model_response, fetch_retrieved_context
from components.score_display import display_scores
from langchain_pinecone import PineconeVectorStore

from vector_store.vectorstore import PineconeManager
pinecone_api_key = "108137e8-b872-4ff4-a279-c61e8a7ec4ca"
groq_api_key ="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"
gemini_api_key ="AIzaSyCo5xy1plbKOps2Gq9K64dvm01njXWiYtY"
from langchain_community.embeddings import HuggingFaceEmbeddings
import google.generativeai as genai
from groq import Groq

# App configuration
st.set_page_config(page_title="KerasInsight -> Multi-Model Chat", layout="wide")

# Sidebar for model selection
st.sidebar.title("Select Model")
selected_model = st.sidebar.radio(
    "Choose a model:",
    options=["Mixtral"]
)

# Main chat interface
st.title("GSOC")
INDEX_NAME = 'dlprojectchecknomic'

@st.cache_resource
def initialize_vector_store():
    pinecone_manager = PineconeManager(pinecone_api_key, INDEX_NAME)
    pinecone_manager.initialize_index()
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    vector_store = PineconeVectorStore(index=pinecone_manager.index, embedding=embeddings)
    retriever = vector_store.as_retriever(search_kwargs={"k": 3})
    genai.configure(api_key=gemini_api_key)
    tuned_model = genai.get_tuned_model('tunedModels/finetuninggemmafordl1-xxcubsl6ftaf')
    model = genai.GenerativeModel(model_name=tuned_model.name)
    client = Groq(
        api_key=groq_api_key
    )
    return retriever, model, client

retriever, model, client = initialize_vector_store()


# Chat functionality
user_query = st.text_input("Ask a question:")
if user_query:
    with st.spinner("Fetching response..."):
        # Fetch response from the selected model
        response, scores, context = get_model_response(selected_model, user_query,retriever,model,client)
    
    # Display the chat interface
    chat_interface(user_query, response)

    # Display the context and scores on the right
    st.write("---")
    st.header("Context and Scores")
    display_scores(context, scores)