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

# Page Configuration
st.set_page_config(page_title="GSOC AI Assistant", layout="wide", initial_sidebar_state="expanded")

# Custom CSS for enhanced design
st.markdown("""
    <style>
        body {
            background-color: #f5f5f5;
        }
        .main-title {
            font-family: 'Arial', sans-serif;
            color: #0047AB;
            font-weight: bold;
            font-size: 40px;
        }
        .chatbox {
            background-color: #ffffff;
            border-radius: 10px;
            padding: 20px;
            margin-bottom: 20px;
            box-shadow: 0px 0px 10px rgba(0, 0, 0, 0.1);
        }
        .stTextInput>div>input {
            border-radius: 10px;
            border: 2px solid #0047AB;
        }
        .context-scores {
            background-color: #E0F7FA;
            border-left: 5px solid #00ACC1;
            padding: 15px;
            margin-top: 20px;
            border-radius: 5px;
        }
        .sidebar .block-container {
            background-color: #333333;
        }
        .sidebar .sidebar-content h2 {
            color: #FFD700;
        }
    </style>
""", unsafe_allow_html=True)

# Sidebar for Model Selection
st.sidebar.title("Select Model")
selected_model = st.sidebar.radio(
    "Choose a model:",
    options=["LLaMA", "Mixtral"]
)

# Title & Header
st.markdown("<h1 class='main-title'>GSOC AI Assistant</h1>", unsafe_allow_html=True)
st.markdown("**Explore GSOC data with AI-powered insights**")

# Main chat interface
st.title("GSOC")
INDEX_NAME = 'rag-gsoc-data-dlproject'

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


st.write("## 🤖 **Ask a Question:**")

# Two-column layout for enhanced structure
col1, col2 = st.columns([2, 1])

# Left column (Chat Interface)
with col1:
    user_query = st.text_input("🔍 Type your question here", key="user_query")

    if user_query:
        with st.spinner("Generating response..."):
            try:
                response, scores, context = get_model_response(selected_model, user_query, retriever, model, client)

                # Display the response inside a chat box
                st.markdown("### 💬 **AI Response**")
                st.markdown(f"<div class='chatbox'>**User:** {user_query}<br>**AI:** {response}</div>", unsafe_allow_html=True)

            except Exception as e:
                st.error(f"Error occurred: {str(e)}")

# Right column (Context & Scores)
with col2:
    st.markdown("### 📊 **Context and Scores**")
    if user_query:
        st.markdown("<div class='context-scores'>", unsafe_allow_html=True)
        
        # Display the retrieved context
        st.subheader("📑 Context")
        st.write(context if context else "No context retrieved.")

        # Display the scores
        st.subheader("🔢 Scores")
        if scores:
            display_scores(context, scores)
        else:
            st.write("No scores available.")
        
        st.markdown("</div>", unsafe_allow_html=True)

# Footer and Branding
st.write("---")
st.markdown("""
    <div style='text-align: center;'>
        <small>GSOC Data Retrieval System | Powered by LangChain, Pinecone, HuggingFace, and Google Gemini</small>
    </div>
""", unsafe_allow_html=True)

