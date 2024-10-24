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

# Set Streamlit page configuration and dark mode theme
st.set_page_config(page_title="GSOC-Data", layout="wide", initial_sidebar_state="collapsed")

# Inject custom CSS for dark mode and styling
st.markdown("""
    <style>
        body {
            background-color: #1E1E1E;
            color: white;
        }
        .stSidebar {
            background-color: #333333;
        }
        .css-1lcbmhc {
            background-color: #333333;
        }
        h1, h2, h3, h4 {
            color: #FFD700; /* Gold */
        }
        .stButton>button {
            background-color: #FFD700;
            color: black;
            font-weight: bold;
        }
    </style>
    """, unsafe_allow_html=True)

# Sidebar for model selection (still giving option but redesigned)
st.sidebar.title("🔍 Model Selection")
selected_model = st.sidebar.selectbox(
    "Choose a model:",
    options=["LLaMA", "Mixtral"]
)

# Title section with subtitle and instructions
st.title("GSOC AI Assistant 💡")
st.subheader("Retrieve contextual information from GSOC Data")

st.write("---")

# Introduce a two-column layout
col1, col2 = st.columns([3, 1])

# Main Column: Chat interface
with col1:
    st.header("💬 Chat with GSOC Data")

    # Input area for the user's query
    user_query = st.text_input("🔍 Ask a question:")

    if user_query:
        with st.spinner("Fetching response..."):
            # Fetch response from the selected model
            retriever, model, client = initialize_vector_store()
            response, scores, context = get_model_response(selected_model, user_query, retriever, model, client)
        
        # Chat interface display
        st.markdown("### 🤖 AI Response")
        chat_interface(user_query, response)

# Right Column: Context and Scores Display
with col2:
    st.header("📊 Context & Scores")
    
    # Dynamically show the response scores and context
    if user_query:
        display_scores(context, scores)
    else:
        st.info("Ask a question to view context and scores!")

# Footer area with a separator
st.write("---")
st.markdown("**GSOC Data Retrieval System | Powered by LangChain and Pinecone**")
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