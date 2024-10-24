import streamlit as st

def chat_interface(user_query, response):
    st.subheader("Chat Interface")
    st.write(f"**User:** {user_query}")
    st.write(f"**AI Response:** {response}")