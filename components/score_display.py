import streamlit as st

def display_scores(context, scores):
    st.subheader("Retrieved Context")
    st.write(context)

    st.subheader("Evaluation Scores")
    st.write(f"Faithfulness Score: {scores['faithfulness']}")
    st.write(f"Answer Relevance Score: {scores['answer_relevance']}")
    st.write(f"Context Relevance Score: {scores['context_relevance']}")