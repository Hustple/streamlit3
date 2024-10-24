import random
from utils.evaluation import evaluate_response
from groq import Groq  # Assuming Groq API
from prompts.prompts import rag_prompt



def get_model_response(model_name, question,retriever,model,client):
    context = fetch_retrieved_context(question,retriever)
    
    if model_name == "Gemini":
        response = generate_gemini_response(question, context,model)
    elif model_name == "LLaMA":
        response = groq_llama_answer_generate(question,context,client)
    else:  # Mixtral
        response = groq_mixtral_answer_generate(question,context,client)

    scores = evaluate_response(response, context, question,model)
    return response, scores, context

def fetch_retrieved_context(question,retriever):
    context_from_pinecone = retriever.get_relevant_documents(question)
    context = "\n\n".join(doc.page_content for doc in context_from_pinecone)
    return context


def generate_gemini_response(question, context,model):
    rag_prompt_formatted = rag_prompt.format(context=context, question=question)
    rag_result = model.generate_content(rag_prompt_formatted)
    rag_response = rag_result.text
    return rag_response

def groq_llama_answer_generate(question,context,client):
    rag_prompt_formatted = rag_prompt.format(context=context, question=question)
    chat_completion = client.chat.completions.create(
        messages=[
            {
                "role": "user",
                "content": rag_prompt_formatted,
            }
        ],
        model="llama-3.2-1b-preview",
    )
    return chat_completion.choices[0].message.content

def groq_mixtral_answer_generate(question,context,client):
    rag_prompt_formatted = rag_prompt.format(context=context, question=question)
    chat_completion = client.chat.completions.create(
        messages=[
            {
                "role": "user",
                "content": rag_prompt_formatted,
            }
        ],
        model="mixtral-8x7b-32768",
    )
    return chat_completion.choices[0].message.content