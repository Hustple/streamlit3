rag_prompt = """You are an intelligent assistant designed to provide accurate and relevant information from GSOC organization data.

        Here is the retrieved context, which may contain both explanatory text and meaningful code snippets:

        {context}

        Carefully analyze the above context, considering both the text and any provided code for clarity.

        Now, review the user's query:

        {question}

        Generate a detailed response that accurately addresses the query using the provided context. If the context includes relevant code, incorporate it into your response. Ensure that your answer is both clear and grounded in the provided content.

        Response:
        """

groundedness_prompt = """You are an AI grader evaluating the groundedness of an answer based on given facts. 

FACTS: {documents}

ANSWER TO EVALUATE: {generation}

Grade the answer based on these criteria:
1. The answer is grounded in the provided facts.
2. The answer does not contain "hallucinated" information outside the scope of the facts.

Explain your reasoning step-by-step. Then, provide a final score between 0 and 1, where:
0 = The answer completely fails to meet the criteria
1 = The answer fully meets all criteria

Your response should be in this format:
Reasoning: [Your step-by-step explanation]
Score: [Your score between 0 and 1]
Explanation: [A brief summary of why you gave this score]
"""

# Answer relevance prompt
answer_relevance_prompt = """You are an AI grader evaluating the relevance of an answer to a given question.

QUESTION: {question}

ANSWER TO EVALUATE: {generation}

Grade the answer based on this criterion:
1. The answer helps to answer the question (it can contain extra relevant information)

Explain your reasoning step-by-step. Then, provide a final score between 0 and 1, where:
0 = The answer is completely irrelevant to the question
1 = The answer is highly relevant and fully addresses the question

Your response should be in this format:
Reasoning: [Your step-by-step explanation]
Score: [Your score between 0 and 1]
Explanation: [A brief summary of why you gave this score]
"""

# Context relevance prompt
context_relevance_prompt = """You are an AI grader evaluating whether a given context contains enough information to answer a question.

QUESTION: {question}

CONTEXT TO EVALUATE: {context}

Grade the context based on this criterion:
1. The context contains enough information to answer the question (it can contain extra relevant information)

Explain your reasoning step-by-step. Then, provide a final score between 0 and 1, where:
0 = The context contains no relevant information to answer the question
1 = The context contains all necessary information to fully answer the question

Your response should be in this format:
Reasoning: [Your step-by-step explanation]
Score: [Your score between 0 and 1]
Explanation: [A brief summary of why you gave this score]
"""