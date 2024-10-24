from random import uniform
import time
from prompts.prompts import answer_relevance_prompt,context_relevance_prompt,groundedness_prompt

def grade_response(model,prompt_template, **kwargs):
    prompt = prompt_template.format(**kwargs)
    result = model.generate_content(prompt)
    return result.text

def extract_score(result_text):
    for line in result_text.split('\n'):
        if line.startswith('Score:'):
            return float(line.split(':')[1].strip())
    return None

def evaluate_response(response, context, question,model,delay_range = (1,3)):
    answer_relevancy_score = grade_response(
                model,
                answer_relevance_prompt,
                question=question,
                generation=response
            )
    time.sleep(uniform(delay_range[0], delay_range[1]))
    
    context_relevancy_score = grade_response(
                model,
                context_relevance_prompt,
                question=question,
                context=context
            )
            
    time.sleep(uniform(delay_range[0], delay_range[1]))
    
    faithfulness_score = grade_response(
                model,
                groundedness_prompt,
                documents=context,
                generation=response
            )
    # Dummy implementation of scores
    scores = {
        "faithfulness": extract_score(faithfulness_score),
        "answer_relevance": extract_score(answer_relevancy_score),
        "context_relevance": extract_score(context_relevancy_score)
    }
    return scores