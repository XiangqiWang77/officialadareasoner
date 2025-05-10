import json
import os
from dotenv import load_dotenv


load_dotenv()


api_key = os.getenv('API_KEY')
def load_questions(file_path):
    with open(file_path, 'r') as f:
        return json.load(f)



def append_result(result, file_path):
    try:
        with open(file_path, 'r') as f:
            results = json.load(f)
    except FileNotFoundError:
        results = []
    
    results.append(result)
    
    with open(file_path, 'w') as f:
        json.dump(results, f, indent=4)



def send_openai_prompt(prompt_text, model_name="gpt-4o", temperature=0.7, top_p=0.7, token_limit=1000):
    
    try:
        from openai import OpenAI
        client = OpenAI(api_key=api_key)
        chat_completion = client.chat.completions.create(
            messages=[
                {
                    "role": "user",
                    "content": prompt_text,
                }
            ],
            model=model_name,
            temperature=temperature,
            max_tokens=token_limit
        )
        print(chat_completion.choices[0].message.content)
        return chat_completion.choices[0].message.content 
    except Exception as e:
        return f"Request failed: {e}"



def robust_reward_v2(question, correct_answer, reasoning_process, options=None):
    prompt = f"""Assess with rigorous precision whether the provided reasoning process match the ground truth answer. 
    
For a given option and response, you need to match the content of option and response. You must not rely on the option index only, as in many cases, the index is actually messed.
    
Apply these criteria for judgement and carefully consider:

# Mandatory Evaluation Criteria
1. **Content Equivalence**: Accept only fully equivalent numerical representations (e.g., 0.5, 50%, 1/2) and variations in units or notation when they completely match the ground truth.
2. **Logical Inference**: Verify that at least one reasoning step directly and logically deduces the entire correct answer in a mathematically or logically sound manner.
3. **Substantive Matching**: For multiple-choice questions, assess the complete content of the answer (e.g., ensure "Option B" is fully equivalent to the correct answer, not just matching the label).
4. **Semantic and Methodological Equivalence**: Recognize alternative phrasing or solution methods only if a single step unambiguously converges on the complete correct answer.
5. **Scientific and Technical Rigor**: In technical contexts, differences in terminology, notation, or intermediate steps are acceptable only when they lead clearly and entirely to the correct conclusion.

Using the criteria outlined above, determine whether any single rule is met—if so, the response is considered a match.

# Question
{question}

# Ground Truth Answer
{correct_answer}

# Provided Reasoning
{reasoning_process}

Provide your final judgment as a JSON object with the following structure:
{{
  "judge_explanation": "<brief explanation>",
  "result": "<Yes or No>"
}}

Make sure you output json in plain text not code format.
"""


    response = send_openai_prompt(prompt).strip().lower()
    return response


if __name__ == "__main__":
    input_file = 'E:/testresult/gpt-4o.json'
    output_file = 'E:/testresultm/judge.json'
    data = load_questions(input_file)
    
    #data=data[]
    for item in data:
        if item["best_responses"]:
            result = robust_reward_v2(
                question=item["question"],
                correct_answer=item["ground_truth"],
                reasoning_process=item["best_responses"]
            )
            result_entry = {
                "question": item["question"],
                "ground_truth": item["ground_truth"],
                "best_responses": item["best_responses"],
                "current_action": item["current_action"],
                "evaluation": result
            }
            append_result(result_entry, output_file)
            print(f"Result appended to {output_file}")
