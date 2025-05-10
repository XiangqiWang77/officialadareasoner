import json
import os
import re
import sys
PROJECT_ROOT = os.path.join(os.path.dirname(__file__), "..")
sys.path.append(PROJECT_ROOT)
import concurrent.futures
from tqdm import tqdm
from module.tools import get_response

def load_questions(file_path):
    with open(file_path, 'r', encoding='utf-8') as f:
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

def robust_parse_json(text):
    if not isinstance(text, str):
        print(f"Error: Input to robust_parse_json is not a string. Type: {type(text)}, Value: {text}") # Debug print
        return None  # Return None if input is not a string

    try:
        return json.loads(text)
    except:
        pattern = r'\{.*\}'
        match = re.search(pattern, text, re.DOTALL)
        if match:
            try:
                return json.loads(match.group())
            except:
                print("Regex extraction JSON parse error")
        return None
    

def reward(question, correct_answer, reasoning_process, model, temperature):
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
    response = get_response(model=model, prompt=prompt, temperature=temperature)
    result = robust_parse_json(response)
    # if result is None:
    #     stripped = response.strip().lower()
    #     if stripped in ["yes", "no"]:
    #         result = {"result": stripped, "details": ""}
    #     else:
    #         result = {"result": None, "details": "Response could not be parsed as JSON."}
    return result

# Helper function to process a single item
def process_item(item, judge_model, temperature):
    judge_result = reward(
        question=item["question"],
        correct_answer=item.get('correct_option') or item.get('ground_truth'),
        reasoning_process=item.get('predicted_reasoning') or item.get('best_candidate'),
        model=judge_model,
        temperature=temperature
    )
    result_entry = {
        "question": item["question"],
        "correct_option": item.get('correct_option') or item.get('ground_truth'),
        "predicted_reasoning": item.get('predicted_reasoning') or item.get('best_candidate'),
        "evaluation": judge_result
    }
    return result_entry

def evaluate_judge(task_type='simple_cot', source_model='gpt-4o-mini', judge_model='gpt-4o', temperature=0.7):
    input_file = f'o3results/{task_type}/{source_model}.json'
    output_dir = f'o3results/{task_type}'
    output_file = f'{output_dir}/{source_model}_judge.json'
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    data = load_questions(input_file)
    results = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=1) as executor:
        futures = []
        for item in data:
            futures.append(executor.submit(process_item, item, judge_model, temperature))
        for future in tqdm(concurrent.futures.as_completed(futures),
                           total=len(data),
                           desc="Concurrent Judge Processing"):
            results.append(future.result())
            
    with open(output_file, "w") as f:
        json.dump(results, f, indent=4)
    print(f"Results appended to {output_file}")

if __name__ == "__main__":
    task_type = 'tot'
    source_model = 'llama-3.1-8B'
    judge_model = 'chatgpt-4o-latest'
    temperature = 0.1
    evaluate_judge(task_type=task_type, source_model=source_model, judge_model=judge_model, temperature=temperature)
