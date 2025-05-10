import numpy as np
import json
import os
import sys
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(PROJECT_ROOT)

import concurrent.futures
from module.tools import get_response
from tqdm import tqdm  

# Prompt Template
prompt_template = (
    "Please think step by step to solve the question. "
    "Question: {question}\n\n"
    "Requirments:"
    "1. Provide one answer that completely satisfies the question's requirements.\n"
    "2. Ensure your reasoning strictly adheres to the specified steps and covers all necessary details.\n"
    "3. Deliver a clear, precise, and accurate answer.\n"
    "4. Avoid repetition or ambiguity; your response should be distinct and well-reasoned.\n\n"
)

def select_best_response(responses):
    """
    Simple heuristic: choose the answer with the most characters.
    If all answers are None, return None.
    """
    valid_responses = [r for r in responses if r is not None]
    if not valid_responses:
        return None
    # Calculate string length for each answer (joining list elements)
    best = max(valid_responses, key=lambda r: len("".join(r)))
    return best

def evaluate_responses(model, question, responses, temperature=1.0):
    # Helper to evaluate a single response and extract detailed scores concurrently.
    def evaluate_one(r):
        if r is None:
            return (r, float("-inf"), {})  # No response => worst score.
        reasoning_process = "\n".join(r)
        # Prepare evaluation prompt inline (same as in improved_dense_reward)
        prompt = (
            f"Question: {question}\n\n"
            "You will evaluate the following reasoning process based on specific dimensions.\n"
            "Then you will provide a set of scores in a specific plain-text format that resembles JSON.\n\n"
            "Evaluation Dimensions:\n"
            "1. **Logical Flaw** (0 to 1): Assess the logical consistency of the reasoning.\n"
            "2. **Coverage** (0 to 1): Evaluate how thoroughly the reasoning addresses all relevant aspects.\n"
            "3. **Confidence** (0 to 1): Rate your confidence in the correctness of the reasoning.\n"
            "4. **Rationale** (0 to 1): Judge the overall explanation and clarity.\n\n"
            "Output Requirements:\n"
            "1. Provide a single set of scores using the following format (plain text, not valid JSON):\n"
            "{\n"
            '  "LogicalFlaw": 0.9,\n'
            '  "Coverage": 0.85,\n'
            '  "Confidence": 0.9,\n'
            '  "Rationale": 0.95\n'
            "}\n"
            "2. Do not deviate from the defined dimensions or format.\n\n"
            "Reasoning Process: " + reasoning_process + "\n"
        )
        eval_response = get_response(model=model, prompt=prompt, temperature=temperature)
        try:
            evaluation = json.loads(eval_response)
            keys = ["LogicalFlaw", "Coverage", "Confidence", "Rationale"]
            avg = sum(evaluation[k] for k in keys) / len(keys)
            details = {k: evaluation[k] for k in keys}
        except Exception:
            avg = -1.0
            details = {}
        return (r, avg, details)
    
    results = []
    with concurrent.futures.ThreadPoolExecutor() as executor:
        futures = [executor.submit(evaluate_one, r) for r in responses]
        for future in concurrent.futures.as_completed(futures):
            results.append(future.result())
    
    # Select best based on average score. In case of ties, the first is chosen.
    best_tuple = max(results, key=lambda tup: tup[1]) if results else (None, None, None)
    best_response = best_tuple[0]
    # Build a detailed responses list containing each response, average score, and per-dimension scores.
    detailed_responses = [{
        "response": res,
        "average_score": avg,
        "detailed_scores": details
    } for res, avg, details in results]
    return best_response, detailed_responses

def evaluate_best_of_N(task, shots, model, temperature=1.0, N=3):
    """
    For the questions in the task, skip the first 'shots' examples,
    then call get_response N times for each question.
    """
    output_dir = 'o3results/best_of_N'
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    output_file = os.path.join(output_dir, f'{model}.json')
    
    cached_dict = {}
    if os.path.exists(output_file):
        with open(output_file, "r", encoding="utf-8") as f:
            cached_results = json.load(f)
        for entry in cached_results:
            cached_dict[entry["question"]] = entry.get("all_responses", [])
    
    def process_question(question_data):
        q_text = question_data['question']
        prompt = prompt_template.format(question=q_text)
        responses = []
        temperatures = np.linspace(0.1, 1.0, N)  # Evenly distribute temperatures based on N
        for i in range(N):
            current_temp = temperatures[i]
            response = get_response(model=model, prompt=prompt, temperature=current_temp)
            responses.append(None if not response else response.split("\n"))
        best_response, detailed_responses = evaluate_responses(model, q_text, responses, temperature=temperature)
        return {
            "question": q_text,
            "correct_option": question_data['answer'],
            "predicted_reasoning": best_response,
            "all_responses": detailed_responses,
        }

    results = []
    test_questions = task[shots:]
    with concurrent.futures.ThreadPoolExecutor() as executor:
        futures = []
        for question_data in test_questions:
            futures.append(executor.submit(process_question, question_data))
        for future in tqdm(concurrent.futures.as_completed(futures),
                           total=len(futures),
                           desc="Processing With Best-of-N"):
            results.append(future.result())
            
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=4, ensure_ascii=False)
        
    return 0

