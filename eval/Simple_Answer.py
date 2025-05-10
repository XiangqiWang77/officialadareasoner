import numpy as np
import json
import os
import sys
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(PROJECT_ROOT)
from module.tools import get_response
import concurrent.futures
from tqdm import tqdm

# Load the dataset

prompt_template = (
    "Please think short and respond quick in a divergent way."
    "Question: {question}\n\n"
    "Requirments:"
    "1. Provide one answer that completely satisfies the question's requirements.\n"
    "2. Ensure your reasoning strictly adheres to the specified steps and covers all necessary details.\n"
    "3. Deliver a clear, precise, and accurate answer.\n"
    "4. Avoid repetition or ambiguity; your response should be distinct and well-reasoned.\n\n"
)

def evaluate_simple_answer(task, shots, model, temperature=0.1):
    test_questions = task[shots:]
    
    def process_question(question_data):
        prompt = prompt_template.format(question=question_data['question'])
        response = get_response(model=model, prompt=prompt, temperature=temperature)
        predicted_reasoning = response.split("\n") if response else None
        result = {
            "question": question_data['question'],
            "correct_option": question_data['answer'],
            "predicted_reasoning": predicted_reasoning,
        }
        return result

    results = []
    futures = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=5) as executor:
        # Submit all tasks
        for question_data in test_questions:
            futures.append(executor.submit(process_question, question_data))
        # Process results as they complete
        for future in tqdm(concurrent.futures.as_completed(futures),
                           total=len(futures),
                           desc="Processing With Simple Answer"):
            results.append(future.result())
            
    output_file = f"limiresults1/simple_answer/{model}.json"
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    with open(output_file, "w") as f:
        json.dump(results, f, indent=4)
        
    return 0
