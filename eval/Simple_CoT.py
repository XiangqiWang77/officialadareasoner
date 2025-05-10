import numpy as np
import json
import os
import sys
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(PROJECT_ROOT)

import concurrent.futures
from module.tools import get_response
from tqdm import tqdm  # Added tqdm import

# Prompt template
prompt_template = (
    "Please think step by step to solve the question"
    "Question: {question}\n\n"
    "Requirments:"
    "1. Provide one answer that completely satisfies the question's requirements.\n"
    "2. Ensure your reasoning strictly adheres to the specified steps and covers all necessary details.\n"
    "3. Deliver a clear, precise, and accurate answer.\n"
    "4. Avoid repetition or ambiguity; your response should be distinct and well-reasoned.\n\n"
)


def evaluate_simple_cot(task, shots, model, temperature=0.1):
    output_file = f'limiresults1/simple_cot/{model}.json'
    if not os.path.exists('limiresults1/simple_cot'):
        os.makedirs('limiresults1/simple_cot')
    # Process a single question
    def process_question(question_data):
        prompt = prompt_template.format(question=question_data['question'])
        # Call get_response with assumed parameters (adjust as needed)
        response = get_response(model=model, prompt=prompt, temperature=temperature)
        if not response:
            # API did not return anything, skip with answer as None
            predicted_reasoning = None
        else:
            predicted_reasoning = response.split("\n")
        correct_option = question_data['answer']
        result = {
            "question": question_data['question'],
            "correct_option": correct_option,
            "predicted_reasoning": predicted_reasoning,
        }
        return result

    results = []
    test_questions = task[shots:]
    print(len(task))
    futures = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=5) as executor:
        # Submit all tasks
        for question_data in test_questions:
            futures.append(executor.submit(process_question, question_data))
        # Process results as they complete
        for future in tqdm(concurrent.futures.as_completed(futures),
                           total=len(futures),
                           desc="Processing With Simple CoT"):
            results.append(future.result())
            
    # Write all results to output_file
    with open(output_file, "w") as f:
        json.dump(results, f, indent=4)
        
    return 0
