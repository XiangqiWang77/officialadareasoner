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
    "In addition, we provide additional question and answer example pairs of this same type of question. Here they are {previousones}."
    "Requirments:"
    "1. Provide one answer that completely satisfies the question's requirements.\n"
    "2. Ensure your reasoning strictly adheres to the specified steps and covers all necessary details.\n"
    "3. Deliver a clear, precise, and accurate answer.\n"
    "4. Avoid repetition or ambiguity; your response should be distinct and well-reasoned.\n\n"
)

prefixes = [
    "For the logical question, please select the most likely a",
    "For the following context sentence, the metaphor word is conta",
    "For this math question, please select the right",
    "For the following question, please select the most likely ans",
    #"please solve this knowledge "
]




def evaluate_fewshot_cot(task, shots, model, temperature=0.1):
    output_file = f'o3results/fewshot_cot/{model}.json'
    if not os.path.exists('o3results/fewshot_cot'):
        os.makedirs('o3results/fewshot_cot')
    # Process a single question
    def process_question(question_data, shots=100):

        prefixes = [
            "please solve this knowledge based question: For the logical question, please select the most likely a",
            "please solve this knowledge based question: For the following context sentence, the metaphor word is conta",
            "please solve this knowledge based question: For this math question, please select the right",
            "please solve this knowledge based question: For the following question, please select the most likely ans",
            #"please solve this knowledge "
        ]
        
        matched_prefix = None
        for prefix in prefixes:
            print(prefix)
            if question_data['question'].startswith(prefix):
                matched_prefix = prefix
                break
        
        if not matched_prefix:
            print("No matching prefix found.")
            return None
        
        
        filtered_shots = [shot for shot in shots if shot['question'].startswith(matched_prefix)]
        selected_shots = filtered_shots[:10]  # 取前20个
        prompt = prompt_template.format(question=question_data['question'], previousones=selected_shots)
        # Call get_response with assumed parameters (adjust as need
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
    futures = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=5) as executor:
        # Submit all tasks
        for question_data in test_questions:
            futures.append(executor.submit(process_question, question_data, shots=task[:shots]))
        # Process results as they complete
        for future in tqdm(concurrent.futures.as_completed(futures),
                           total=len(futures),
                           desc="Processing With Fewshot CoT"):
            results.append(future.result())
            
    # Write all results to output_file
    with open(output_file, "w") as f:
        json.dump(results, f, indent=4)
        
    return 0
