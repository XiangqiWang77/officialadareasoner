import sys
import os
import json

PROJECT_ROOT = os.path.join(os.path.dirname(__file__), "..")
sys.path.append(PROJECT_ROOT)
sys.path.append('./baselines/auto-cot-main')
sys.argv=['']
from api import cot

import concurrent.futures
from tqdm import tqdm


def evaluate_auto_cot(task, shots, model, temperature=0.1):
    output_dir = 'o3results/auto_cot'
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    output_file = os.path.join(output_dir, f'{model}.json')

    # Process a single question
    def process_question(question_data):
        response = cot(method="zero_shot", question=question_data['question'], model=model, temperature=temperature)
        print(response)
        reasoning_steps = response.split("\n")
        correct_option = question_data['answer']
        result = {
            "question": question_data['question'],
            "correct_option": correct_option,
            "predicted_reasoning": reasoning_steps,
        }
        return result

    test_questions = task[shots:]
    results = []
    futures = []
    with concurrent.futures.ThreadPoolExecutor() as executor:
        # Submit all tasks
        for question_data in test_questions:
            futures.append(executor.submit(process_question, question_data))
        # Process results as they complete
        for future in tqdm(concurrent.futures.as_completed(futures),
                           total=len(futures),
                           desc="Processing With Auto CoT"):
            results.append(future.result())

    # Write all results to output_file
    with open(output_file, "w") as f:
        json.dump(results, f, indent=4)
    return 0


# with open('data/Combine_of_5.json', 'r', encoding='utf-8') as f:
#     all_task = json.load(f)

# accuracy = evaluate_auto_cot(
#     task=all_task,
#     shots=1200,
#     model="gpt-4o-mini",
#     temperature=1.0
# )
