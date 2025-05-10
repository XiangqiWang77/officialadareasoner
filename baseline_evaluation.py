import os
import json
import csv
from eval.auto_cot import evaluate_auto_cot
from eval.best_of_N import evaluate_best_of_N
from eval.ToT import evaluate_tot
from eval.Simple_CoT import evaluate_simple_cot
from eval.Simple_Answer import evaluate_simple_answer
from eval.fewshot_cot import evaluate_fewshot_cot
from eval.judge import evaluate_judge

with open('./dataset/source.json', 'r', encoding='utf-8') as f:
    all_task = json.load(f)
    print(all_task)
prefixes = [
    "For the logical question, please select the most likely a",
    "For the following context sentence, the metaphor word is conta",
    "For this math question, please select the right",
    "For the following question, please select the most likely ans",
    #"please solve this knowledge "
]

#prefixes=[
#    "What is the summary of the following dialogue? Answ",
#    "What is the emotion expressed in the following t",
#    "You are required to solve this spatial reasoni"
#]

#prefixes = [
#            "please solve this knowledge based question: For the logical question, please select the most likely a",
#            "please solve this knowledge based question: For the following context sentence, the metaphor word is conta",
#            "please solve this knowledge based question: For this math question, please select the right",
#            "please solve this knowledge based question: For the following question, please select the most likely ans",
            #"please solve this knowledge "
#        ]

eval_funcs = {
    'auto_cot': evaluate_auto_cot,
    'best_of_N': evaluate_best_of_N,
    'tot': evaluate_tot,
    'simple_cot': evaluate_simple_cot,
    'simple_answer': evaluate_simple_answer,
    'fewshot_cot': evaluate_fewshot_cot
    #'adaptive_cot': None
}

def generate(model):
    for task_type, eval_func in eval_funcs.items():
        if eval_func is None:
            continue
        print(f"Evaluating {model} on {task_type}...")
        eval_func(
            model=model,
            task=all_task,
            shots=0
        )
        print(f"Completed evaluation on {task_type}.")

def eval(model):
    for task_type in eval_funcs.keys():
        print(f"Evaluating {model} on judge...")
        evaluate_judge(
            task_type=task_type,
            source_model=model,
            judge_model='gpt-4o',
            temperature=0.1
        )
        print(f"Completed evaluation on judge.")

def statistics(model):
    results_dir = "o3results"
    os.makedirs("o3results/statistics", exist_ok=True)

    overall_correct = 0
    overall_total = 0

    csv_data = [["Task Type", "Prefix", "Total", "Correct", "Accuracy"]]
    
    task_type_stats = {}

    for task_type in eval_funcs.keys():
        task_dir = f"{results_dir}/{task_type}"
        result_file = f"{task_dir}/{model}_judge.json"

        result = {prefix: {"total": 0, "correct": 0} for prefix in prefixes}
        task_correct = 0
        task_total = 0

        try:
            with open(result_file, 'r', encoding='utf-8') as f:
                data = json.load(f)
        except FileNotFoundError:
            print(f"Warning: {result_file} not found. Skipping...")
            continue
        
        data = data[-900:]

        for task in data:
            for prefix in prefixes:
                if prefix in task['question']:
                    result[prefix]["total"] += 1
                    task_total += 1
                    if task['evaluation']['result'].strip().lower() == "yes":
                        result[prefix]["correct"] += 1
                        task_correct += 1
                    break

        task_type_stats[task_type] = {"total": task_total, "correct": task_correct}

        print(f"Statistics for {model} on {task_type}:")
        for prefix, stats in result.items():
            total = stats["total"]
            correct = stats["correct"]
            accuracy = (correct / total) if total > 0 else 0
            print(f"  {prefix[:30]}... -> Total: {total}, Correct: {correct}, Accuracy: {accuracy:.2%}")

            csv_data.append([task_type, prefix, total, correct, f"{accuracy:.2%}"])

        print(f"  Overall for {task_type} -> Total: {task_total}, Correct: {task_correct}, Accuracy: {(task_correct / task_total) if task_total > 0 else 0:.2%}")
        csv_data.append([task_type, "Overall", task_total, task_correct, f"{(task_correct / task_total) if task_total > 0 else 0:.2%}"])

        overall_correct += task_correct
        overall_total += task_total

    overall_accuracy = (overall_correct / overall_total) if overall_total > 0 else 0
    csv_data.append(["Overall", "All Task Types", overall_total, overall_correct, f"{overall_accuracy:.2%}"])
    print(f"\nOverall Accuracy for {model}: {overall_accuracy:.2%}")

    csv_path = f"results/statistics/{model}.csv"
    with open(csv_path, 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerows(csv_data)

    print(f"Statistics saved to {csv_path}")

if __name__ == "__main__":
    model = "o3-mini"
    generate(model)
    eval(model)
    statistics(model)
    

