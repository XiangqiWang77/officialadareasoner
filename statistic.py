import json
import pandas as pd


file_path = "./judge.json" 

with open(file_path, "r", encoding="utf-8") as file:
    data = json.load(file)


def classify_and_count(data, prefixes):
    results = {prefix: {'total': 0, 'positive': 0} for prefix in prefixes}
    
    for item in data:
        if 'question' in item and 'evaluation' in item:
            question = item['question']
            evaluation_str = item['evaluation']

            try:
                evaluation_data = json.loads(evaluation_str)
            except json.JSONDecodeError as e:
                print(f"JSONDecodeError for question '{question}': {e}")
                continue

            judge_value = evaluation_data.get("result", "").strip().lower()
            
            for prefix in prefixes:
                if question.startswith(prefix):
                    results[prefix]['total'] += 1
                    if 'yes' in judge_value:
                        results[prefix]['positive'] += 1
                    break
    
    return results


def calculate_accuracy(results):
    accuracy = {}
    for prefix, counts in results.items():
        total = counts['total']
        positive = counts['positive']
        accuracy[prefix] = positive / total if total > 0 else 0
    return accuracy



prefixes1 = [
        "For the logical question, please select the most likely a",
        "For the following context sentence, the metaphor word is conta",
        "For this math",
        "For the following question, please select the most likely ans"
    ]

pref=[
    "You are required to solve this spatial reasoni",
    "What is the summary of the following dialogue? A",
    "What is the emotion expressed "
]


results = classify_and_count(data, prefixes1)
accuracy = calculate_accuracy(results)


accuracy_df = pd.DataFrame.from_dict(results, orient='index')
accuracy_df["accuracy"] = accuracy_df["positive"] / accuracy_df["total"]


print(accuracy_df)
