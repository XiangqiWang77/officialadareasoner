from transformers import pipeline


judge_classifier = pipeline("zero-shot-classification", model="facebook/bart-large-mnli")


import numpy as np
def add_gaussian_perturbation(x, sigma=0.1):
    perturbation = np.random.normal(0, sigma)
    x_perturbed = x + perturbation
    return np.clip(x_perturbed, 0.0, 1.0)


def compute_judge_score(question: str, candidate: str, reference: str) -> float:
    
    prompt = (
        f"Question: {question}\n"
        f"Candidate Answer: {candidate}\n"
        f"Ground Truth: {reference}\n"
        "Does the candidate answer match the ground truth? Answer yes or no."
    )
    candidate_labels = ["yes", "no"]
    
    
    result = judge_classifier(prompt, candidate_labels=candidate_labels)
    
    
    yes_score = 0.0
    for label, score in zip(result["labels"], result["scores"]):
        if label.lower() == "yes":
            yes_score = score
            break
            
    return yes_score

def evaluate_responses1(question: str, reference: str, candidate: str) -> float:
    

    
    score = compute_judge_score(question, candidate, reference)
    
    print("Judge Score:", score)
    return score

