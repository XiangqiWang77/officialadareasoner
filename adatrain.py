import numpy as np
import json
import os
from transformers import AutoTokenizer, AutoModelForMaskedLM
from module.feature_extraction import extract_features
from module.Judge_reward import evaluate_responses1
from module.twin_toolbox import FactorizedAdaptiveContextualMLPAgent
from module.tools import get_response
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from scipy.stats import gaussian_kde
from sentence_transformers import SentenceTransformer, util
from sklearn.cluster import AgglomerativeClustering
from collections import defaultdict
import asyncio
from transformers import AutoModelForSequenceClassification



def load_questions(file_path):
    with open(file_path, 'r') as f:
        return json.load(f)

all_task = load_questions("./dataset/source.json")


embedding_dim = 768
step_lengths = list(range(3, 10))  # 0 to 9

# Prompt pool replaced with a dynamic generator
# Ultimate Enhanced Base Templates for Diverse and Robust Reasoning:
base_templates = [
    "Break down your reasoning into clear, sequential steps: {variation}",
    "Systematically structure your analysis, elaborating on each step with thorough detail: {variation}",
    "Examine the logical connections between concepts and articulate each step in depth: {variation}",
    "Consider multiple perspectives and explore alternative viewpoints comprehensively: {variation}",
    "Apply creative reasoning to unearth unconventional insights and challenge standard assumptions: {variation}",
    "Adopt a detailed and rigorous approach, balancing specific details with overarching themes: {variation}",
    "Reflect on your assumptions and refine your argument through critical self-questioning and validation: {variation}",
    "Explain your reasoning step-by-step in a clear, accessible manner for all audiences: {variation}",
    "Include a systematic self-check and verification of your reasoning process to ensure consistency: {variation}",
    "Conclude by summarizing your key points and re-evaluating your final answer for completeness: {variation}"
]

# Ultimate Enhanced Variations for Maximum Reasoning Diversity:
variations = [
    "Thoroughly analyze all possible interpretations to guarantee a comprehensive understanding.",
    "Decompose the problem into smaller, logical components to enhance clarity and precision.",
    "Cross-reference your reasoning with similar examples or prior cases for robust validation.",
    "Review and double-check each reasoning step to ensure no key detail is overlooked.",
    "Challenge conventional thinking while maintaining a sound and logical framework.",
    "Ensure every premise is clearly understood and meticulously applied.",
    "Pay close attention to minor details that might otherwise be neglected, ensuring depth in your analysis.",
    "Explain your reasoning in simple, straightforward language to guarantee clarity and accessibility.",
    "Perform a detailed self-audit of your reasoning to detect and correct any inconsistencies.",
    "Validate your conclusions by aligning them with established principles or empirical data."
]


def prompt_generator(base_templates, variations):
    """
    Generate diverse prompts using templates and variations.
    """
    generated_prompts = []
    for template in base_templates:
        for variation in variations:
            generated_prompts.append(template.format(variation=variation))
    return generated_prompts

dynamic_prompts = prompt_generator(base_templates, variations)

few_shot_ts = FactorizedAdaptiveContextualMLPAgent(
    step_lengths,
    dynamic_prompts,
    embedding_dim
)

# Initialize BERT-bas
# model for reasoning evaluation
tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
hf_model = AutoModelForMaskedLM.from_pretrained("bert-base-uncased")

# Prompt tem

prompt_template = (
    "### 1. Objective\n"
    "Your task is to generate a comprehensive answer to the provided question while tailoring your reasoning and response style to the specific demands of the task. Ensure that your answer fully adheres to the requirements without inventing any details.\n\n"

    "### 2. Question\n"
    "{question}\n\n"

    "### 3. Adaptive Reasoning Strategy\n"
    "Use the following instructions to shape your response: {instruction_prompt}\n\n"
    "Adjust your reasoning approach dynamically based on the nature of the question:\n"
    "- **For creative or diversity-oriented tasks (e.g., metaphors, humor, lateral thinking):** Employ rapid, divergent thinking. Start your answer with distinct and original ideas.\n"
    "- **For analytical, factual, or precision-demanding tasks (e.g., logical reasoning, scientific inquiry, structured problem-solving):** Follow a rigorous, step-by-step process where the conclusion emerges naturally rather than being stated upfront.\n\n"
    "You must follow no more than {optimal_steps} reasoning steps\n"
    "Only when it comes to creative questions (e.g. metaphor), you should and are required allowed to just say Yes or No, or the correct option with nothing else. But you still need to follow the requirements above."

    "### 4. Output Requirements\n"
    "1. Deliver a single, well-structured answer that completely addresses the question.\n"
    "2. Maintain logical consistency, ensuring each step meaningfully contributes to the final conclusion.\n"
    "3. Prioritize clarity, precision, and coherence in your response.\n"
    "4. Avoid redundancy and ambiguity—the answer should be distinct, well-organized, and logically sound.\n\n"

)






def extract_embeddings_and_params(model):
    embeddings = []
    step_lengths = []
    prompts = []
    temperatures = []
    token_limits = []

    for (step_length, prompt), data in model.models.items():
        embeddings.append(data['mu'])
        step_lengths.append(step_length)
        prompts.append(hash(prompt) % 100)  # Map prompt to unique ID
        temperatures.append(model.temperature)
        token_limits.append(model.token_limit)

    return np.array(embeddings), step_lengths, prompts, temperatures, token_limits



# Initialize semantic model for embedding-based evaluations
semantic_model = SentenceTransformer('all-MiniLM-L6-v2')

# Initialize HuggingFace model for logical evaluation
logic_tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
logic_model = AutoModelForSequenceClassification.from_pretrained("bert-base-uncased")




class NumpyEncoder(json.JSONEncoder):
    """
    Custom JSON encoder to handle non-serializable types such as numpy types.
    """
    def default(self, obj):
        if isinstance(obj, (np.integer, np.int32, np.int64)):
            return int(obj)
        elif isinstance(obj, (np.floating, np.float32, np.float64)):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        return super().default(obj)


async def run_pipeline_async(question_data, prompt_template, prompt, step_length, temperature, eval_model, num_steps=1, tasks_per_step=1):
    
    all_outputs = []
    loop = asyncio.get_running_loop()
    for step in range(1, num_steps + 1):
        
        previous_outputs = all_outputs if all_outputs else ""
        formatted_prompt = prompt_template.format(
            question=question_data['question'],
            instruction_prompt=prompt,
            optimal_steps=step_length
        )


        # Use asyncio.gather and run_in_executor to call get_response
        tasks = [loop.run_in_executor(None, get_response, eval_model, formatted_prompt, temperature) # use eval_model here
                 for _ in range(tasks_per_step)]
        step_outputs = await asyncio.gather(*tasks)

        print(f"Step {step} outputs: {step_outputs}")
        all_outputs.extend(step_outputs)

    return all_outputs


def evaluate_few_shot_with_multiple_responses(ts_model, task, shots, eval_model): # Added eval_model as input
    """
    Evaluate using Thompson Sampling with iterative response generation, selection, and hyperparameter tuning.
    After few-shot evaluation, apply the trained model to the full dataset.
    """
    print("Let's do it.")
    correct_counts = []

    output_dir = "./result/"
    os.makedirs(output_dir, exist_ok=True) # Create directory if not exists
    output_file = os.path.join(output_dir, f"{eval_model}.json") # Dynamic output file path

    with open(output_file, "w") as f:
        f.write("[\n")

        # Few-shot training and evaluation
        for idx, question_data in enumerate(task[:shots]):
            context = extract_features(question_data['question'])
            ground_truth = question_data['answer']
            best_reward = -999  # Initialize reward as -1
            attempts = 0
            best_responses = []  # Store generated responses iteratively
            current_action = None

            #while len(best_responses) < 5:
            attempts += 1

            epochs=5

            # Select action
            for epoch in range(1, epochs):
                current_action = ts_model.select_action(context,idx,few_shot=True)
                step_length, prompt, temperature, token_limit = current_action

                best_responses = asyncio.run(run_pipeline_async(question_data, prompt_template, prompt, step_length, temperature, eval_model)) # Pass eval_model here

            # Evaluate and update model with the best response
                if best_responses:
                    reward = evaluate_responses1(question_data['question'], ground_truth, best_responses)
                    #tem_action=step_length, prompt, temperature
                    ts_model.update(context, reward, current_action,idx)

                    print(f"Few-shot Question {idx + 1}: Best Reward: {reward}")
                    #print(f"Best Responses: {best_responses}")

                    # Save few-shot results
                    result = {
                        "question": question_data['question'],
                        "ground_truth": question_data['answer'],
                        "best_responses": best_responses,
                        #"best_candidate": best_candidate,
                        "best_reward": float(reward),
                        "attempts": int(attempts),
                        "current_action": [
                            int(current_action[0]) if isinstance(current_action[0], np.integer) else current_action[0],
                            str(current_action[1]),
                            float(current_action[2]),
                            int(current_action[3])
                        ]
                    }
                    f.write(json.dumps(result, indent=4, cls=NumpyEncoder))
                    if idx < len(task[:shots]) - 1 or shots < len(task):
                        f.write(",\n")

        print("Few-shot training completed.")

        ts_model.save_parameters('./gpt4o.pkl')


        f.write("\n]")  # Close the JSON array

    print("Full dataset application completed.")


eval_model_name = "gpt-4o"
accuracy = evaluate_few_shot_with_multiple_responses(
    ts_model=few_shot_ts,
    task=all_task,
    shots=100,
    eval_model=eval_model_name # Pass eval_model here
)