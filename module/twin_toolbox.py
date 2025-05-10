import numpy as np
from sentence_transformers import SentenceTransformer
import pickle
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt




import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


semantic_model = SentenceTransformer('all-MiniLM-L6-v2')

class FactorizedAdaptiveContextualMLPAgent:
    def __init__(self, step_lengths, prompts, embedding_dim, hidden_dim=64, learning_rate=0.001):
        
        self.step_lengths = step_lengths
        self.prompts = self.rank_prompts_by_tfidf_pca(prompts)  
        self.embedding_dim = embedding_dim
        self.hidden_dim = hidden_dim
        self.learning_rate = learning_rate

        
        self.temperature_options = np.array([0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0])
        
        
        self.num_steps = len(self.step_lengths)
        self.num_prompts = len(self.prompts)
        self.num_temps = len(self.temperature_options)
        
        
        self.W_shared = np.random.randn(embedding_dim, hidden_dim) * 0.01
        self.b_shared = np.random.randn(hidden_dim) * 0.01

        
        self.W_step = np.random.randn(hidden_dim, self.num_steps) * 0.01
        self.b_step = np.random.randn(self.num_steps) * 0.01

        # Prompt Head
        self.W_prompt = np.random.randn(hidden_dim, self.num_prompts) * 0.01
        self.b_prompt = np.random.randn(self.num_prompts) * 0.01

        # Temperature Head
        self.W_temp = np.random.randn(hidden_dim, self.num_temps) * 0.01
        self.b_temp = np.random.randn(self.num_temps) * 0.01

        self.token_limit = 250
        self.embedding_memory = []  
        self.training_logs = {
            "training_step": [],
            "reward": [],
            "step_length_distribution": [],  
            "prompt_distribution": [],  
            "temperature_distribution": []  
        }

        self.reward_history = []

    def rank_prompts_by_tfidf_pca(self, prompts):
        
        vectorizer = TfidfVectorizer()
        tfidf_matrix = vectorizer.fit_transform(prompts)
        pca = PCA(n_components=1)
        pca_embeddings = pca.fit_transform(tfidf_matrix.toarray()).flatten()
        sorted_prompts = [p for _, p in sorted(zip(pca_embeddings, prompts))]
        return sorted_prompts

    def forward(self, x):
        
        h = np.tanh(np.dot(x, self.W_shared) + self.b_shared)
        
        logits_step = np.dot(h, self.W_step) + self.b_step
        logits_prompt = np.dot(h, self.W_prompt) + self.b_prompt
        logits_temp = np.dot(h, self.W_temp) + self.b_temp
        return h, logits_step, logits_prompt, logits_temp

    def select_action(self, task_embedding, training_step=0, few_shot=True):
        
        x = task_embedding / (np.linalg.norm(task_embedding) + 1e-8)
        decay_rate = 0.0001
        temperature_tau = max(1.0 - decay_rate * training_step, 0.1)

        h, logits_step, logits_prompt, logits_temp = self.forward(x)
        
        
        exp_logits_step = np.exp((logits_step - np.max(logits_step)) / temperature_tau)
        probs_step = exp_logits_step / np.sum(exp_logits_step)
        if few_shot:
            step_index = np.random.choice(self.num_steps, p=probs_step)
        else:
            step_index = np.argmax(probs_step)
        chosen_step = self.step_lengths[step_index]

        
        exp_logits_prompt = np.exp((logits_prompt - np.max(logits_prompt)) / temperature_tau)
        probs_prompt = exp_logits_prompt / np.sum(exp_logits_prompt)
        if few_shot:
            prompt_index = np.random.choice(self.num_prompts, p=probs_prompt)
        else:
            prompt_index = np.argmax(probs_prompt)
        chosen_prompt = self.prompts[prompt_index]

        
        exp_logits_temp = np.exp((logits_temp - np.max(logits_temp)) / temperature_tau)
        probs_temp = exp_logits_temp / np.sum(exp_logits_temp)
        if few_shot:
            temp_index = np.random.choice(self.num_temps, p=probs_temp)
        else:
            temp_index = np.argmax(probs_temp)
        chosen_temp = self.temperature_options[temp_index]

        
        self.embedding_memory.append(x.copy())
        if len(self.embedding_memory) > 100:
            self.embedding_memory.pop(0)
        self.token_limit = min(250 + chosen_step * 20, 500)


        
        self.training_logs.setdefault("step_length_distribution", []).append(probs_step.tolist())
        self.training_logs.setdefault("prompt_distribution", []).append(probs_prompt.tolist())
        self.training_logs.setdefault("temperature_distribution", []).append(probs_temp.tolist())

       
        self.training_logs["training_step"].append(training_step)

        return (chosen_step, chosen_prompt, chosen_temp, self.token_limit)

    def update(self, task_embedding, reward, chosen_action, training_step):
        
        chosen_step, chosen_prompt, chosen_temp, _ = chosen_action
        
        step_index = self.step_lengths.index(chosen_step)
        prompt_index = self.prompts.index(chosen_prompt)
        temp_index = int(np.where(self.temperature_options == chosen_temp)[0][0])

        x = task_embedding / (np.linalg.norm(task_embedding) + 1e-8)
        decay_rate = 0.0001
        temperature_tau = max(1.0 - decay_rate * training_step, 0.1)
        
        
        h, logits_step, logits_prompt, logits_temp = self.forward(x)
        
        exp_logits_step = np.exp((logits_step - np.max(logits_step)) / temperature_tau)
        probs_step = exp_logits_step / np.sum(exp_logits_step)
        exp_logits_prompt = np.exp((logits_prompt - np.max(logits_prompt)) / temperature_tau)
        probs_prompt = exp_logits_prompt / np.sum(exp_logits_prompt)
        exp_logits_temp = np.exp((logits_temp - np.max(logits_temp)) / temperature_tau)
        probs_temp = exp_logits_temp / np.sum(exp_logits_temp)
        prob_step = probs_step[step_index]
        prob_prompt = probs_prompt[prompt_index]
        prob_temp = probs_temp[temp_index]
        
        chosen_prob = prob_step * prob_prompt * prob_temp

        
        self.reward_history.append(reward)
        if len(self.reward_history) > 100:
            self.reward_history.pop(0)
        reward_mean = np.mean(self.reward_history)
        reward_std = np.std(self.reward_history) + 1e-8
        norm_reward = (reward - reward_mean) / reward_std

        
        dlogits_step = np.zeros_like(logits_step)
        dlogits_step[step_index] = - norm_reward / (prob_step + 1e-8)
        dlogits_prompt = np.zeros_like(logits_prompt)
        dlogits_prompt[prompt_index] = - norm_reward / (prob_prompt + 1e-8)
        dlogits_temp = np.zeros_like(logits_temp)
        dlogits_temp[temp_index] = - norm_reward / (prob_temp + 1e-8)

        
        dW_step = np.outer(h, dlogits_step)
        db_step = dlogits_step
        dW_prompt = np.outer(h, dlogits_prompt)
        db_prompt = dlogits_prompt
        dW_temp = np.outer(h, dlogits_temp)
        db_temp = dlogits_temp

        
        dh_step = np.dot(self.W_step, dlogits_step)
        dh_prompt = np.dot(self.W_prompt, dlogits_prompt)
        dh_temp = np.dot(self.W_temp, dlogits_temp)
        dh_total = dh_step + dh_prompt + dh_temp
        
        dz_shared = dh_total * (1 - h**2)
        dW_shared = np.outer(x, dz_shared)
        db_shared = dz_shared

        
        self.W_shared += self.learning_rate * dW_shared
        self.b_shared += self.learning_rate * db_shared
        self.W_step   += self.learning_rate * dW_step
        self.b_step   += self.learning_rate * db_step
        self.W_prompt += self.learning_rate * dW_prompt
        self.b_prompt += self.learning_rate * db_prompt
        self.W_temp   += self.learning_rate * dW_temp
        self.b_temp   += self.learning_rate * db_temp
        
    def plot_reward_heatmap(self,save_path=None):
        
        training_logs = self.training_logs
            
        step_length_probs = np.array(training_logs["step_length_distribution"]).T  # shape: (num_steps, num_training_steps)
        prompt_probs = np.array(training_logs["prompt_distribution"]).T  # shape: (num_prompts, num_training_steps)
        temperature_probs = np.array(training_logs["temperature_distribution"]).T  # shape: (num_temps, num_training_steps)

        fig, axes = plt.subplots(1, 3, figsize=(18, 5))

        # Step Length Policy Heatmap
        sns.heatmap(step_length_probs, ax=axes[0], cmap="viridis", cbar=True, xticklabels=10, yticklabels=1)
        axes[0].set_title("Step Length Policy Distribution")
        axes[0].set_xlabel("Few-shot Step")
        axes[0].set_ylabel("Step Length Index")

        # Prompt Policy Heatmap
        sns.heatmap(prompt_probs, ax=axes[1], cmap="viridis", cbar=True, xticklabels=10, yticklabels=1)
        axes[1].set_title("Prompt Policy Distribution")
        axes[1].set_xlabel("Few-shot Step")
        axes[1].set_ylabel("Prompt Index")

        # Temperature Policy Heatmap
        sns.heatmap(temperature_probs, ax=axes[2], cmap="viridis", cbar=True, xticklabels=10, yticklabels=1)
        axes[2].set_title("Temperature Policy Distribution")
        axes[2].set_xlabel("Few-shot Step")
        axes[2].set_ylabel("Temperature Index")

        plt.tight_layout()
        if save_path:
            plt.savefig(save_path)
        #plt.show()



    def save_parameters(self, file_path):
        
        params = {
            'W_shared': self.W_shared,
            'b_shared': self.b_shared,
            'W_step': self.W_step,
            'b_step': self.b_step,
            'W_prompt': self.W_prompt,
            'b_prompt': self.b_prompt,
            'W_temp': self.W_temp,
            'b_temp': self.b_temp,
            'step_lengths': self.step_lengths,
            'prompts': self.prompts,
            'temperature_options': self.temperature_options,
            'embedding_dim': self.embedding_dim,
            'hidden_dim': self.hidden_dim,
            'learning_rate': self.learning_rate,
            'token_limit': self.token_limit,
            'embedding_memory': self.embedding_memory
        }
        with open(file_path, 'wb') as f:
            pickle.dump(params, f)

    def load_parameters(self, file_path):
        
        with open(file_path, 'rb') as f:
            params = pickle.load(f)
        self.W_shared = params['W_shared']
        self.b_shared = params['b_shared']
        self.W_step = params['W_step']
        self.b_step = params['b_step']
        self.W_prompt = params['W_prompt']
        self.b_prompt = params['b_prompt']
        self.W_temp = params['W_temp']
        self.b_temp = params['b_temp']
        self.step_lengths = params['step_lengths']
        self.prompts = params['prompts']
        self.temperature_options = params['temperature_options']
        self.embedding_dim = params['embedding_dim']
        self.hidden_dim = params['hidden_dim']
        self.learning_rate = params['learning_rate']
        self.token_limit = params['token_limit']
        self.embedding_memory = params['embedding_memory']

