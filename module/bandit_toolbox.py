import numpy as np
from sentence_transformers import SentenceTransformer
import pickle
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import seaborn as sns

semantic_model = SentenceTransformer('all-MiniLM-L6-v2')

class BanditHead:
    def __init__(self, num_arms, context_dim, alpha=1.0):   
        self.num_arms = num_arms
        self.context_dim = context_dim
        self.alpha = alpha
        self.A = [np.eye(context_dim) for _ in range(num_arms)]
        self.b = [np.zeros(context_dim) for _ in range(num_arms)]
    
    def predict(self, context):
        scores = []
        for i in range(self.num_arms):
            A_inv = np.linalg.inv(self.A[i])
            theta = A_inv.dot(self.b[i])
            bonus = self.alpha * np.sqrt(context.dot(A_inv).dot(context))
            score = theta.dot(context) + bonus
            scores.append(score)
        return np.array(scores)

    def select_arm(self, context, training_step, few_shot=True):
        scores = self.predict(context)
        if few_shot:
            temperature_tau = max(1.0 - 0.0001 * training_step, 0.1)
            exp_scores = np.exp((scores - np.max(scores)) / temperature_tau)
            probs = exp_scores / np.sum(exp_scores)
            chosen_arm = np.random.choice(self.num_arms, p=probs)
        else:
            chosen_arm = np.argmax(scores)
        return chosen_arm, scores

    def update(self, arm_index, context, reward):
        self.A[arm_index] += np.outer(context, context)
        self.b[arm_index] += reward * context

class FactorizedContextualAdaptiveBandit:
    def __init__(self, step_lengths, prompts, embedding_dim, hidden_dim=64, learning_rate=0.001, bandit_alpha=1.0):
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

       
        self.step_bandit = BanditHead(self.num_steps, hidden_dim, alpha=bandit_alpha)
        self.prompt_bandit = BanditHead(self.num_prompts, hidden_dim, alpha=bandit_alpha)
        self.temp_bandit = BanditHead(self.num_temps, hidden_dim, alpha=bandit_alpha)

        self.token_limit = 250
        self.embedding_memory = []  
        self.training_logs = {
            "training_step": [],
            "reward": [],
            "step_score_distribution": [],
            "prompt_score_distribution": [],
            "temperature_score_distribution": []
        }
        self.reward_history = []

    def rank_prompts_by_tfidf_pca(self, prompts):
        
        vectorizer = TfidfVectorizer()
        tfidf_matrix = vectorizer.fit_transform(prompts)
        pca = PCA(n_components=1)
        pca_embeddings = pca.fit_transform(tfidf_matrix.toarray()).flatten()
        sorted_prompts = [p for _, p in sorted(zip(pca_embeddings, prompts))]
        return sorted_prompts

    def shared_forward(self, x):
        
        x_norm = x / (np.linalg.norm(x) + 1e-8)
        h = np.tanh(np.dot(x_norm, self.W_shared) + self.b_shared)
        return h

    def select_action(self, task_embedding, training_step=0, few_shot=True):
        
        h = self.shared_forward(task_embedding)

        step_index, step_scores = self.step_bandit.select_arm(h, training_step, few_shot)
        prompt_index, prompt_scores = self.prompt_bandit.select_arm(h, training_step, few_shot)
        temp_index, temp_scores = self.temp_bandit.select_arm(h, training_step, few_shot)

        chosen_step = self.step_lengths[step_index]
        chosen_prompt = self.prompts[prompt_index]
        chosen_temp = self.temperature_options[temp_index]

        
        self.embedding_memory.append(h.copy())
        if len(self.embedding_memory) > 100:
            self.embedding_memory.pop(0)
        self.token_limit = min(250 + chosen_step * 20, 500)

        
        self.training_logs.setdefault("step_score_distribution", []).append(step_scores.tolist())
        self.training_logs.setdefault("prompt_score_distribution", []).append(prompt_scores.tolist())
        self.training_logs.setdefault("temperature_score_distribution", []).append(temp_scores.tolist())
        self.training_logs["training_step"].append(training_step)

        return chosen_step, chosen_prompt, chosen_temp, self.token_limit

    def update(self, context, reward, current_action, idx, few_shot=True):
        
        if not few_shot:
            return

        
        h = self.shared_forward(context)

        if idx == 0:
            
            chosen_value = current_action[0]
            arm_index = self.step_lengths.index(chosen_value)
            self.step_bandit.update(arm_index, h, reward)
        elif idx == 1:
            
            chosen_value = current_action[1]
            arm_index = self.prompts.index(chosen_value)
            self.prompt_bandit.update(arm_index, h, reward)
        elif idx == 2:
            
            chosen_value = current_action[2]
            arm_index = int(np.where(self.temperature_options == chosen_value)[0][0])
            self.temp_bandit.update(arm_index, h, reward)


    def plot_reward_heatmap(self, save_path=None):
        
        logs = self.training_logs
        step_scores = np.array(logs["step_score_distribution"]).T
        prompt_scores = np.array(logs["prompt_score_distribution"]).T
        temp_scores = np.array(logs["temperature_score_distribution"]).T

        fig, axes = plt.subplots(1, 3, figsize=(18, 5))
        sns.heatmap(step_scores, ax=axes[0], cmap="viridis", cbar=True, xticklabels=10, yticklabels=1)
        axes[0].set_title("Step Score Distribution")
        axes[0].set_xlabel("Few-shot Step")
        axes[0].set_ylabel("Step Index")

        sns.heatmap(prompt_scores, ax=axes[1], cmap="viridis", cbar=True, xticklabels=10, yticklabels=1)
        axes[1].set_title("Prompt Score Distribution")
        axes[1].set_xlabel("Few-shot Step")
        axes[1].set_ylabel("Prompt Index")

        sns.heatmap(temp_scores, ax=axes[2], cmap="viridis", cbar=True, xticklabels=10, yticklabels=1)
        axes[2].set_title("Temperature Score Distribution")
        axes[2].set_xlabel("Few-shot Step")
        axes[2].set_ylabel("Temperature Index")

        plt.tight_layout()
        if save_path:
            plt.savefig(save_path)
        # plt.show()

    def save_parameters(self, file_path):
        
        params = {
            'W_shared': self.W_shared,
            'b_shared': self.b_shared,
            'step_bandit_A': self.step_bandit.A,
            'step_bandit_b': self.step_bandit.b,
            'prompt_bandit_A': self.prompt_bandit.A,
            'prompt_bandit_b': self.prompt_bandit.b,
            'temp_bandit_A': self.temp_bandit.A,
            'temp_bandit_b': self.temp_bandit.b,
            'step_lengths': self.step_lengths,
            'prompts': self.prompts,
            'temperature_options': self.temperature_options,
            'embedding_dim': self.embedding_dim,
            'hidden_dim': self.hidden_dim,
            'token_limit': self.token_limit,
            'embedding_memory': self.embedding_memory,
            'training_logs': self.training_logs,
            'reward_history': self.reward_history
        }
        with open(file_path, 'wb') as f:
            pickle.dump(params, f)

    def load_parameters(self, file_path):
        
        with open(file_path, 'rb') as f:
            params = pickle.load(f)
        self.W_shared = params['W_shared']
        self.b_shared = params['b_shared']
        self.step_bandit.A = params['step_bandit_A']
        self.step_bandit.b = params['step_bandit_b']
        self.prompt_bandit.A = params['prompt_bandit_A']
        self.prompt_bandit.b = params['prompt_bandit_b']
        self.temp_bandit.A = params['temp_bandit_A']
        self.temp_bandit.b = params['temp_bandit_b']
        self.step_lengths = params['step_lengths']
        self.prompts = params['prompts']
        self.temperature_options = params['temperature_options']
        self.embedding_dim = params['embedding_dim']
        self.hidden_dim = params['hidden_dim']
        self.token_limit = params['token_limit']
        self.embedding_memory = params['embedding_memory']
        self.training_logs = params['training_logs']
        self.reward_history = params['reward_history']
