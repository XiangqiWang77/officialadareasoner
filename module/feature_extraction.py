import os
import numpy as np
import torch
from transformers import AutoTokenizer, AutoModelForMaskedLM
from sklearn.decomposition import PCA
tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
hf_model = AutoModelForMaskedLM.from_pretrained("bert-base-uncased")

def extract_features(question):
    
    inputs = tokenizer(question, return_tensors="pt", truncation=True, max_length=512)
    with torch.no_grad():
        embeddings = hf_model.bert(**inputs).last_hidden_state.mean(dim=1).squeeze().numpy()
    return embeddings

def debias_embeddings(embeddings):
    
    pca = PCA(n_components=1)
    bias_direction = pca.fit(embeddings).components_[0]
    return embeddings - np.outer(embeddings @ bias_direction, bias_direction)
                                