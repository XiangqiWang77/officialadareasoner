# 🧠 AdaReasoner: Adaptive Reasoning Enables More Flexible Thinking

<div align="center">

[![arXiv](https://img.shields.io/badge/arXiv-2505.17312-b31b1b.svg)](https://arxiv.org/abs/2505.17312)
[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/release/python-380/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)


*Official implementation of "AdaReasoner: Adaptive Reasoning Enables More Flexible Thinking"*

[📖 Paper](https://arxiv.org/abs/2505.17312) • [🚀 Quick Start](#-quick-start) • [📊 Benchmarks](#-benchmarks) • [🔧 Installation](#-installation)

</div>

---

## 🌟 Overview

**AdaReasoner** is a groundbreaking reinforcement learning-based framework that revolutionizes how Large Language Models (LLMs) approach reasoning tasks. Unlike traditional static prompting methods, AdaReasoner dynamically adapts reasoning configurations—including instruction prompts, temperature settings, and reasoning step counts—based on the specific characteristics of each question.

### 🎯 Key Features

- **🔄 Dynamic Adaptation**: Automatically adjusts reasoning strategies based on question type and complexity
- **🎛️ Multi-dimensional Optimization**: Simultaneously optimizes prompts, temperature, and reasoning steps
- **🏆 Superior Performance**: Consistently outperforms baseline methods across multiple benchmarks
- **🔌 Plug-and-Play**: Easy integration with existing LLM workflows
- **📈 Reinforcement Learning**: Uses Thompson Sampling for efficient exploration and exploitation

### 🧪 How It Works

AdaReasoner employs a sophisticated multi-armed bandit approach with contextual embeddings to:

1. **Extract Features**: Analyze question characteristics using BERT-based embeddings
2. **Select Strategy**: Choose optimal reasoning configuration using Thompson Sampling
3. **Generate Response**: Apply selected strategy to produce high-quality answers
4. **Learn & Adapt**: Update policy based on response quality feedback

---

## 🏗️ Repository Structure

```
AdaReasoner/
├── 📁 baselines/              # Baseline reasoning methods
│   └── auto-cot-main/         # Auto-CoT implementation
├── 📁 dataset/                # Training and evaluation datasets
│   └── source.json            # Mixed dataset (MMLU, LogiQA, Metaphor, TruthfulQA)
├── 📁 eval/                   # Evaluation modules for all methods
│   ├── auto_cot.py           # Auto-CoT evaluation
│   ├── best_of_N.py          # Best-of-N sampling
│   ├── fewshot_cot.py        # Few-shot Chain-of-Thought
│   ├── Simple_Answer.py      # Direct answering baseline
│   ├── Simple_CoT.py         # Simple Chain-of-Thought
│   └── ToT.py                # Tree-of-Thought evaluation
├── 📁 module/                 # Core AdaReasoner implementation
│   ├── bandit_toolbox.py     # Multi-armed bandit algorithms
│   ├── twin_toolbox.py       # Factorized adaptive agent
│   ├── feature_extraction.py # Question feature extraction
│   ├── Judge_reward.py       # Response evaluation
│   ├── tools.py              # LLM interaction utilities
│   └── utils.py              # Helper functions
├── 🎯 adatrain.py            # Training script for AdaReasoner
├── 🔍 adatest.py             # Inference with pretrained models
├── 📊 baseline_evaluation.py  # Comprehensive baseline comparison
├── 📈 statistic.py           # Results analysis and visualization
├── 🤖 LLM_as_Judge.py        # LLM-based evaluation
├── 🎲 gpt4oadapt.pkl         # Pretrained AdaReasoner for GPT-4o
├── 📋 requirements.txt       # Python dependencies
└── 📖 README.md              # This file
```

---

## 🔧 Installation

### Prerequisites

- Python 3.8 or higher
- CUDA-compatible GPU (recommended for faster training)
- OpenAI API key or Anthropic API key

### Step 1: Clone the Repository

```bash
git clone https://github.com/your-username/AdaReasoner.git
cd AdaReasoner
```

### Step 2: Create Virtual Environment

```bash
python -m venv adareasoner_env
source adareasoner_env/bin/activate  # On Windows: adareasoner_env\Scripts\activate
```

### Step 3: Install Dependencies

```bash
pip install -r requirements.txt
```

### Step 4: Set Up API Keys

Create a `.env` file in the root directory:

```bash
# For OpenAI models
OPENAI_API_KEY=your_openai_api_key_here

# For Anthropic models
ANTHROPIC_API_KEY=your_anthropic_api_key_here
```

---

## 🚀 Quick Start

### 🔍 Run Inference with Pretrained Model

Test AdaReasoner with our pretrained GPT-4o model:

```bash
python adatest.py
```

This will:
- Load the pretrained model from `gpt4oadapt.pkl`
- Evaluate on the test dataset in `dataset/source.json`
- Generate adaptive responses for each question
- Save results to `./testresult/gpt-4o1.json`

### 🎯 Train Your Own Model

Train AdaReasoner from scratch on your dataset:

```bash
python adatrain.py
```

Training process:
- Uses Thompson Sampling for policy optimization
- Employs DeBERTa-based reward model for response evaluation
- Saves trained model to `gpt4o.pkl`
- Results saved to `./result/gpt-4o.json`

### 📊 Compare with Baselines

Run comprehensive evaluation against baseline methods:

```bash
python baseline_evaluation.py
```

### 📈 Analyze Results

Generate statistics and visualizations:

```bash
python statistic.py
```



## 📊 Benchmarks

AdaReasoner demonstrates superior performance across diverse reasoning tasks:

### 🏆 Main Results (GPT-4o)

| Method | Metaphor | TruthfulQA | MMLU (Math) | LogiQA | **Average** |
|--------|----------|------------|-------------|---------|-------------|
| Direct Answer | 45.20 | 72.30 | 68.15 | 65.40 | 62.76 |
| Simple CoT | 50.40 | 78.40 | 76.04 | 70.00 | 68.71 |
| Auto-CoT | 62.33 | 83.09 | 72.15 | 71.71 | 72.32 |
| Tree-of-Thought | 58.90 | 80.25 | 78.92 | 74.15 | 73.06 |
| Best-of-N (N=5) | 65.78 | 84.12 | 81.33 | 76.89 | 77.03 |
| **AdaReasoner** | **71.56** | **81.30** | **86.49** | **82.31** | **80.42** |

### 📈 Performance Improvements

- **+8.1%** average improvement over best baseline
- **+11.8%** improvement on complex reasoning tasks
- **Consistent gains** across all evaluated LLMs

### 🎯 Cross-Model Evaluation

AdaReasoner shows robust performance across 6 different LLMs:
- GPT-4o, GPT-4, GPT-3.5-turbo
- Claude-3, Claude-3.5-Sonnet
- Llama-2-70B

---


## 📚 Usage Examples

### Basic Usage

```python
from module.twin_toolbox import FactorizedAdaptiveContextualMLPAgent
from module.feature_extraction import extract_features

# Initialize AdaReasoner
agent = FactorizedAdaptiveContextualMLPAgent(
    step_lengths=list(range(3, 10)),
    prompts=dynamic_prompts,
    embedding_dim=768
)

# Load pretrained model
agent.load_parameters('gpt4oadapt.pkl')

# Extract question features
question = "What is the relationship between quantum entanglement and information theory?"
context = extract_features(question)

# Select optimal strategy
step_length, prompt, temperature, token_limit = agent.select_action(context)

print(f"Selected strategy:")
print(f"  Steps: {step_length}")
print(f"  Temperature: {temperature}")
print(f"  Prompt: {prompt[:100]}...")
```

### Custom Training

```python
# Train on custom dataset
def train_custom_model(dataset_path, model_name="custom_model"):
    # Load your dataset
    with open(dataset_path, 'r') as f:
        custom_data = json.load(f)
    
    # Initialize and train
    agent = FactorizedAdaptiveContextualMLPAgent(...)
    
    for question_data in custom_data:
        context = extract_features(question_data['question'])
        action = agent.select_action(context)
        
        # Generate response and get reward
        response = generate_response(question_data, action)
        reward = evaluate_response(response, question_data['answer'])
        
        # Update policy
        agent.update(context, reward, action)
    
    # Save trained model
    agent.save_parameters(f'{model_name}.pkl')
```

---

## 🤝 Contributing

We welcome contributions! Please see our [Contributing Guidelines](CONTRIBUTING.md) for details.

### 🐛 Reporting Issues

- Use GitHub Issues for bug reports
- Include minimal reproduction examples
- Specify your environment details

### 🔧 Development Setup

```bash
# Install development dependencies
pip install -r requirements-dev.txt

# Run tests
python -m pytest tests/

# Format code
black .
isort .
```

---

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

## 📖 Citation

If you find AdaReasoner helpful in your research, please cite our paper:

```bibtex
@misc{wang2025adareasoneradaptivereasoningenables,
    title={AdaReasoner: Adaptive Reasoning Enables More Flexible Thinking}, 
    author={Xiangqi Wang and Yue Huang and Yanbo Wang and Xiaonan Luo and Kehan Guo and Yujun Zhou and Xiangliang Zhang},
    year={2025},
    eprint={2505.17312},
    archivePrefix={arXiv},
    primaryClass={cs.AI},
    url={https://arxiv.org/abs/2505.17312}, 
}
```

---



## 🙏 Acknowledgments

- Thanks to the open-source community for foundational tools
- Special thanks to the evaluation benchmark creators
- Supported by University of Notre Dame and MBZUAI

---

## 📞 Contact

For questions, suggestions, or collaborations:

- 📧 Email: [xwang76@nd.edu](mailto:xwang76@nd.edu)
- 🐛 Issues: [GitHub Issues](https://github.com/your-username/AdaReasoner/issues)
- 💬 Discussions: [GitHub Discussions](https://github.com/your-username/AdaReasoner/discussions)

---

<div align="center">

**⭐ Star this repository if you find it helpful! ⭐**

Made with ❤️ by the AdaReasoner team

</div>
