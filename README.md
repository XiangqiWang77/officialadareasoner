# AdaReasoner: Adaptive Reasoning Enables More Flexible Thinking

Official implementation of the paper:

> **AdaReasoner: Adaptive Reasoning Enables More Flexible Thinking**
> Xiangqi Wang, Yue Huang, Yanbo Wang, Xiaonan Luo, Kehan Guo, Yujun Zhou, Xiangliang Zhang
> University of Notre Dame & MBZUAI Contact: [xwang76@nd.edu](mailto:xwang76@nd.edu)

---

## Overview

**AdaReasoner** is a lightweight, reinforcement learning–based plugin designed to dynamically adapt reasoning configurations for Large Language Models (LLMs). Unlike traditional prompting methods that rely on fixed settings, AdaReasoner enables LLMs to adjust reasoning instructions, temperature, and step count based on the question type — thereby achieving **more flexible and task-specific thinking**.

This repository provides the official codebase for training, evaluating, and running AdaReasoner across various reasoning tasks.

---

## Repository Structure

OFFICIAL_ADAREASONER/

├── baselines/                # Baseline reasoning methods (e.g., CoT, ToT)

├── dataset/                 # Contains source.json and data loaders

├── eval/                    # Evaluation utilities of all baselines

├── module/                  # Core RL policy implementation for AdaReasoner

├── adatrain.py              # Train AdaReasoner from scratch

├── adatest.py               # Run inference using pretrained policy (gpt4oadapt.pkl)

├── baseline_evaluation.py   # Compare with baseline strategies

├── statistic.py             # Compute and visualize statistics

├── gpt4oadapt.pkl           # Pretrained AdaReasoner weights for GPT-4o

├── requirements.txt         # Python dependencies (auto-generated)

└── README.md                # This file

## Quick Start

### Setup Environment

Install dependencies (see notes below for minimal dependency extraction):

```bash
pip install -r requirements.txt
```

### Run Demo Inference

```python
python adatest.py
```

* This will load the pretrained model from `gpt4oadapt.pkl`.
* Evaluation is performed on the test sets included in `dataset/source.json`, which contains a mix of MMLU (Math), LogiQA, Metaphor and TruthfulQA

### Train Your Own Adapter

To train AdaReasoner on a new or existing dataset:

```python
python adatrain.py
```

* Reward model: by default, a DeBERTa-based reward model is used (see paper Appendix for details).
* Configuration space: consists of generation temperature, reasoning steps, and instruction formats.

## Benchmarks

AdaReasoner has been evaluated across **six LLMs** and multiple benchmark datasets. It consistently outperforms existing baselines in both in-domain and out-of-distribution reasoning tasks.

For example, using GPT-4o:

| Method                | Metaphor        | TruthfulQA      | MMLU (Math)     | LogiQA          | Average         |
| --------------------- | --------------- | --------------- | --------------- | --------------- | --------------- |
| CoT                   | 50.40           | 78.40           | 76.04           | 70.00           | 68.71           |
| Auto-CoT              | 62.33           | 83.09           | 72.15           | 71.71           | 72.32           |
| **AdaReasoner** | **71.56** | **81.30** | **86.49** | **82.31** | **80.42** |

## Citation

If you find this work helpful, please consider citing:

```
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
