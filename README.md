# MAML-GAN: Few-Shot Distribution Learning using Meta-Learning

This repository contains the implementation and accompanying report for a **Model-Agnostic Meta-Learning (MAML)** framework applied to **Generative Adversarial Networks (GANs)**. The project demonstrates how meta-learning enables generative models to rapidly adapt to unseen probability distributions using only a few gradient updates.

The implementation was developed as part of coursework exploring **Meta-Learning in Neural Networks**, with a focus on combining MAML and GANs for few-shot generative modeling.

---

## Overview

Traditional GANs require large amounts of data and extensive training to learn new data distributions. This project applies **Model-Agnostic Meta-Learning (MAML)** to train a GAN that learns a strong initialization capable of quickly adapting to new Gaussian distributions with only a small number of examples.

During meta-training, the model is exposed to multiple Gaussian distributions (tasks). Rather than optimizing for a single distribution, it learns parameters that can rapidly adapt to previously unseen tasks after only a few gradient updates.

---

## Problem Statement

Generative models often struggle in low-data scenarios due to:

- Limited training samples
- Slow convergence
- Poor generalization to unseen distributions
- Overfitting on small datasets

The objective of this project is to investigate whether **Model-Agnostic Meta-Learning (MAML)** can improve the adaptability of GANs by learning an initialization that enables rapid few-shot learning on unseen probability distributions.

---

## Features

- Implementation of Model-Agnostic Meta-Learning (MAML)
- Conditional GAN architecture
- Meta-training across multiple Gaussian distributions
- Inner-loop and outer-loop optimization
- Wasserstein GAN (WGAN) loss formulation
- Few-shot adaptation to unseen tasks
- Visualization of generated versus real distributions
- Experimental analysis and observations

---

## Repository Structure

```
Knowledge-Distillation-3-Alternating-Functions/
│
├── MAML_Algorithm_using_GANs.ipynb
├── Report.pdf
└── README.md
```

---

## Technologies Used

- Python
- PyTorch
- NumPy
- Matplotlib
- Jupyter Notebook

---

## Methodology

The project follows the standard MAML training procedure:

1. Sample multiple Gaussian distributions as independent tasks.
2. Train a conditional Generator and Discriminator.
3. Perform task-specific adaptation using the inner optimization loop.
4. Aggregate losses across tasks.
5. Update the shared model parameters using the outer optimization loop.
6. Evaluate the meta-trained model on previously unseen Gaussian distributions.

The implementation uses Wasserstein GAN losses for improved training stability while enabling rapid adaptation through meta-learning.

---

## Results

The experiments demonstrate that:

- The meta-trained generator quickly adapts to unseen Gaussian distributions.
- Few gradient updates are sufficient for learning new tasks.
- Generated samples closely approximate target distributions.
- Meta-learning significantly improves sample efficiency compared to conventional training.

---

## Running the Project

### 1. Clone the repository

```bash
git clone https://github.com/aadyatalreja/Knowledge-Distillation-3-Alternating-Functions.git

cd Knowledge-Distillation-3-Alternating-Functions
```

### 2. Install dependencies

```bash
pip install torch numpy matplotlib notebook
```

### 3. Launch Jupyter Notebook

```bash
jupyter notebook
```

Open:

```
MAML_Algorithm_using_GANs.ipynb
```

and execute all cells sequentially.

---

## Report

The repository includes a detailed project report covering:

- Meta-Learning fundamentals
- Model-Agnostic Meta-Learning (MAML)
- GAN architecture
- Conditional GAN implementation
- Training methodology
- Experimental setup
- Results and analysis
- Applications and future work

---

## Learning Outcomes

This project demonstrates practical understanding of:

- Meta-Learning
- Model-Agnostic Meta-Learning (MAML)
- Few-shot learning
- Generative Adversarial Networks (GANs)
- Conditional GANs
- Wasserstein GAN training
- PyTorch implementation
- Deep learning research

---

## License

This repository is intended for educational and research purposes.
