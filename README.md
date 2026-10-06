# Planar Data Classification with One Hidden Layer

**CS4085 Deep Learning — Lab 2**
Effat University · College of Engineering · Computer Science
**Instructor:** Dr. Naila Marir
**Authors:** Danya Alshurafa & Sedra Massalha

---

## Overview

This lab builds a 2-class neural network with one hidden layer **from scratch in NumPy** and compares it to logistic regression on a non-linearly separable "flower" dataset. Every part of the training loop (initialization, forward propagation, cross-entropy cost, backpropagation, and gradient descent) is implemented by hand, with no deep learning frameworks.

## Model

| Layer  | Size | Activation |
|--------|------|------------|
| Input  | n_x = 2 (coordinates x₁, x₂) | — |
| Hidden | n_h = 4 (tuned to 5) | tanh |
| Output | n_y = 1 | sigmoid (threshold 0.5) |

Weights are initialized with small random values to break symmetry, and biases start at zero.

## Results

**Flower dataset (400 points)**

| Model | Accuracy |
|-------|----------|
| Logistic regression | 47% |
| Neural network (n_h = 4, lr = 1.2) | 90% |

**Effect of hidden layer size** (5,000 iterations)

| n_h | 1 | 2 | 3 | 4 | 5 | 20 | 50 |
|-----|---|---|---|---|---|----|----|
| Accuracy | 67.50% | 67.25% | 90.75% | 90.50% | **91.25%** | 91.00% | 90.50% |

n_h = 5 gave the best balance; larger networks began fitting noise in the training data.

**Effect of learning rate** (n_h = 4, 3,000 iterations)

| Learning rate | 0.01 | 0.1 | 1.2 | 10.0 |
|---------------|------|-----|-----|------|
| Accuracy | 57% | 64% | 90% | 90% (unstable cost) |

**Generalization to another dataset (noisy moons, n_h = 5)**

| Model | Accuracy |
|-------|----------|
| Logistic regression | 85% |
| Neural network | 99% |

## Key takeaways

- A non-linear activation (tanh) in the hidden layer is what lets the network bend its decision boundary around curved data that a linear model cannot separate.
- More hidden units help up to a point; beyond that, the model starts memorizing noise (overfitting).
- The learning rate controls convergence: too small barely learns, too large overshoots and oscillates.

## Repository contents

| File | Description |
|------|-------------|
| `Lab2_Mastering_Non_Linearity_DL.ipynb` | Full lab notebook with code, outputs, plots, and reflection answers |
| `Mastering_Non-Linear_Neural_Networks.pdf` | Presentation summarizing the lab and results |

## How to run

Open the notebook in [Google Colab](https://colab.research.google.com/) (File → Upload notebook) or Jupyter, then run all cells in order. It only needs `numpy`, `matplotlib`, and `scikit-learn`, which come preinstalled in Colab. All dataset loaders and helper functions are included in the notebook's utilities cell.

## Reference

[CS231n: Neural Networks Case Study](https://cs231n.github.io/neural-networks-case-study/)
