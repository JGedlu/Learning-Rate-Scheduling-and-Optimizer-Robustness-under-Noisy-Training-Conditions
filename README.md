# Robust Optimization Strategies for CNNs Under Label Noise  
**Advanced Machine Learning – Final Project**  
**Authors:** Daevon Wing & Jason Gedlu  

---

## Overview

This project investigates how different optimization strategies influence the robustness of Convolutional Neural Networks (CNNs) under *symmetric label noise*. Using the Fashion-MNIST dataset, we evaluate six optimizers and learning-rate schedules across three noise levels (0%, 10%, 20%) to determine which configurations generalize best in the presence of corrupted labels.

The study aligns with course topics such as optimization algorithms, robustness, loss landscapes, and deep learning generalization.

---

## Project Objectives

- Evaluate how label noise affects model performance.
- Compare SGD and Adam (with and without learning-rate schedules).
- Assess the robustness of each optimizer across noise levels.
- Visualize performance trends and noise-sensitivity patterns.
- Produce reproducible experimental outputs and documentation.

---

## Dataset

**Fashion-MNIST**

- 60,000 training images  
- 10,000 test images  
- 28×28 grayscale  
- 10 clothing categories  

### Label Noise

Noise injection uses **symmetric noise**, replacing a percentage of labels randomly with other classes. Noise levels used:

- **0% (clean)**
- **10%**
- **20%**

This simulates mislabeled real-world datasets common in production ML systems.

---

## Model Architecture

A compact CNN suitable for noisy-data experiments:

- Conv2D (32 filters, 3×3) + ReLU  
- MaxPooling (2×2)  
- Conv2D (64 filters, 3×3) + ReLU  
- MaxPooling (2×2)  
- Flatten  
- Dense (128, ReLU)  
- Dense (10, softmax)  

Optimizer, learning rate, and LR schedule are passed dynamically for each experiment.

---

## Optimizers Evaluated

| Optimizer | LR Strategy |
|----------|-------------|
| SGD | Constant LR |
| Adam | Constant LR |
| SGD | Step Decay |
| Adam | Step Decay |
| SGD | Cosine Annealing |
| Adam | Cosine Annealing |

---

## Estimated Final Accuracy Results

| Optimizer       | 0% Noise | 10% Noise | 20% Noise |
|-----------------|----------|-----------|-----------|
| SGD_Constant    | 0.8595   | 0.8554    | 0.8504    |
| Adam_Constant   | 0.9069   | 0.9016    | 0.8867    |
| SGD_StepDecay   | 0.8210   | 0.8018    | 0.7973    |
| Adam_StepDecay  | 0.9060   | 0.9010    | 0.9007    |
| SGD_Cosine      | 0.8430   | 0.8430    | 0.8344    |
| **Adam_Cosine** | **0.9157** | **0.9065** | **0.9011** |

### Key Observations

- Adam consistently outperforms SGD across all noise levels.  
- Cosine Annealing improves robustness for both optimizers.  
- Step Decay performs the worst, especially with SGD.  
- Increasing noise reduces accuracy, but adaptive optimizers degrade more gracefully.

---

## Visualizations

All plots are generated automatically in `Results/`:

- **Accuracy vs Noise** (all optimizers)
- **Bar charts** for 0%, 10%, and 20% noise  
- **Robustness drop (0% → 20%)**
- **Optimizer × Noise Heatmap**
- **Training curves for clean data**

These graphs are intended for inclusion in the final presentation.

---

## How to Run

Clone the repository:

```bash
  git clone <your-repository-url>
  cd <your-repository>
```
Install dependencies
```bash
  pip install -r requirements.txt
```
Run the training pipeline
```bash
  python Main.py
```
**Results will be stored in the Results/ directory**