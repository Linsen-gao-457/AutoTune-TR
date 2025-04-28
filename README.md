# AutoTR

Trust Region Optimization vs SGD: An Empirical Evaluation with Automatic Hyperparameter Tuning

#### Table of Contents

- [AutoTR](#autotr) - [Table of Contents](#table-of-contents)
  - [📖Overview](#overview)
  - [📁Install](#install)
  - [🔍 Reproducible Research](#-reproducible-research)
  - [⚙️Code Extention](#️code-extention)
  - [📖Summary](#summary)

## 📖Overview

In machine learning, first-order methods like SGD are widely used but suffer from instability sensitivity to hyperparameteres, and difficulty escaping from saddle points.

Second-order methods, like Trust Region(TR) method, leverage the curvature information(Hessian) to address these challenges, at the cost of increased computational complexity.

This project includes:

- Reproduction of key experiments from the paper ["Second-order Optimization for Non-convex Machine Learning: An Empirical Study"](https://epubs.siam.org/doi/10.1137/1.9781611976236.23)
- A new extension: automatic hyperparameter tuning comparison between SGD and TR using Ray Tune.

## 📁Install

This project is based on Python and Jupyer Notebook.
You can install the required packages with:

```
pip install torch ray[tune] numpy matplotlib
```

1. Clone this repository:

```
git clone https://github.com/Linsen-gao-457/AutoTune-TR.git
cd AutoTune-TR
```

2. Open the notebooks:
   jupyter notebook

3. Run the cells step-by-step to reproduce the experiments

## 🔍 Reproducible Research

We reproduce experiments compareing SGD and TR on one-hidden-layer MLP trained on the CIFAR-10 dataset.

Key findings:

- Generalization: TR achieves better or comparable test accuracy.
- Robustness: TR is more stable across different hyperparameters.
- Saddle Point Escape: TR avoids saddle points much more effectively than SGD.
- Efficiency: SGD is faster per iteration but less stable.

To reduce training time while preserving the ability to compare model performance, we randomly selected a subset of 5000 training images from the original training set.

Figures: Training loss and test error comparisons from random initialization and saddle-point initialization.

## ⚙️Code Extention

We extend the experiments by introduce automatic hypoerparameter tuning using Ray Tune.

Model setup:

- We initialize two hidden layers feedforward neural network(NLP) trained on CIFAR-10
- Tunable hyperparameters:
  - Layer sizes: {1, 2, 4, 8, 16, 32, 64, 128, 256}
  - Learning rate: log-uniform {1e-3, 1e-1}
  - Batch size: {2, 4, 8, 16}
- Optimizers Compared
  - SGD
  - TR

Result compare:
| Optimizer | Avg Validation Accuracy | Avg Training Time per Trial |
|:---------:|:------------------------:|:---------------------------:|
| TR | 0.26 | ~2140 seconds |
| SGD | 0.24 | ~334 seconds |

## 📖Summary

Our experiments show a clear trade-off between Trust Region (TR) and Stochastic Gradient Descent (SGD):

- **Trust Region (TR)** achieves higher stability, better saddle-point avoidance, and more robust performance across different hyperparameters. It is well-suited for tasks where solution quality and robustness are prioritized. However, this comes at the cost of significantly higher computational time.

- **Stochastic Gradient Descent (SGD)** offers faster training and lower computational cost, but its performance varies widely depending on hyperparameter settings and initialization. It is better suited for scenarios with limited resources where speed is a higher priority.

Overall, Trust Region is recommended for applications where stability and convergence reliability are critical, while SGD remains a strong choice for quick experiments or resource-constrained settings.

## 📚 References

[1] P. Xu, F. Roosta, and M. W. Mahoney,  
_Second-order Optimization for Non-convex Machine Learning: An Empirical Study_,  
Proceedings of the 2020 SIAM International Conference on Data Mining, pp. 199–207, 2020.  
[🔗 Paper Link](https://epubs.siam.org/doi/10.1137/1.9781611976236.23)

[2] A. Krizhevsky and G. Hinton,  
_Learning Multiple Layers of Features from Tiny Images_,  
Technical Report, University of Toronto, 2009.  
[🔗 Paper Link](https://www.cs.toronto.edu/~kriz/learning-features-2009-TR.pdf)

[3] R. Liaw, E. Liang, R. Nishihara, P. Moritz, J. E. Gonzalez, and I. Stoica,  
_Tune: A Research Platform for Distributed Model Selection and Training_,  
Proceedings of the 2018 Workshop on Systems for ML and Open Source Software at NeurIPS, 2018.  
[🔗 Paper Link](https://arxiv.org/abs/1807.05118)

[4] N. Agarwal, B. Bullins, and E. Hazan,  
_Second-order Optimization for Non-convex Machine Learning: An Empirical Study_,  
Proceedings of the 34th International Conference on Machine Learning (ICML), 2017.  
[🔗 Paper Link](https://arxiv.org/abs/1611.04970)
