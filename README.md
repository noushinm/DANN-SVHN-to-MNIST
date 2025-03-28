# Domain-Adversarial Neural Networks (DANN): SVHN to MNIST Adaptation

This project implements Domain-Adversarial Training of Neural Networks (DANN) for unsupervised domain adaptation between SVHN (source domain) and MNIST (target domain) datasets.

## Folder Structure
```
DANN-SVHN-to-MNIST/
├── data/
│   └── __init__.py 
│   └── digits.py
├── models/
│   ├── __init__.py
│   ├── feature_extractor.py
│   ├── label_classifier.py
│   └── domain_classifier.py
├── utils/
│   ├── __init__.py
│   └── helpers.py
├── train.py
├── evaluate.py
├── requirements.txt
└── README.md
```

## Project Overview

The goal is to adapt a model trained on SVHN (Street View House Numbers) to perform well on MNIST (handwritten digits) without using MNIST labels during training. This is achieved through adversarial learning that aligns the source and target distributions.

## Network Architecture

### Feature Extractor
- Input: Batch_Size × 3 × 32 × 32 (or 28 × 28)
- Architecture:
  ```
  Conv(32) → Conv(32) → MaxPool(2) → BatchNorm2d(32) 
  → Conv(64) → Conv(64) → MaxPool(2) → BatchNorm2d(64) 
  → Conv(128) → Conv(128) → AdaptiveAvgPool2d(1) 
  → Linear(128) → BatchNorm1d(128)
  ```

### Label Classifier
- Input: Batch_Size × 128
- Architecture:
  ```
  Linear(64) → BatchNorm1d(64) → Linear(10) → SoftmaxCrossEntropyLoss
  ```

### Domain Classifier
- Input: Batch_Size × 128
- Architecture:
  ```
  Linear(64) → BatchNorm1d(64) → Linear(64) → BatchNorm1d(64) 
  → Linear(1) → SigmoidBinaryCrossEntropyLoss
  ```

## Technical Requirements

### Environment
- Python 3.8+
- CUDA-capable GPU (recommended)

### Dependencies
- PyTorch
- torchvision
- numpy
- matplotlib
- scikit-learn

## Installation

1. Clone the repository:
```bash
git clone https://github.com/noushinm/DANN-SVHN-to-MNIST.git
cd DANN-SVHN-to-MNIST
```

2. Install requirements:
```bash
pip install -r requirements.txt
```

## Usage

1. Train source-only model (without domain adaptation):
```bash
python train.py --mode source_only --epochs 5 --batch_size 64 --learning_rate 0.01
```

2. Train DANN model (with domain adaptation):
```bash
python train.py --mode dann --epochs 5 --batch_size 64 --learning_rate 0.01
```

3. Evaluate models:
```bash
python evaluate.py --model_type source_only --checkpoint_path checkpoint_source_only.pth
# or
python evaluate.py --model_type dann --checkpoint_path checkpoint_dann.pth
```

## Implementation Details

### Domain Adaptation Strategy
- Uses gradient reversal layer for adversarial training
- Implements progressive lambda scheduling:
  λ = 2/(1 + exp(-γp)) - 1
  where γ = 10 and p is the training progress

### Training Process
1. Source-only training:
   - Trains feature extractor and label classifier on SVHN
   - Establishes baseline performance

2. DANN training:
   - Simultaneously trains all components
   - Domain classifier learns to distinguish domains
   - Feature extractor learns domain-invariant features
   - Label classifier learns to classify digits

## Expected Results
- Source-only model: Lower performance on target domain
- DANN model: Expected improvement of at least 10 points on target domain
- Target test accuracy should reach approximately 71%

## References
- [Domain-Adversarial Training of Neural Networks (Ganin et al.)](https://arxiv.org/abs/1505.07818)

---

© 2024 Noushin Mirnezami. All Rights Reserved.

IMPORTANT: This code is provided for educational purposes only. No commercial use, reproduction, or distribution is permitted without explicit permission from the author. Unauthorized copying or plagiarism of this project is strictly prohibited.
