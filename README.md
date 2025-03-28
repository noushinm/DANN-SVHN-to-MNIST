# Domain Adaptation with PyTorch

This project implements Domain Adversarial Neural Network (DANN) for domain adaptation between SVHN and MNIST datasets.

© 2024 Noushin Mirnezami. All Rights Reserved.

IMPORTANT: This code is provided for educational purposes only. No commercial use, reproduction, or distribution is permitted without explicit permission from the author. Unauthorized copying or plagiarism of this project is strictly prohibited.

## Features
- Feature extraction network
- Label classifier
- Domain classifier with gradient reversal
- Domain adaptation training


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


## Requirements
- PyTorch
- torchvision
- numpy
- scikit-learn
- matplotlib

## Installation
```bash
pip install -r requirements.txt
```

## Usage
1. Train source-only model:
```bash
python train.py --mode source_only
```

2. Train DANN model:
```bash
python train.py --mode dann
```

3. Evaluate models:
```bash
python evaluate.py
```

## Architecture
- Feature Extractor: CNN with 6 conv layers
- Label Classifier: 2-layer MLP
- Domain Classifier: 3-layer MLP with gradient reversal
