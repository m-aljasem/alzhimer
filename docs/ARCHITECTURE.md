# Model Architecture

## Overview

Early Alzheimer's Disease Detection uses a deep learning approach for binary classification.

## Architecture Diagram

```
Input Layer
    ↓
[Preprocessing]
    ↓
Base Model (Pre-trained)
    ↓
[Feature Extraction]
    ↓
Custom Head
    ↓
Output Layer (Binary Classification)
```

## Layer Details

### Base Model
- Architecture: Pre-trained CNN
- Weights: ImageNet
- Trainable: Initially frozen, then fine-tuned

### Custom Head
- Global Average Pooling
- Dense layers with dropout
- Batch normalization
- Final classification/regression layer

## Hyperparameters

- Learning Rate: 1e-4 (initial), 1e-5 (fine-tuning)
- Batch Size: 32
- Epochs: 50
- Optimizer: Adam
- Loss Function: Task-specific

## Data Flow

1. Input preprocessing (normalization, augmentation)
2. Feature extraction via base model
3. Feature aggregation (pooling)
4. Classification/regression via custom head
5. Output prediction
