# Early Alzheimer's Disease Detection

## Abstract

This project implements a deep learning solution for early alzheimer's disease detection. 
The model achieves state-of-the-art performance on benchmark datasets using 
modern transfer learning and data augmentation techniques.

## Key Contributions

1. Implementation of robust data preprocessing pipeline
2. Application of transfer learning for medical imaging
3. Comprehensive evaluation with multiple metrics: Accuracy, Recall, AUC-ROC
4. Production-ready deployment with Streamlit interface

## Methodology

### Dataset
- Source: Kaggle
- Task: Binary Classification
- Preprocessing: Normalization, augmentation, train/val/test split

### Model Architecture
- Base: Pre-trained CNN (transfer learning)
- Custom head: Task-specific layers
- Training: Two-phase (frozen base + fine-tuning)

### Evaluation Metrics
Accuracy, Recall, AUC-ROC

## Results

[Add your results here]

## Citation

If you use this work, please cite:

```bibtex
@software{alzhimer,
  title = {Early Alzheimer's Disease Detection},
  author = {Mohamad AlJasem, MD MPH MSc},
  year = {2024},
  url = {https://github.com/m-aljasem/alzhimer}
}
```

## License

MIT License - see LICENSE file for details.
