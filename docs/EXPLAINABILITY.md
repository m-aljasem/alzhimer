# Model Explainability Guide

## Overview

This project includes comprehensive explainability features using SHAP (SHapley Additive exPlanations) and LIME (Local Interpretable Model-agnostic Explanations) to provide interpretable insights into model predictions.

## Why Explainability Matters in Medical AI

- **Clinical Trust**: Healthcare professionals need to understand why a model makes specific predictions
- **Regulatory Compliance**: Many medical AI regulations require model interpretability
- **Error Detection**: Understanding model decisions helps identify potential biases or errors
- **Research Insights**: Feature importance reveals which biomarkers/features are most predictive

## Available Methods

### 1. SHAP (SHapley Additive exPlanations)

SHAP provides a unified framework for explaining model outputs by assigning each feature an importance value for a particular prediction.

**For ML models:**
- Uses TreeExplainer/KernelExplainer
- Provides both local (single prediction) and global (dataset-wide) explanations
- Visualizes feature contributions to predictions

### 2. LIME (Local Interpretable Model-agnostic Explanations)

LIME explains individual predictions by approximating the model locally with an interpretable model.

**Best for:**
- Image-based models
- Understanding which image regions contribute to predictions
- Visual explanations

## Usage Examples

### Basic SHAP Explanation

```python
from src.explainability import ModelExplainer

# Initialize explainer
explainer = ModelExplainer(model, X_train_sample)

# Explain single prediction
shap_values = explainer.explain_instance(X_test[0], plot=True)

# Explain multiple predictions
shap_values = explainer.explain_dataset(X_test[:10], plot=True)

# Get feature importance
importance = explainer.get_feature_importance(X_test[:100])
```

### LIME Explanation (for images)

```python
from src.explainability import create_lime_explainer, explain_with_lime, plot_lime_explanation

# Create explainer
lime_explainer = create_lime_explainer(model)

# Explain image
explanation = explain_with_lime(lime_explainer, image, model)

# Plot explanation
plot_lime_explanation(explanation, label=predicted_class)
```

## Integration with Streamlit App

The Streamlit app includes an "Explainability" tab where you can:

1. Upload an image/input
2. Get prediction
3. View SHAP/LIME explanations
4. See feature importance plots

## Interpretation Guide

### SHAP Values

- **Positive SHAP value**: Feature increases the prediction
- **Negative SHAP value**: Feature decreases the prediction
- **Magnitude**: Larger absolute values indicate stronger influence

### Feature Importance

Features are ranked by their average absolute SHAP values across the dataset, showing which features the model relies on most.

## Best Practices

1. **Use appropriate background data**: For SHAP, use representative training samples
2. **Limit instances**: Explaining many instances can be slow; use samples
3. **Combine methods**: Use both SHAP and LIME for comprehensive understanding
4. **Validate explanations**: Compare explanations with clinical knowledge

## References

- SHAP Paper: Lundberg & Lee (2017). "A Unified Approach to Interpreting Model Predictions"
- LIME Paper: Ribeiro et al. (2016). "Why Should I Trust You?"
- Medical AI Explainability: Topol (2019). "High-performance medicine: the convergence of human and artificial intelligence"
