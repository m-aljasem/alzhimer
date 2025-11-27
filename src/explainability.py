"""
Explainability utilities for alzhimer using SHAP and other interpretability tools.

This module provides model interpretability features crucial for medical AI applications,
including SHAP values, feature importance, and visualization tools.
"""

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

# SHAP imports
try:
    import shap
    SHAP_AVAILABLE = True
except ImportError:
    SHAP_AVAILABLE = False
    print("⚠️ SHAP not installed. Install with: pip install shap")

# LIME imports (for image explanations)
try:
    import lime
    from lime import lime_image
    LIME_AVAILABLE = True
except ImportError:
    LIME_AVAILABLE = False
    print("⚠️ LIME not installed. Install with: pip install lime")


class ModelExplainer:
    """
    Explainability wrapper for scikit-learn models.
    Supports TreeExplainer for tree-based models and KernelExplainer for others.
    """
    
    def __init__(self, model, X_train, feature_names=None):
        """
        Initialize explainer.
        
        Args:
            model: Trained scikit-learn model
            X_train: Training data (for background)
            feature_names: List of feature names
        """
        if not SHAP_AVAILABLE:
            raise ImportError("SHAP is required. Install with: pip install shap")
        
        self.model = model
        self.X_train = X_train
        self.feature_names = feature_names or [f"Feature_{i}" for i in range(X_train.shape[1])]
        
        # Choose appropriate explainer
        model_type = type(model).__name__
        if any(tree in model_type for tree in ['Tree', 'Forest', 'GradientBoosting', 'XGB', 'LGBM']):
            self.explainer = shap.TreeExplainer(model)
            self.explainer_type = 'TreeExplainer'
        else:
            # Use subset of training data for KernelExplainer (faster)
            background = shap.sample(X_train, min(100, len(X_train)))
            self.explainer = shap.KernelExplainer(model.predict_proba, background)
            self.explainer_type = 'KernelExplainer'
    
    def explain_instance(self, instance, plot=True):
        """
        Explain a single prediction.
        
        Args:
            instance: Single instance to explain (1D array)
            plot: Whether to plot SHAP values
            
        Returns:
            shap_values: SHAP values
        """
        if len(instance.shape) == 1:
            instance = instance.reshape(1, -1)
        
        if self.explainer_type == 'TreeExplainer':
            shap_values = self.explainer.shap_values(instance)
            if isinstance(shap_values, list):
                shap_values = shap_values[1]  # For binary classification, use positive class
        else:
            shap_values = self.explainer.shap_values(instance)
        
        if plot:
            shap.waterfall_plot(
                shap.Explanation(
                    values=shap_values[0],
                    base_values=self.explainer.expected_value if hasattr(self.explainer, 'expected_value') else 0,
                    data=instance[0],
                    feature_names=self.feature_names
                )
            )
        
        return shap_values
    
    def explain_dataset(self, X, max_instances=100, plot=True):
        """
        Explain multiple instances.
        
        Args:
            X: Instances to explain
            max_instances: Maximum number of instances to explain
            plot: Whether to plot summary
            
        Returns:
            shap_values: SHAP values for all instances
        """
        if len(X) > max_instances:
            X = X[:max_instances]
        
        if self.explainer_type == 'TreeExplainer':
            shap_values = self.explainer.shap_values(X)
            if isinstance(shap_values, list):
                shap_values = shap_values[1]
        else:
            shap_values = self.explainer.shap_values(X)
        
        if plot:
            shap.summary_plot(shap_values, X, feature_names=self.feature_names, show=False)
            plt.tight_layout()
            plt.show()
        
        return shap_values
    
    def get_feature_importance(self, X=None):
        """
        Get global feature importance.
        
        Args:
            X: Optional data for permutation importance
            
        Returns:
            importance_dict: Dictionary of feature importances
        """
        if hasattr(self.model, 'feature_importances_'):
            importances = self.model.feature_importances_
        elif X is not None:
            # Use SHAP for permutation importance
            shap_values = self.explain_dataset(X, max_instances=100, plot=False)
            importances = np.abs(shap_values).mean(0)
        else:
            raise ValueError("Model doesn't have feature_importances_ and no data provided")
        
        importance_dict = dict(zip(self.feature_names, importances))
        return dict(sorted(importance_dict.items(), key=lambda x: x[1], reverse=True))


def create_lime_explainer(model, preprocess_fn=None):
    """
    Create LIME explainer for image models.
    
    Args:
        model: Trained model
        preprocess_fn: Optional preprocessing function
        
    Returns:
        explainer: LIME explainer
    """
    if not LIME_AVAILABLE:
        raise ImportError("LIME is required. Install with: pip install lime")
    
    explainer = lime_image.LimeImageExplainer()
    return explainer


def explain_with_lime(explainer, image, model, top_labels=5, num_features=10):
    """
    Explain image prediction using LIME.
    
    Args:
        explainer: LIME explainer
        image: Input image
        model: Trained model
        top_labels: Number of top labels to explain
        num_features: Number of features to show
        
    Returns:
        explanation: LIME explanation
    """
    if not LIME_AVAILABLE:
        raise ImportError("LIME is required. Install with: pip install lime")
    
    explanation = explainer.explain_instance(
        image.astype('double'),
        model.predict,
        top_labels=top_labels,
        hide_color=0,
        num_samples=1000
    )
    
    return explanation


def plot_lime_explanation(explanation, label=1, figsize=(10, 5)):
    """
    Plot LIME explanation.
    
    Args:
        explanation: LIME explanation object
        label: Label to explain
        figsize: Figure size
    """
    if not LIME_AVAILABLE:
        raise ImportError("LIME is required. Install with: pip install lime")
    
    temp, mask = explanation.get_image_and_mask(
        label,
        positive_only=True,
        num_features=10,
        hide_rest=True
    )
    
    fig, axes = plt.subplots(1, 2, figsize=figsize)
    axes[0].imshow(temp)
    axes[0].set_title('Original Image')
    axes[0].axis('off')
    
    axes[1].imshow(mask)
    axes[1].set_title('LIME Explanation (Important Regions)')
    axes[1].axis('off')
    
    plt.tight_layout()
    plt.show()
