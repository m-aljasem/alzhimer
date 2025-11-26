"""
Machine Learning Models Module

This module contains implementations of various ML models for
Alzheimer's disease detection, including hyperparameter tuning.
"""

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.model_selection import cross_val_score
from sklearn.metrics import accuracy_score, recall_score, roc_curve, auc
from typing import Dict, Tuple, Optional, Any


class ModelTrainer:
    """
    Trains and evaluates multiple ML models for Alzheimer's detection.
    
    Supports:
    - Logistic Regression
    - Support Vector Machine (SVM)
    - Decision Tree
    - Random Forest
    - AdaBoost
    """
    
    def __init__(self, kfolds: int = 5):
        """
        Initialize the ModelTrainer.
        
        Args:
            kfolds: Number of folds for cross-validation
        """
        self.kfolds = kfolds
        self.models = {}
        self.results = []
    
    def train_logistic_regression(
        self,
        X_trainval: np.ndarray,
        Y_trainval: np.ndarray,
        X_test: np.ndarray,
        Y_test: np.ndarray,
        C_values: list = [0.001, 0.1, 1, 10, 100]
    ) -> Dict[str, Any]:
        """
        Train Logistic Regression model with hyperparameter tuning.
        
        Args:
            X_trainval: Training/validation features
            Y_trainval: Training/validation labels
            X_test: Test features
            Y_test: Test labels
            C_values: List of C (regularization) values to try
            
        Returns:
            Dictionary containing model, metrics, and predictions
        """
        best_score = 0
        best_parameters = None
        
        # Cross-validation to find best C
        for c in C_values:
            log_reg_model = LogisticRegression(C=c, max_iter=1000)
            scores = cross_val_score(
                log_reg_model, X_trainval, Y_trainval,
                cv=self.kfolds, scoring='accuracy'
            )
            score = np.mean(scores)
            
            if score > best_score:
                best_score = score
                best_parameters = c
        
        # Train final model with best parameters
        model = LogisticRegression(C=best_parameters, max_iter=1000)
        model.fit(X_trainval, Y_trainval)
        
        # Evaluate on test set
        test_score = model.score(X_test, Y_test)
        predicted = model.predict(X_test)
        test_recall = recall_score(Y_test, predicted, pos_label=1)
        fpr, tpr, thresholds = roc_curve(Y_test, predicted, pos_label=1)
        test_auc = auc(fpr, tpr)
        
        result = {
            'model_name': 'Logistic Regression',
            'model': model,
            'best_parameters': best_parameters,
            'best_cv_score': best_score,
            'test_accuracy': test_score,
            'test_recall': test_recall,
            'test_auc': test_auc,
            'fpr': fpr,
            'tpr': tpr,
            'thresholds': thresholds,
            'predictions': predicted
        }
        
        self.models['logistic_regression'] = model
        self.results.append(result)
        
        return result
    
    def train_svm(
        self,
        X_trainval: np.ndarray,
        Y_trainval: np.ndarray,
        X_test: np.ndarray,
        Y_test: np.ndarray,
        C_values: list = [0.001, 0.01, 0.1, 1, 10, 100, 1000],
        gamma_values: list = [0.001, 0.01, 0.1, 1, 10, 100, 1000],
        kernels: list = ['rbf', 'linear', 'poly', 'sigmoid']
    ) -> Dict[str, Any]:
        """
        Train SVM model with hyperparameter tuning.
        
        Args:
            X_trainval: Training/validation features
            Y_trainval: Training/validation labels
            X_test: Test features
            Y_test: Test labels
            C_values: List of C values to try
            gamma_values: List of gamma values to try
            kernels: List of kernel types to try
            
        Returns:
            Dictionary containing model, metrics, and predictions
        """
        best_score = 0
        best_parameters = {}
        
        # Grid search with cross-validation
        for c_param in C_values:
            for gamma_param in gamma_values:
                for k_param in kernels:
                    svm_model = SVC(kernel=k_param, C=c_param, gamma=gamma_param)
                    scores = cross_val_score(
                        svm_model, X_trainval, Y_trainval,
                        cv=self.kfolds, scoring='accuracy'
                    )
                    score = np.mean(scores)
                    
                    if score > best_score:
                        best_score = score
                        best_parameters = {
                            'C': c_param,
                            'gamma': gamma_param,
                            'kernel': k_param
                        }
        
        # Train final model
        model = SVC(
            C=best_parameters['C'],
            gamma=best_parameters['gamma'],
            kernel=best_parameters['kernel']
        )
        model.fit(X_trainval, Y_trainval)
        
        # Evaluate
        test_score = model.score(X_test, Y_test)
        predicted = model.predict(X_test)
        test_recall = recall_score(Y_test, predicted, pos_label=1)
        fpr, tpr, thresholds = roc_curve(Y_test, predicted, pos_label=1)
        test_auc = auc(fpr, tpr)
        
        result = {
            'model_name': 'SVM',
            'model': model,
            'best_parameters': best_parameters,
            'best_cv_score': best_score,
            'test_accuracy': test_score,
            'test_recall': test_recall,
            'test_auc': test_auc,
            'fpr': fpr,
            'tpr': tpr,
            'thresholds': thresholds,
            'predictions': predicted
        }
        
        self.models['svm'] = model
        self.results.append(result)
        
        return result
    
    def train_decision_tree(
        self,
        X_trainval: np.ndarray,
        Y_trainval: np.ndarray,
        X_test: np.ndarray,
        Y_test: np.ndarray,
        max_depth_range: range = range(1, 9)
    ) -> Dict[str, Any]:
        """
        Train Decision Tree model with hyperparameter tuning.
        
        Args:
            X_trainval: Training/validation features
            Y_trainval: Training/validation labels
            X_test: Test features
            Y_test: Test labels
            max_depth_range: Range of max_depth values to try
            
        Returns:
            Dictionary containing model, metrics, and predictions
        """
        best_score = 0
        best_parameter = None
        
        # Cross-validation to find best max_depth
        for md in max_depth_range:
            tree_model = DecisionTreeClassifier(
                random_state=0, max_depth=md, criterion='gini'
            )
            scores = cross_val_score(
                tree_model, X_trainval, Y_trainval,
                cv=self.kfolds, scoring='accuracy'
            )
            score = np.mean(scores)
            
            if score > best_score:
                best_score = score
                best_parameter = md
        
        # Train final model
        model = DecisionTreeClassifier(max_depth=best_parameter, random_state=0)
        model.fit(X_trainval, Y_trainval)
        
        # Evaluate
        test_score = model.score(X_test, Y_test)
        predicted = model.predict(X_test)
        test_recall = recall_score(Y_test, predicted, pos_label=1)
        fpr, tpr, thresholds = roc_curve(Y_test, predicted, pos_label=1)
        test_auc = auc(fpr, tpr)
        
        result = {
            'model_name': 'Decision Tree',
            'model': model,
            'best_parameters': {'max_depth': best_parameter},
            'best_cv_score': best_score,
            'test_accuracy': test_score,
            'test_recall': test_recall,
            'test_auc': test_auc,
            'fpr': fpr,
            'tpr': tpr,
            'thresholds': thresholds,
            'predictions': predicted,
            'feature_importances': model.feature_importances_
        }
        
        self.models['decision_tree'] = model
        self.results.append(result)
        
        return result
    
    def train_random_forest(
        self,
        X_trainval: np.ndarray,
        Y_trainval: np.ndarray,
        X_test: np.ndarray,
        Y_test: np.ndarray,
        n_estimators_range: range = range(2, 15, 2),
        max_features_range: range = range(1, 9),
        max_depth_range: range = range(1, 9)
    ) -> Dict[str, Any]:
        """
        Train Random Forest model with hyperparameter tuning.
        
        Args:
            X_trainval: Training/validation features
            Y_trainval: Training/validation labels
            X_test: Test features
            Y_test: Test labels
            n_estimators_range: Range of n_estimators values
            max_features_range: Range of max_features values
            max_depth_range: Range of max_depth values
            
        Returns:
            Dictionary containing model, metrics, and predictions
        """
        best_score = 0
        best_parameters = {}
        
        # Grid search
        for M in n_estimators_range:
            for d in max_features_range:
                for m in max_depth_range:
                    rf_model = RandomForestClassifier(
                        n_estimators=M,
                        max_features=d,
                        max_depth=m,
                        random_state=0,
                        n_jobs=-1
                    )
                    scores = cross_val_score(
                        rf_model, X_trainval, Y_trainval,
                        cv=self.kfolds, scoring='accuracy'
                    )
                    score = np.mean(scores)
                    
                    if score > best_score:
                        best_score = score
                        best_parameters = {
                            'n_estimators': M,
                            'max_features': d,
                            'max_depth': m
                        }
        
        # Train final model
        model = RandomForestClassifier(
            n_estimators=best_parameters['n_estimators'],
            max_features=best_parameters['max_features'],
            max_depth=best_parameters['max_depth'],
            random_state=0,
            n_jobs=-1
        )
        model.fit(X_trainval, Y_trainval)
        
        # Evaluate
        test_score = model.score(X_test, Y_test)
        predicted = model.predict(X_test)
        test_recall = recall_score(Y_test, predicted, pos_label=1)
        fpr, tpr, thresholds = roc_curve(Y_test, predicted, pos_label=1)
        test_auc = auc(fpr, tpr)
        
        result = {
            'model_name': 'Random Forest',
            'model': model,
            'best_parameters': best_parameters,
            'best_cv_score': best_score,
            'test_accuracy': test_score,
            'test_recall': test_recall,
            'test_auc': test_auc,
            'fpr': fpr,
            'tpr': tpr,
            'thresholds': thresholds,
            'predictions': predicted,
            'feature_importances': model.feature_importances_
        }
        
        self.models['random_forest'] = model
        self.results.append(result)
        
        return result
    
    def train_adaboost(
        self,
        X_trainval: np.ndarray,
        Y_trainval: np.ndarray,
        X_test: np.ndarray,
        Y_test: np.ndarray,
        n_estimators_range: range = range(2, 15, 2),
        learning_rates: list = [0.0001, 0.001, 0.01, 0.1, 1]
    ) -> Dict[str, Any]:
        """
        Train AdaBoost model with hyperparameter tuning.
        
        Args:
            X_trainval: Training/validation features
            Y_trainval: Training/validation labels
            X_test: Test features
            Y_test: Test labels
            n_estimators_range: Range of n_estimators values
            learning_rates: List of learning rate values to try
            
        Returns:
            Dictionary containing model, metrics, and predictions
        """
        best_score = 0
        best_parameters = {}
        
        # Grid search
        for M in n_estimators_range:
            for lr in learning_rates:
                boost_model = AdaBoostClassifier(
                    n_estimators=M,
                    learning_rate=lr,
                    random_state=0
                )
                scores = cross_val_score(
                    boost_model, X_trainval, Y_trainval,
                    cv=self.kfolds, scoring='accuracy'
                )
                score = np.mean(scores)
                
                if score > best_score:
                    best_score = score
                    best_parameters = {
                        'n_estimators': M,
                        'learning_rate': lr
                    }
        
        # Train final model
        model = AdaBoostClassifier(
            n_estimators=best_parameters['n_estimators'],
            learning_rate=best_parameters['learning_rate'],
            random_state=0
        )
        model.fit(X_trainval, Y_trainval)
        
        # Evaluate
        test_score = model.score(X_test, Y_test)
        predicted = model.predict(X_test)
        test_recall = recall_score(Y_test, predicted, pos_label=1)
        fpr, tpr, thresholds = roc_curve(Y_test, predicted, pos_label=1)
        test_auc = auc(fpr, tpr)
        
        result = {
            'model_name': 'AdaBoost',
            'model': model,
            'best_parameters': best_parameters,
            'best_cv_score': best_score,
            'test_accuracy': test_score,
            'test_recall': test_recall,
            'test_auc': test_auc,
            'fpr': fpr,
            'tpr': tpr,
            'thresholds': thresholds,
            'predictions': predicted,
            'feature_importances': model.feature_importances_
        }
        
        self.models['adaboost'] = model
        self.results.append(result)
        
        return result
    
    def get_results_summary(self) -> pd.DataFrame:
        """
        Get a summary of all model results.
        
        Returns:
            DataFrame with model performance metrics
        """
        summary_data = []
        for result in self.results:
            summary_data.append({
                'Model': result['model_name'],
                'Accuracy': result['test_accuracy'],
                'Recall': result['test_recall'],
                'AUC': result['test_auc']
            })
        
        return pd.DataFrame(summary_data)

