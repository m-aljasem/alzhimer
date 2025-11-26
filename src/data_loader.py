"""
Data Loading and Preprocessing Module

This module handles loading MRI data, preprocessing, and feature engineering
for Alzheimer's disease detection.
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from typing import Tuple, Optional


class DataLoader:
    """
    Handles data loading and preprocessing for Alzheimer's detection.
    
    This class manages:
    - Loading CSV data
    - Handling missing values (imputation)
    - Feature encoding
    - Data splitting
    - Feature scaling
    """
    
    def __init__(self, csv_path: str):
        """
        Initialize the DataLoader.
        
        Args:
            csv_path: Path to the CSV file containing MRI data
        """
        self.csv_path = csv_path
        self.df = None
        self.scaler = MinMaxScaler()
        self.X_trainval = None
        self.X_test = None
        self.Y_trainval = None
        self.Y_test = None
        
    def load_data(self) -> pd.DataFrame:
        """
        Load the dataset from CSV file.
        
        Returns:
            DataFrame containing the loaded data
        """
        self.df = pd.read_csv(self.csv_path)
        return self.df
    
    def preprocess_data(self, use_first_visit_only: bool = True) -> pd.DataFrame:
        """
        Preprocess the data for model training.
        
        This includes:
        - Filtering to first visit only (if specified)
        - Encoding categorical variables
        - Handling missing values with imputation
        - Dropping unnecessary columns
        
        Args:
            use_first_visit_only: Whether to use only first visit data
            
        Returns:
            Preprocessed DataFrame
        """
        if self.df is None:
            self.load_data()
        
        # Use first visit data only for consistency
        if use_first_visit_only and 'Visit' in self.df.columns:
            self.df = self.df.loc[self.df['Visit'] == 1]
            self.df = self.df.reset_index(drop=True)
        
        # Encode gender: F=0, M=1
        if 'M/F' in self.df.columns:
            self.df['M/F'] = self.df['M/F'].replace(['F', 'M'], [0, 1])
        
        # Encode target variable: Converted -> Demented, then to binary
        if 'Group' in self.df.columns:
            self.df['Group'] = self.df['Group'].replace(['Converted'], ['Demented'])
            self.df['Group'] = self.df['Group'].replace(['Demented', 'Nondemented'], [1, 0])
        
        # Drop unnecessary columns
        columns_to_drop = ['MRI ID', 'Visit', 'Hand']
        existing_cols_to_drop = [col for col in columns_to_drop if col in self.df.columns]
        if existing_cols_to_drop:
            self.df = self.df.drop(existing_cols_to_drop, axis=1)
        
        # Handle missing values in SES column using median imputation grouped by EDUC
        if 'SES' in self.df.columns and self.df['SES'].isnull().sum() > 0:
            self.df["SES"].fillna(
                self.df.groupby("EDUC")["SES"].transform("median"),
                inplace=True
            )
        
        return self.df
    
    def split_data(
        self,
        test_size: float = 0.25,
        random_state: int = 0,
        features: Optional[list] = None
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        Split data into training/validation and test sets.
        
        Args:
            test_size: Proportion of data to use for testing
            random_state: Random seed for reproducibility
            features: List of feature column names. If None, uses default features.
            
        Returns:
            Tuple of (X_trainval, X_test, Y_trainval, Y_test) as numpy arrays
        """
        if self.df is None:
            raise ValueError("Data must be preprocessed first. Call preprocess_data()")
        
        # Default features if not specified
        if features is None:
            features = ['M/F', 'Age', 'EDUC', 'SES', 'MMSE', 'eTIV', 'nWBV', 'ASF']
        
        # Extract features and target
        Y = self.df['Group'].values
        X = self.df[features]
        
        # Split into train/val and test
        self.X_trainval, self.X_test, self.Y_trainval, self.Y_test = train_test_split(
            X, Y, test_size=test_size, random_state=random_state
        )
        
        return self.X_trainval, self.X_test, self.Y_trainval, self.Y_test
    
    def scale_features(
        self,
        X_trainval: Optional[np.ndarray] = None,
        X_test: Optional[np.ndarray] = None
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Scale features using MinMaxScaler.
        
        Args:
            X_trainval: Training/validation features. If None, uses stored values.
            X_test: Test features. If None, uses stored values.
            
        Returns:
            Tuple of (scaled_X_trainval, scaled_X_test)
        """
        if X_trainval is None:
            X_trainval = self.X_trainval
        if X_test is None:
            X_test = self.X_test
        
        if X_trainval is None or X_test is None:
            raise ValueError("Data must be split first. Call split_data()")
        
        # Fit scaler on training data
        self.scaler.fit(X_trainval)
        
        # Transform both sets
        X_trainval_scaled = self.scaler.transform(X_trainval)
        X_test_scaled = self.scaler.transform(X_test)
        
        return X_trainval_scaled, X_test_scaled
    
    def get_feature_names(self) -> list:
        """
        Get the list of feature names used in the model.
        
        Returns:
            List of feature column names
        """
        return ['M/F', 'Age', 'EDUC', 'SES', 'MMSE', 'eTIV', 'nWBV', 'ASF']

