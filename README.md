# 🧠 Alzheimer's Disease Detection System

A comprehensive machine learning system for detecting early Alzheimer's disease using MRI biomarker data. This project provides both a Python package for programmatic use and a Streamlit web application for interactive predictions.

![Python](https://img.shields.io/badge/python-3.8+-blue.svg)
![License](https://img.shields.io/badge/license-MIT-green.svg)
![Streamlit](https://img.shields.io/badge/streamlit-1.28+-red.svg)

## 📋 Table of Contents

- [Overview](#overview)
- [Features](#features)
- [Installation](#installation)
- [Quick Start](#quick-start)
- [Usage](#usage)
- [Project Structure](#project-structure)
- [Model Performance](#model-performance)
- [Dataset](#dataset)
- [Contributing](#contributing)
- [License](#license)
- [Disclaimer](#disclaimer)

## 🎯 Overview

This project implements multiple machine learning models to predict Alzheimer's disease (dementia) using MRI biomarker data from the OASIS (Open Access Series of Imaging Studies) dataset. The system supports five different algorithms and provides an intuitive web interface for both training and prediction.

### Key Capabilities

- **Multiple ML Models**: Logistic Regression, SVM, Decision Tree, Random Forest, and AdaBoost
- **Hyperparameter Tuning**: Automatic cross-validation for optimal parameters
- **Interactive Web Interface**: User-friendly Streamlit app for predictions
- **Comprehensive Evaluation**: Accuracy, Recall, and AUC metrics
- **Feature Engineering**: Automatic data preprocessing and scaling

## ✨ Features

### 🔬 Data Processing
- Automatic handling of missing values (median imputation)
- Feature encoding and normalization
- Train/validation/test splitting
- Cross-validation support

### 🤖 Machine Learning Models
1. **Logistic Regression** - Fast, interpretable linear model
2. **Support Vector Machine** - Non-linear classification with multiple kernels
3. **Decision Tree** - Interpretable tree-based model with feature importance
4. **Random Forest** - Robust ensemble method (typically best performer)
5. **AdaBoost** - Adaptive boosting for improved accuracy

### 🌐 Web Application
- Interactive prediction interface
- Model training interface
- Real-time performance visualization
- Model comparison and selection

## 🚀 Installation

### Prerequisites

- Python 3.8 or higher
- pip package manager

### Step 1: Clone the Repository

```bash
git clone <repository-url>
cd neurocare-ai
```

### Step 2: Create Virtual Environment (Recommended)

```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

### Step 3: Install Dependencies

```bash
pip install -r requirements.txt
```

## 🏃 Quick Start

### Option 1: Using the Web Application (Recommended)

1. **Launch the Streamlit app**:
   ```bash
   streamlit run app.py
   ```

2. **Open your browser** to the URL shown (typically `http://localhost:8501`)

3. **Navigate to "Prediction"** tab and enter patient data, or
   **Navigate to "Model Training"** to train models on your data

### Option 2: Using Python Package

```python
from src.data_loader import DataLoader
from src.models import ModelTrainer

# Load and preprocess data
loader = DataLoader("path/to/oasis_longitudinal.csv")
loader.preprocess_data()

# Split data
X_trainval, X_test, Y_trainval, Y_test = loader.split_data()

# Scale features
X_trainval_scaled, X_test_scaled = loader.scale_features()

# Train Random Forest model
trainer = ModelTrainer(kfolds=5)
result = trainer.train_random_forest(
    X_trainval_scaled, Y_trainval,
    X_test_scaled, Y_test
)

print(f"Accuracy: {result['test_accuracy']:.2%}")
print(f"AUC: {result['test_auc']:.2%}")
```

## 📖 Usage

### Web Application Usage

#### Making Predictions

1. Go to the **"Prediction"** page
2. Enter patient information:
   - Demographics (Gender, Age, Education, SES)
   - MRI biomarkers (MMSE, eTIV, nWBV, ASF)
3. Click **"Predict"** to get results
4. View prediction, probabilities, and interpretation

#### Training Models

1. Go to the **"Model Training"** page
2. Upload a CSV file with MRI data
3. Select which models to train
4. Adjust training parameters (optional)
5. Click **"Start Training"**
6. View performance metrics and save the best model

### Programmatic Usage

#### Data Loading and Preprocessing

```python
from src.data_loader import DataLoader

# Initialize loader
loader = DataLoader("data/oasis_longitudinal.csv")

# Load and preprocess
df = loader.load_data()
df_processed = loader.preprocess_data(use_first_visit_only=True)

# Split into train/test
X_trainval, X_test, Y_trainval, Y_test = loader.split_data(
    test_size=0.25,
    random_state=0
)

# Scale features
X_trainval_scaled, X_test_scaled = loader.scale_features()
```

#### Model Training

```python
from src.models import ModelTrainer

# Initialize trainer
trainer = ModelTrainer(kfolds=5)

# Train multiple models
lr_result = trainer.train_logistic_regression(
    X_trainval_scaled, Y_trainval, X_test_scaled, Y_test
)

rf_result = trainer.train_random_forest(
    X_trainval_scaled, Y_trainval, X_test_scaled, Y_test
)

# Get summary
summary = trainer.get_results_summary()
print(summary)
```

#### Making Predictions

```python
import pickle

# Load saved model
with open("models/best_model.pkl", "rb") as f:
    model = pickle.load(f)

# Prepare features
features = {
    'M/F': 1,  # 1 for Male, 0 for Female
    'Age': 75,
    'EDUC': 14,
    'SES': 2,
    'MMSE': 27,
    'eTIV': 1490,
    'nWBV': 0.73,
    'ASF': 1.20
}

# Make prediction (requires scaler)
# See app.py for complete prediction example
```

## 📁 Project Structure

```
neurocare-ai/
├── src/                      # Source code package
│   ├── __init__.py          # Package initialization
│   ├── data_loader.py       # Data loading and preprocessing
│   └── models.py            # ML model implementations
├── app.py                    # Streamlit web application
├── requirements.txt          # Python dependencies
├── README.md                # This file
├── models/                   # Saved models (created after training)
│   ├── best_model.pkl
│   └── scaler.pkl
└── data/                     # Data directory (optional)
    └── oasis_longitudinal.csv
```

## 📊 Model Performance

Based on the original notebook analysis, typical performance metrics:

| Model | Accuracy | Recall | AUC |
|-------|----------|--------|-----|
| Logistic Regression | ~78.9% | ~0.79 | ~0.79 |
| SVM | ~81.6% | ~1.00 | ~0.82 |
| Decision Tree | ~81.6% | ~0.82 | ~0.83 |
| **Random Forest** | **~84.2%** | **~0.84** | **~0.84** |
| AdaBoost | ~84.2% | ~0.82 | ~0.83 |

*Note: Results may vary based on data split and hyperparameters. Random Forest typically performs best.*

## 📦 Dataset

This project uses the **OASIS Longitudinal MRI Dataset**:

- **Source**: [OASIS Brains](http://www.oasis-brains.org)
- **Kaggle**: [MRI and Alzheimer's Dataset](https://www.kaggle.com/jboysen/mri-and-alzheimers)
- **Size**: 150 subjects, longitudinal data
- **Features**: 8 MRI biomarkers and patient characteristics

### Required Columns

- `M/F`: Gender (F/M)
- `Age`: Patient age
- `EDUC`: Years of education
- `SES`: Socioeconomic status (1-5)
- `MMSE`: Mini Mental State Examination score (0-30)
- `eTIV`: Estimated Total Intracranial Volume
- `nWBV`: Normalized Whole Brain Volume
- `ASF`: Atlas Scaling Factor
- `Group`: Target variable (Nondemented/Demented/Converted)

## 🔧 Configuration

### Model Hyperparameters

Default hyperparameter ranges can be adjusted in `src/models.py`:

- **Logistic Regression**: C values `[0.001, 0.1, 1, 10, 100]`
- **SVM**: C, gamma, and kernel options
- **Decision Tree**: max_depth range `[1-8]`
- **Random Forest**: n_estimators, max_features, max_depth
- **AdaBoost**: n_estimators and learning_rate

### Data Preprocessing

Options in `src/data_loader.py`:

- `use_first_visit_only`: Use only first visit data (default: True)
- `test_size`: Proportion for test set (default: 0.25)
- `random_state`: Random seed for reproducibility

## 🧪 Testing

To test the package functionality:

```python
# Test data loading
python -c "from src.data_loader import DataLoader; print('DataLoader OK')"

# Test model training (requires data file)
python -c "from src.models import ModelTrainer; print('ModelTrainer OK')"
```

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request. Areas for improvement:

- Additional ML models
- Enhanced feature engineering
- Better visualization
- Performance optimizations
- Documentation improvements

## 📝 License

This project is licensed under the MIT License - see the LICENSE file for details.

## ⚠️ Disclaimer

**IMPORTANT MEDICAL DISCLAIMER**

This software is provided for **research and educational purposes only**. It is **NOT** intended for:

- Clinical diagnosis
- Medical treatment decisions
- Patient care
- Any medical or healthcare use

**Always consult qualified healthcare professionals** for medical diagnosis and treatment. The authors and contributors are not responsible for any misuse of this software.

## 📚 References

1. Marcus DS, Fotenos AF, Csernansky JG, Morris JC, Buckner RL. (2010). Open Access Series of Imaging Studies (OASIS): Longitudinal MRI Data in Nondemented and Demented Older Adults. *Journal of Cognitive Neuroscience*, 22(12), 2677-2684.

2. OASIS Dataset: [www.oasis-brains.org](http://www.oasis-brains.org)

3. Original Kaggle Notebook: [Detecting Early Alzheimer's](https://www.kaggle.com/code)

## 👥 Authors

- Original Notebook: Hyunseok Choi, Kyuri Song, Saurin Parikh
- Refactored Package: AI Projects Team

## 🙏 Acknowledgments

- OASIS project for providing the dataset
- Kaggle community for inspiration and code snippets
- Scikit-learn team for excellent ML tools

---

**Made with ❤️ for medical research and education**

