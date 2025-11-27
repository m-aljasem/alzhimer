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

Typical performance metrics:

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
- **Dataset**: [MRI and Alzheimer's Dataset](https://www.kaggle.com/jboysen/mri-and-alzheimers)
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




## 🌐 RESTful API

The project includes a FastAPI server for programmatic access to the model. This allows you to integrate predictions into your own applications, web services, or scripts.

### Installation

Make sure you have installed all dependencies:

```bash
pip install -r requirements.txt
```

### Starting the API Server

Start the API server using one of these methods:

**Method 1: Direct Python execution**
```bash
python api.py
```

**Method 2: Using uvicorn directly**
```bash
uvicorn api:app --reload --host 0.0.0.0 --port 8000
```

**Method 3: Production mode (no auto-reload)**
```bash
uvicorn api:app --host 0.0.0.0 --port 8000 --workers 4
```

The API will be available at `http://localhost:8000`

### API Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/` | GET | Root endpoint with API information |
| `/health` | GET | Health check endpoint (checks if model is loaded) |
| `/model/info` | GET | Get detailed model information |
| `/predict` | POST | Make a prediction |

### Interactive API Documentation

Once the server is running, you can access interactive documentation:

- **Swagger UI**: `http://localhost:8000/docs` - Interactive API explorer with "Try it out" feature
- **ReDoc**: `http://localhost:8000/redoc` - Beautiful, responsive API documentation

### Using the API

#### Health Check

```python
import requests

response = requests.get("http://localhost:8000/health")
print(response.json())
# Output: {"status": "healthy", "model_loaded": true}
```

#### Get Model Information

```python
import requests

response = requests.get("http://localhost:8000/model/info")
print(response.json())
# Output: Model type, input shape, classes, etc.
```

#### Make Predictions

# Example: Alzheimer's Disease Prediction
import requests

# Prepare patient data
patient_data = {
    "gender": 1.0,      # 1 for Male, 0 for Female
    "age": 75.0,
    "education": 14.0,
    "ses": 2.0,
    "mmse": 27.0,
    "etiv": 1490.0,
    "nwbv": 0.73,
    "asf": 1.20
}

# Make prediction
response = requests.post("http://localhost:8000/predict", json=patient_data)
result = response.json()

print(f"Prediction: {result['prediction']}")
print(f"Confidence: {result['confidence']:.2%}")
print(f"Probabilities: {result['probabilities']}")

### Using cURL

You can also use cURL to interact with the API:

**Health Check:**
```bash
curl http://localhost:8000/health
```

**Get Model Info:**
```bash
curl http://localhost:8000/model/info
```

**Make Prediction (for image-based models):**
```bash
curl -X POST "http://localhost:8000/predict" \
  -F "file=@your_image.jpg"
```

**Make Prediction (for JSON-based models like Alzheimer's):**
```bash
curl -X POST "http://localhost:8000/predict" \
  -H "Content-Type: application/json" \
  -d '{"gender": 1.0, "age": 75.0, "education": 14.0, "ses": 2.0, "mmse": 27.0, "etiv": 1490.0, "nwbv": 0.73, "asf": 1.20}'
```

### Error Handling

The API returns appropriate HTTP status codes:

- `200 OK` - Successful request
- `400 Bad Request` - Invalid input data
- `503 Service Unavailable` - Model not loaded (train the model first)
- `500 Internal Server Error` - Server error during prediction

Example error handling:

```python
import requests

try:
    response = requests.post("http://localhost:8000/predict", json=data)
    response.raise_for_status()  # Raises exception for bad status codes
    result = response.json()
except requests.exceptions.HTTPError as e:
    print(f"HTTP Error: {e}")
    print(f"Response: {response.json()}")
except requests.exceptions.RequestException as e:
    print(f"Request Error: {e}")
```

### API Response Format

**Successful Prediction Response:**
```json
{
    "prediction": "Demented",
    "confidence": 0.85,
    "probabilities": {
        "Nondemented": 0.15,
        "Demented": 0.85
    }
}
```

**Error Response:**
```json
{
    "detail": "Model not loaded. Please train the model first."
}
```

### Deployment

For production deployment, consider:

1. **Using a production ASGI server**: Use `uvicorn` with multiple workers or `gunicorn` with uvicorn workers
2. **Adding authentication**: Implement API keys or OAuth2
3. **Rate limiting**: Add rate limiting to prevent abuse
4. **Logging**: Configure proper logging for monitoring
5. **HTTPS**: Use HTTPS in production with SSL certificates

Example production command:
```bash
uvicorn api:app --host 0.0.0.0 --port 8000 --workers 4 --log-level info
```

## 🔌 MCP Server

The project includes a Model Context Protocol (MCP) server that exposes the model as tools for AI assistants and other MCP-compatible clients. This allows AI assistants like Claude, ChatGPT, or custom MCP clients to interact with your model.

### What is MCP?

Model Context Protocol (MCP) is a standardized protocol for AI assistants to interact with external tools and services. It enables AI assistants to:
- Call your model for predictions
- Get model information
- Check model health status

### Installation

The MCP server requires the MCP SDK:

```bash
pip install mcp
```

### Starting the MCP Server

Start the MCP server:

```bash
python mcp_server.py
```

The server runs as a stdio-based server, communicating via standard input/output. It's designed to be used with MCP clients.

### MCP Tools

The server exposes the following tools:

| Tool | Description | Parameters |
|------|-------------|------------|
| `predict` | Make a prediction using the model | `input` (string): Input data as JSON string or file path |
| `model_info` | Get information about the loaded model | None |
| `health_check` | Check if the model is loaded and ready | None |

### Using MCP with Python Client

```python
from mcp import ClientSession, StdioServerParameters
import asyncio
import json

async def main():
    # Connect to MCP server
    async with ClientSession(
        StdioServerParameters(
            command="python",
            args=["mcp_server.py"],
            env=None
        )
    ) as session:
        # Initialize the session
        await session.initialize()
        
        # List available tools
        tools = await session.list_tools()
        print("Available tools:", [tool.name for tool in tools])
        
        # Health check
        health_result = await session.call_tool(
            "health_check",
            {}
        )
        print("Health:", health_result.content[0].text)
        
        # Get model info
        model_info = await session.call_tool(
            "model_info",
            {}
        )
        print("Model Info:", model_info.content[0].text)
        
        # Make prediction
        # For image-based models, provide base64 encoded image or file path
        prediction_input = json.dumps({
            "file_path": "test_image.jpg"
        })
        
        prediction_result = await session.call_tool(
            "predict",
            {"input": prediction_input}
        )
        print("Prediction:", prediction_result.content[0].text)

if __name__ == "__main__":
    asyncio.run(main())
```

### Using MCP with Claude Desktop

To use with Claude Desktop, add this to your MCP configuration file:

```json
{
  "mcpServers": {
    "alzhimer": {
      "command": "python",
      "args": ["/home/m-aljasem/projects/ai-projects/alzhimer/mcp_server.py"],
      "env": {
        "PYTHONPATH": "/home/m-aljasem/projects/ai-projects/alzhimer"
      }
    }
  }
}
```

### MCP Tool Responses

**Health Check Response:**
```json
{
  "status": "healthy",
  "model_loaded": true
}
```

**Model Info Response:**
```json
{
  "model_type": "TensorFlow/Keras",
  "model_path": "models/model.h5",
  "classes": ["Class1", "Class2"],
  "description": "Model description"
}
```

**Prediction Response:**
```json
{
  "prediction": "Class1",
  "confidence": 0.95,
  "probabilities": {
    "Class1": 0.95,
    "Class2": 0.05
  }
}
```

### Error Handling

The MCP server returns error messages in JSON format:

```json
{
  "error": "Model not loaded. Please train the model first."
}
```

### Integration Examples

**Example 1: Batch Predictions**
```python
import asyncio
from mcp import ClientSession, StdioServerParameters

async def batch_predict(file_paths):
    async with ClientSession(
        StdioServerParameters(
            command="python",
            args=["mcp_server.py"]
        )
    ) as session:
        await session.initialize()
        
        results = []
        for file_path in file_paths:
            result = await session.call_tool(
                "predict",
                {"input": json.dumps({"file_path": file_path})}
            )
            results.append(json.loads(result.content[0].text))
        
        return results

# Usage
predictions = asyncio.run(batch_predict([
    "image1.jpg",
    "image2.jpg",
    "image3.jpg"
]))
```

**Example 2: Model Monitoring**
```python
import asyncio
from mcp import ClientSession, StdioServerParameters
import time

async def monitor_model():
    async with ClientSession(
        StdioServerParameters(
            command="python",
            args=["mcp_server.py"]
        )
    ) as session:
        await session.initialize()
        
        while True:
            health = await session.call_tool("health_check", {})
            print(f"[{time.strftime('%H:%M:%S')}] Health: {health.content[0].text}")
            await asyncio.sleep(60)  # Check every minute

# Run monitoring
asyncio.run(monitor_model())
```

### Troubleshooting

**Issue: MCP server not starting**
- Ensure `mcp` package is installed: `pip install mcp`
- Check that the model file exists in the `models/` directory
- Verify Python path and dependencies

**Issue: Tool calls failing**
- Check that the model is trained and weights are saved
- Verify input format matches expected format
- Check server logs for detailed error messages

**Issue: Connection errors**
- Ensure the MCP server process is running
- Check that stdio communication is working
- Verify environment variables if needed


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

## 👥 Authors

- **Mohamad AlJasem** (MD, MPH, MSc)
  - Email: mohamad@aljasem.eu.org
  - GitHub: [@m-aljasem](https://github.com/m-aljasem)
  - Website: [aljasem.eu.org](https://aljasem.eu.org)

## 🙏 Acknowledgments

- OASIS project for providing the dataset
- Scikit-learn team for excellent ML tools

---

**Made with ❤️ for medical research and education**

