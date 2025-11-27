"""
RESTful API Server for Alzheimer's Disease Detection

This FastAPI server provides endpoints for model predictions and health checks.
"""

from fastapi import FastAPI, HTTPException, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Optional
import pickle
import numpy as np
import pandas as pd
from pathlib import Path
import sys

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))
from src.data_loader import DataLoader

app = FastAPI(
    title="Alzheimer's Disease Detection API",
    description="RESTful API for Alzheimer's Disease Detection predictions",
    version="1.0.0"
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Paths
MODELS_DIR = Path("models")
MODEL_PATH = MODELS_DIR / "best_model.pkl"
SCALER_PATH = MODELS_DIR / "scaler.pkl"

# Global model and scaler
model = None
scaler = None

class PredictionRequest(BaseModel):
    """Request model for predictions."""
    gender: float  # 1 for Male, 0 for Female
    age: float
    education: float
    ses: float
    mmse: float
    etiv: float
    nwbv: float
    asf: float

class PredictionResponse(BaseModel):
    """Response model for predictions."""
    prediction: str
    probability: float
    probabilities: dict
    confidence: float

@app.on_event("startup")
async def load_model():
    """Load model and scaler on startup."""
    global model, scaler
    try:
        if MODEL_PATH.exists():
            with open(MODEL_PATH, 'rb') as f:
                model = pickle.load(f)
        else:
            print(f"⚠️  Model not found at {MODEL_PATH}. API will return errors until model is trained.")
        
        if SCALER_PATH.exists():
            with open(SCALER_PATH, 'rb') as f:
                scaler = pickle.load(f)
        else:
            print(f"⚠️  Scaler not found at {SCALER_PATH}. API will return errors until model is trained.")
    except Exception as e:
        print(f"Error loading model: {e}")

@app.get("/")
async def root():
    """Root endpoint."""
    return {
        "message": "Alzheimer's Disease Detection API",
        "version": "1.0.0",
        "status": "running"
    }

@app.get("/health")
async def health_check():
    """Health check endpoint."""
    model_loaded = model is not None
    scaler_loaded = scaler is not None
    return {
        "status": "healthy" if (model_loaded and scaler_loaded) else "degraded",
        "model_loaded": model_loaded,
        "scaler_loaded": scaler_loaded
    }

@app.post("/predict", response_model=PredictionResponse)
async def predict(request: PredictionRequest):
    """Make a prediction."""
    if model is None or scaler is None:
        raise HTTPException(
            status_code=503,
            detail="Model or scaler not loaded. Please train the model first."
        )
    
    try:
        # Prepare features
        features = np.array([[
            request.gender,
            request.age,
            request.education,
            request.ses,
            request.mmse,
            request.etiv,
            request.nwbv,
            request.asf
        ]])
        
        # Scale features
        features_scaled = scaler.transform(features)
        
        # Predict
        prediction = model.predict(features_scaled)[0]
        probabilities = model.predict_proba(features_scaled)[0]
        
        # Map prediction
        class_names = ['Nondemented', 'Demented']
        pred_class = class_names[int(prediction)]
        confidence = float(max(probabilities))
        
        prob_dict = {class_names[i]: float(prob) for i, prob in enumerate(probabilities)}
        
        return PredictionResponse(
            prediction=pred_class,
            probability=float(probabilities[int(prediction)]),
            probabilities=prob_dict,
            confidence=confidence
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction error: {str(e)}")

@app.get("/model/info")
async def model_info():
    """Get model information."""
    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    return {
        "model_type": type(model).__name__,
        "features": ['M/F', 'Age', 'EDUC', 'SES', 'MMSE', 'eTIV', 'nWBV', 'ASF'],
        "classes": ['Nondemented', 'Demented']
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
