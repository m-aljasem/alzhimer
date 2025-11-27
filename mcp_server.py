"""
MCP Server for Alzheimer's Disease Detection

Model Context Protocol (MCP) server that exposes the model as tools
for AI assistants and other MCP clients.
"""

import asyncio
import json
from pathlib import Path
import sys
from typing import Any, Dict, List, Optional
import numpy as np
import pickle

# MCP SDK
try:
    from mcp.server import Server
    from mcp.server.stdio import stdio_server
    from mcp.types import Tool, TextContent
except ImportError:
    print("⚠️  MCP SDK not installed. Install with: pip install mcp")
    sys.exit(1)

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

app = Server("alzhimer-mcp-server")

# Paths
MODELS_DIR = Path("models")
MODEL_PATH = MODELS_DIR / "best_model.pkl"
SCALER_PATH = MODELS_DIR / "scaler.pkl"

# Global model and scaler
model = None
scaler = None

def load_model():
    """Load scikit-learn model and scaler."""
    global model, scaler
    try:
        if MODEL_PATH.exists():
            with open(MODEL_PATH, 'rb') as f:
                model = pickle.load(f)
            print(f"✓ Model loaded from {MODEL_PATH}")
        else:
            print(f"⚠️  Model not found at {MODEL_PATH}")
            return
        
        if SCALER_PATH.exists():
            with open(SCALER_PATH, 'rb') as f:
                scaler = pickle.load(f)
            print(f"✓ Scaler loaded from {SCALER_PATH}")
        else:
            print(f"⚠️  Scaler not found at {SCALER_PATH}")
    except Exception as e:
        print(f"Error loading model: {e}")

@app.list_tools()
async def list_tools() -> List[Tool]:
    """List available tools."""
    return [
        Tool(
            name="predict",
            description="Make a prediction using the Alzheimer's Disease Detection model. Input should be JSON with keys: gender, age, education, ses, mmse, etiv, nwbv, asf",
            inputSchema={
                "type": "object",
                "properties": {
                    "input": {
                        "type": "string",
                        "description": "JSON string with patient data or file path to JSON file"
                    }
                },
                "required": ["input"]
            }
        ),
        Tool(
            name="model_info",
            description="Get information about the loaded model",
            inputSchema={
                "type": "object",
                "properties": {},
                "required": []
            }
        ),
        Tool(
            name="health_check",
            description="Check if the model is loaded and ready",
            inputSchema={
                "type": "object",
                "properties": {},
                "required": []
            }
        )
    ]

@app.call_tool()
async def call_tool(name: str, arguments: Dict[str, Any]) -> List[TextContent]:
    """Handle tool calls."""
    if name == "health_check":
        model_loaded = model is not None
        scaler_loaded = scaler is not None
        return [TextContent(
            type="text",
            text=json.dumps({
                "status": "healthy" if (model_loaded and scaler_loaded) else "degraded",
                "model_loaded": model_loaded,
                "scaler_loaded": scaler_loaded
            }, indent=2)
        )]
    
    elif name == "model_info":
        if model is None:
            return [TextContent(
                type="text",
                text=json.dumps({
                    "error": "Model not loaded"
                }, indent=2)
            )]
        
        info = {
            "model_type": "scikit-learn",
            "model_path": str(MODEL_PATH),
            "classes": ['Nondemented', 'Demented'],
            "description": "Alzheimer's Disease Detection",
            "features": ["gender", "age", "education", "ses", "mmse", "etiv", "nwbv", "asf"]
        }
        
        return [TextContent(
            type="text",
            text=json.dumps(info, indent=2)
        )]
    
    elif name == "predict":
        if model is None or scaler is None:
            return [TextContent(
                type="text",
                text=json.dumps({
                    "error": "Model or scaler not loaded. Please train the model first."
                }, indent=2)
            )]
        
        try:
            input_data = arguments.get("input", "")
            
            # Parse input (could be JSON string or file path)
            if Path(input_data).exists():
                with open(input_data, 'r') as f:
                    data = json.load(f)
            else:
                data = json.loads(input_data)
            
            # Prepare features
            features = np.array([[
                data.get("gender", 0.0),
                data.get("age", 0.0),
                data.get("education", 0.0),
                data.get("ses", 0.0),
                data.get("mmse", 0.0),
                data.get("etiv", 0.0),
                data.get("nwbv", 0.0),
                data.get("asf", 0.0)
            ]])
            
            # Scale features
            features_scaled = scaler.transform(features)
            
            # Predict
            prediction = model.predict(features_scaled)[0]
            probabilities = model.predict_proba(features_scaled)[0]
            
            class_names = ['Nondemented', 'Demented']
            pred_class = class_names[int(prediction)]
            confidence = float(max(probabilities))
            
            prob_dict = {class_names[i]: float(prob) for i, prob in enumerate(probabilities)}
            
            result = {
                "prediction": pred_class,
                "probability": float(probabilities[int(prediction)]),
                "probabilities": prob_dict,
                "confidence": confidence
            }
            
            return [TextContent(
                type="text",
                text=json.dumps(result, indent=2)
            )]
        except Exception as e:
            return [TextContent(
                type="text",
                text=json.dumps({
                    "error": f"Prediction error: {str(e)}"
                }, indent=2)
            )]
    
    else:
        return [TextContent(
            type="text",
            text=json.dumps({
                "error": f"Unknown tool: {name}"
            }, indent=2)
        )]

async def main():
    """Main entry point."""
    # Load model
    load_model()
    
    # Run server
    async with stdio_server() as (read_stream, write_stream):
        await app.run(
            read_stream,
            write_stream,
            app.create_initialization_options()
        )

if __name__ == "__main__":
    asyncio.run(main())
