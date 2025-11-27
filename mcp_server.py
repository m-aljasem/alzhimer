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

# Global model
model = None

def load_model():
    """Load model based on type."""
    global model
    try:
        if not MODEL_PATH.exists():
            print(f"⚠️  Model not found at {MODEL_PATH}")
            return
        
        if "sklearn" == "sklearn":
            import pickle
            with open(MODEL_PATH, 'rb') as f:
                model = pickle.load(f)
        elif "sklearn" == "tensorflow":
            import tensorflow as tf
            model = tf.keras.models.load_model(str(MODEL_PATH))
        elif "sklearn" == "pytorch":
            import torch
            from src.model import ECGAttentionModel
            model = ECGAttentionModel(num_classes=2)
            model.load_state_dict(torch.load(MODEL_PATH, map_location=torch.device('cpu')))
            model.eval()
        
        print(f"✓ Model loaded from {MODEL_PATH}")
    except Exception as e:
        print(f"Error loading model: {e}")

@app.list_tools()
async def list_tools() -> List[Tool]:
    """List available tools."""
    return [
        Tool(
            name="predict",
            description="Make a prediction using the Alzheimer's Disease Detection model",
            inputSchema={
                "type": "object",
                "properties": {
                    "input": {
                        "type": "string",
                        "description": "Input data (JSON string or file path)"
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
        return [TextContent(
            type="text",
            text=json.dumps({
                "status": "healthy" if model_loaded else "degraded",
                "model_loaded": model_loaded
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
            "model_type": "sklearn",
            "model_path": str(MODEL_PATH),
            "classes": ['Nondemented', 'Demented'],
            "description": "Alzheimer's Disease Detection"
        }
        
        return [TextContent(
            type="text",
            text=json.dumps(info, indent=2)
        )]
    
    elif name == "predict":
        if model is None:
            return [TextContent(
                type="text",
                text=json.dumps({
                    "error": "Model not loaded. Please train the model first."
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
            
            # Make prediction based on model type
            if "sklearn" == "sklearn":
                # Handle sklearn prediction
                result = {"prediction": "sklearn prediction", "data": data}
            elif "sklearn" == "tensorflow":
                # Handle TensorFlow prediction
                result = {"prediction": "tensorflow prediction", "data": data}
            elif "sklearn" == "pytorch":
                # Handle PyTorch prediction
                result = {"prediction": "pytorch prediction", "data": data}
            
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
