#!/usr/bin/env python3
"""
Model evaluation script for alzhimer.
"""
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent / 'src'))

def evaluate_model():
    """Evaluate trained model."""
    print("Evaluating model...")
    # Add evaluation logic here
    print("✓ Model evaluation complete")

if __name__ == '__main__':
    evaluate_model()
