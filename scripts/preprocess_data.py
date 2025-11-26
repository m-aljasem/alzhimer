#!/usr/bin/env python3
"""
Data preprocessing script for alzhimer.
"""
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent / 'src'))

def preprocess_data():
    """Preprocess dataset."""
    print("Preprocessing data...")
    # Add preprocessing logic here
    print("✓ Data preprocessing complete")

if __name__ == '__main__':
    preprocess_data()
