import pytest
import numpy as np
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

def test_model_import():
    """Test that model can be imported."""
    try:
        from model import build_model
        assert True
    except ImportError:
        pytest.fail("Model import failed")

def test_data_loader():
    """Test data loading functionality."""
    # Add your tests here
    assert True

if __name__ == '__main__':
    pytest.main([__file__, '-v'])
