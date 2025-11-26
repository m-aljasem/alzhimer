import pytest
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

@pytest.fixture
def sample_data():
    """Sample data fixture."""
    import numpy as np
    return np.random.rand(10, 224, 224, 3)
