import pytest
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

def test_training_pipeline():
    """Test complete training pipeline."""
    # Add integration tests here
    assert True

if __name__ == '__main__':
    pytest.main([__file__, '-v'])
