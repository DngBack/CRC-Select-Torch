"""
Reproducibility utilities for CRC-Select experiments.
"""
import random
import numpy as np
import torch


def set_seed(seed: int):
    """
    Fix all random seeds for reproducibility.
    
    Args:
        seed: Random seed to use
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    
    # Make cudnn deterministic (may reduce performance)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def get_device():
    """Get the appropriate device for computation."""
    return 'cuda' if torch.cuda.is_available() else 'cpu'


if __name__ == '__main__':
    print("Testing reproducibility utilities...")
    
    # Test seed setting
    set_seed(42)
    x1 = torch.randn(10)
    
    set_seed(42)
    x2 = torch.randn(10)
    
    assert torch.allclose(x1, x2), "Seeds not working properly!"
    print("✓ Seed setting works correctly")
    
    # Test device
    device = get_device()
    print(f"✓ Device: {device}")
    
    print("\n✓ All tests passed!")

