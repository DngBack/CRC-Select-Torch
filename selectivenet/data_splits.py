"""
Data splitting utilities for CRC-Select training.
Creates fixed 3-way splits (train/cal/test) for reproducible experiments.
"""
import os
import json
import numpy as np
import torch
from torch.utils.data import DataLoader, Subset
from typing import Tuple, List, Dict
from pathlib import Path


def set_seed(seed: int):
    """Fix all random seeds for reproducibility."""
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def create_splits(
    dataset_size: int,
    seed: int,
    train_ratio: float = 0.8,
    cal_ratio: float = 0.1,
    test_ratio: float = 0.1
) -> Tuple[List[int], List[int], List[int]]:
    """
    Create fixed 3-way splits for train/cal/test.
    
    Args:
        dataset_size: Total number of samples in dataset
        seed: Random seed for reproducibility
        train_ratio: Proportion of data for training (default: 0.8)
        cal_ratio: Proportion of data for calibration (default: 0.1)
        test_ratio: Proportion of data for test (default: 0.1)
    
    Returns:
        train_indices, cal_indices, test_indices
    """
    assert abs(train_ratio + cal_ratio + test_ratio - 1.0) < 1e-6, \
        "Ratios must sum to 1.0"
    
    # Set seed for reproducibility
    set_seed(seed)
    
    # Create shuffled indices
    indices = np.arange(dataset_size)
    np.random.shuffle(indices)
    
    # Compute split sizes
    train_size = int(dataset_size * train_ratio)
    cal_size = int(dataset_size * cal_ratio)
    
    # Split indices
    train_indices = indices[:train_size].tolist()
    cal_indices = indices[train_size:train_size + cal_size].tolist()
    test_indices = indices[train_size + cal_size:].tolist()
    
    return train_indices, cal_indices, test_indices


def save_splits(
    train_indices: List[int],
    cal_indices: List[int],
    test_indices: List[int],
    dataset_name: str,
    seed: int,
    save_dir: str = 'data/splits'
):
    """
    Save split indices to JSON file for reproducibility.
    
    Args:
        train_indices: Training set indices
        cal_indices: Calibration set indices
        test_indices: Test set indices
        dataset_name: Name of dataset (e.g., 'cifar10')
        seed: Random seed used for splitting
        save_dir: Directory to save splits
    """
    os.makedirs(save_dir, exist_ok=True)
    
    splits_data = {
        'seed': seed,
        'dataset': dataset_name,
        'train_indices': train_indices,
        'cal_indices': cal_indices,
        'test_indices': test_indices,
        'sizes': {
            'train': len(train_indices),
            'cal': len(cal_indices),
            'test': len(test_indices)
        }
    }
    
    filepath = os.path.join(save_dir, f'{dataset_name}_seed_{seed}.json')
    with open(filepath, 'w') as f:
        json.dump(splits_data, f, indent=2)
    
    print(f"✓ Saved splits to {filepath}")
    print(f"  Train: {len(train_indices)}, Cal: {len(cal_indices)}, Test: {len(test_indices)}")


def load_splits(
    dataset_name: str,
    seed: int,
    save_dir: str = 'data/splits'
) -> Tuple[List[int], List[int], List[int]]:
    """
    Load split indices from JSON file.
    
    Args:
        dataset_name: Name of dataset (e.g., 'cifar10')
        seed: Random seed used for splitting
        save_dir: Directory where splits are saved
    
    Returns:
        train_indices, cal_indices, test_indices
    """
    filepath = os.path.join(save_dir, f'{dataset_name}_seed_{seed}.json')
    
    if not os.path.exists(filepath):
        raise FileNotFoundError(
            f"Split file not found: {filepath}. "
            f"Please create splits first using create_and_save_splits()."
        )
    
    with open(filepath, 'r') as f:
        splits_data = json.load(f)
    
    return (
        splits_data['train_indices'],
        splits_data['cal_indices'],
        splits_data['test_indices']
    )


def create_and_save_splits(
    dataset,
    dataset_name: str,
    seed: int,
    train_ratio: float = 0.8,
    cal_ratio: float = 0.1,
    test_ratio: float = 0.1,
    save_dir: str = 'data/splits'
) -> Tuple[List[int], List[int], List[int]]:
    """
    Create and save 3-way splits for a dataset.
    
    Args:
        dataset: PyTorch dataset object
        dataset_name: Name of dataset for saving
        seed: Random seed for reproducibility
        train_ratio: Proportion for training
        cal_ratio: Proportion for calibration
        test_ratio: Proportion for test
        save_dir: Directory to save splits
    
    Returns:
        train_indices, cal_indices, test_indices
    """
    train_indices, cal_indices, test_indices = create_splits(
        len(dataset), seed, train_ratio, cal_ratio, test_ratio
    )
    
    save_splits(
        train_indices, cal_indices, test_indices,
        dataset_name, seed, save_dir
    )
    
    return train_indices, cal_indices, test_indices


def get_split_loaders(
    dataset,
    dataset_name: str,
    seed: int,
    batch_size: int,
    num_workers: int = 8,
    pin_memory: bool = True,
    save_dir: str = 'data/splits',
    create_if_missing: bool = True
) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """
    Get DataLoaders for train/cal/test splits.
    
    Args:
        dataset: PyTorch dataset object (should be the training set)
        dataset_name: Name of dataset (e.g., 'cifar10')
        seed: Random seed for splitting
        batch_size: Batch size for DataLoaders
        num_workers: Number of workers for DataLoaders
        pin_memory: Pin memory for DataLoaders
        save_dir: Directory where splits are saved
        create_if_missing: If True, create splits if they don't exist
    
    Returns:
        train_loader, cal_loader, test_loader
    """
    # Try to load existing splits
    try:
        train_indices, cal_indices, test_indices = load_splits(
            dataset_name, seed, save_dir
        )
        print(f"✓ Loaded existing splits for {dataset_name} (seed={seed})")
    except FileNotFoundError:
        if create_if_missing:
            print(f"✓ Creating new splits for {dataset_name} (seed={seed})")
            train_indices, cal_indices, test_indices = create_and_save_splits(
                dataset, dataset_name, seed, save_dir=save_dir
            )
        else:
            raise
    
    # Create subsets
    train_subset = Subset(dataset, train_indices)
    cal_subset = Subset(dataset, cal_indices)
    test_subset = Subset(dataset, test_indices)
    
    # Create DataLoaders
    train_loader = DataLoader(
        train_subset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=pin_memory
    )
    
    cal_loader = DataLoader(
        cal_subset,
        batch_size=batch_size,
        shuffle=False,  # Don't shuffle calibration set
        num_workers=num_workers,
        pin_memory=pin_memory
    )
    
    test_loader = DataLoader(
        test_subset,
        batch_size=batch_size,
        shuffle=False,  # Don't shuffle test set
        num_workers=num_workers,
        pin_memory=pin_memory
    )
    
    return train_loader, cal_loader, test_loader


def create_splits_for_multiple_seeds(
    dataset,
    dataset_name: str,
    seeds: List[int],
    save_dir: str = 'data/splits'
):
    """
    Create and save splits for multiple random seeds.
    
    Args:
        dataset: PyTorch dataset object
        dataset_name: Name of dataset
        seeds: List of random seeds
        save_dir: Directory to save splits
    """
    print(f"Creating splits for {len(seeds)} seeds...")
    for seed in seeds:
        create_and_save_splits(dataset, dataset_name, seed, save_dir=save_dir)
    print(f"✓ Created splits for all {len(seeds)} seeds")


if __name__ == '__main__':
    # Example usage and testing
    import sys
    base = os.path.join(os.path.dirname(os.path.abspath(__file__)), '../')
    sys.path.append(base)
    
    from selectivenet.data import DatasetBuilder
    
    # Create dataset
    dataset_builder = DatasetBuilder(name='cifar10', root_path='../data')
    train_dataset = dataset_builder(train=True, normalize=True, augmentation='original')
    
    print(f"Dataset size: {len(train_dataset)}")
    
    # Create splits for multiple seeds
    seeds = [42, 123, 456]
    create_splits_for_multiple_seeds(train_dataset, 'cifar10', seeds)
    
    # Test loading
    train_loader, cal_loader, test_loader = get_split_loaders(
        train_dataset, 'cifar10', seed=42, batch_size=128
    )
    
    print(f"\nDataLoader sizes:")
    print(f"  Train batches: {len(train_loader)}")
    print(f"  Cal batches: {len(cal_loader)}")
    print(f"  Test batches: {len(test_loader)}")

