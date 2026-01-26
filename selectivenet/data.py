import os
import sys

base = os.path.join(os.path.dirname(os.path.abspath(__file__)), '../')
sys.path.append(base)

from collections import namedtuple

import torch
import torchvision

class DatasetBuilder(object):
    # tuple for dataset config
    DC = namedtuple('DatasetConfig', ['mean', 'std', 'input_size', 'num_classes'])
    
    DATASET_CONFIG = {
        'svhn' :   DC([0.43768210, 0.44376970, 0.47280442], [0.19803012, 0.20101562, 0.19703614], 32, 10),
        'cifar10': DC([0.49139968, 0.48215841, 0.44653091], [0.24703223, 0.24348513, 0.26158784], 32, 10),
    } 

    def __init__(self, name:str, root_path:str):
        """
        Args
        - name: name of dataset
        - root_path: root path to datasets
        """
        if name not in self.DATASET_CONFIG.keys():
            raise ValueError('name of dataset is invalid')
        self.name = name
        self.root_path = os.path.join(root_path, self.name)

    def __call__(self, train:bool, normalize:bool, augmentation:str='original'):
        input_size = self.DATASET_CONFIG[self.name].input_size
        transform = self._get_transform(self.name, input_size, train, normalize, augmentation)
        if self.name == 'svhn':
            dataset = torchvision.datasets.SVHN(root=self.root_path, split='train' if train else 'test', transform=transform, download=True)
        elif self.name == 'cifar10':
            dataset = torchvision.datasets.CIFAR10(root=self.root_path, train=train, transform=transform, download=True)
        else: 
            raise NotImplementedError 

        return dataset

    def _get_transform(self, name:str, input_size:int, train:bool, normalize:bool, augmentation:str):
        transform = []
        # arugmentation
        if train:
            if augmentation == 'original':
                transform.extend([
                    torchvision.transforms.RandomHorizontalFlip(),
                ])
            elif augmentation == 'tf':
                transform.extend([
                    torchvision.transforms.RandomRotation(degrees=15),
                    torchvision.transforms.RandomAffine(degrees=0, translate=(0.1,0.1), scale=None, shear=None, resample=False, fillcolor=0),
                    torchvision.transforms.RandomHorizontalFlip(),
                ])
            elif augmentation == 'lili':
                transform.extend([
                torchvision.transforms.RandomResizedCrop(input_size),
                torchvision.transforms.RandomHorizontalFlip(),
                ])
            else: raise ValueError('Incorrect augmentation type')
        else:
            pass

        # to tensor
        transform.extend([torchvision.transforms.ToTensor(),])

        # normalize
        if normalize:
            transform.extend([
                torchvision.transforms.Normalize(mean=self.DATASET_CONFIG[name].mean, std=self.DATASET_CONFIG[name].std),
            ])

        return torchvision.transforms.Compose(transform)
    
    @property
    def input_size(self):
        return self.DATASET_CONFIG[self.name].input_size

    @property
    def num_classes(self):
        return self.DATASET_CONFIG[self.name].num_classes

    def get_ood_loader(
        self, 
        ood_name: str, 
        batch_size: int, 
        normalize_to_id: bool = True,
        num_workers: int = 8,
        pin_memory: bool = True
    ):
        """
        Get DataLoader for OOD dataset.
        
        Args:
            ood_name: Name of OOD dataset (e.g., 'svhn')
            batch_size: Batch size for DataLoader
            normalize_to_id: If True, normalize OOD using ID dataset statistics
            num_workers: Number of workers for DataLoader
            pin_memory: Pin memory for DataLoader
        
        Returns:
            DataLoader for OOD dataset
        """
        if ood_name not in self.DATASET_CONFIG.keys():
            raise ValueError(f'OOD dataset {ood_name} is not supported')
        
        # Get normalization stats - use ID dataset if requested
        norm_name = self.name if normalize_to_id else ood_name
        input_size = self.DATASET_CONFIG[ood_name].input_size
        
        # Create transform (no augmentation for OOD evaluation)
        transform = self._get_transform(
            norm_name, input_size, train=False, 
            normalize=True, augmentation='original'
        )
        
        # Load OOD dataset
        ood_root = os.path.join(os.path.dirname(self.root_path), ood_name)
        if ood_name == 'svhn':
            ood_dataset = torchvision.datasets.SVHN(
                root=ood_root, split='test', transform=transform, download=True
            )
        elif ood_name == 'cifar10':
            ood_dataset = torchvision.datasets.CIFAR10(
                root=ood_root, train=False, transform=transform, download=True
            )
        else:
            raise NotImplementedError(f'OOD dataset {ood_name} not implemented')
        
        # Create DataLoader
        ood_loader = torch.utils.data.DataLoader(
            ood_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=pin_memory
        )
        
        return ood_loader

    def _get_mean_and_std(self):
        """
        Function that computes mean and std used in DATASET_CONFIG
        """
        import numpy as np
        if self.name == 'cifar10':
            dataset = torchvision.datasets.CIFAR10(root=self.root_path, train=True, download=True)
            all_data = np.stack([np.asarray(x[0]) for x in dataset])
            mean = np.mean(all_data, axis=(0,2,3)) / 255.0
            std = np.std(all_data, axis=(0,2,3)) / 255.0
        else: 
            raise NotImplementedError 
        return mean, std
            