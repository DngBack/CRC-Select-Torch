import os
import sys

base = os.path.join(os.path.dirname(os.path.abspath(__file__)), '../')
sys.path.append(base)

from collections import namedtuple

import torch
import torchvision
import numpy as np

class DatasetBuilder(object):
    # tuple for dataset config
    DC = namedtuple('DatasetConfig', ['mean', 'std', 'input_size', 'num_classes'])
    
    DATASET_CONFIG = {
        'svhn':       DC([0.43768210, 0.44376970, 0.47280442], [0.19803012, 0.20101562, 0.19703614], 32, 10),
        'cifar10':    DC([0.49139968, 0.48215841, 0.44653091], [0.24703223, 0.24348513, 0.26158784], 32, 10),
        'cifar100':   DC([0.50707516, 0.48654887, 0.44091784], [0.26733429, 0.25643846, 0.27615047], 32, 100),
        'tinyimagenet': DC([0.4802, 0.4481, 0.3975], [0.2770, 0.2691, 0.2821], 64, 200),
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
        elif self.name == 'cifar100':
            dataset = torchvision.datasets.CIFAR100(root=self.root_path, train=train, transform=transform, download=True)
        elif self.name == 'tinyimagenet':
            dataset = self._get_tinyimagenet(train, transform)
        else: 
            raise NotImplementedError(f'Dataset {self.name} not implemented')

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
            elif augmentation == 'randcrop':
                # Standard augmentation for CIFAR-100: RandomCrop + HorizontalFlip
                transform.extend([
                    torchvision.transforms.RandomCrop(input_size, padding=4),
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
        
        Supports: svhn, cifar10, cifar100, cifar10c, cifar100c
        
        Args:
            ood_name: Name of OOD dataset (e.g., 'svhn', 'cifar10c')
            batch_size: Batch size for DataLoader
            normalize_to_id: If True, normalize OOD using ID dataset statistics
            num_workers: Number of workers for DataLoader
            pin_memory: Pin memory for DataLoader
        
        Returns:
            DataLoader for OOD dataset
        """
        # Handle corruption datasets
        if ood_name in ('cifar10c', 'cifar100c'):
            return self._get_corruption_loader(
                ood_name, batch_size, normalize_to_id, num_workers, pin_memory
            )
        
        if ood_name not in self.DATASET_CONFIG.keys():
            raise ValueError(f'OOD dataset {ood_name} is not supported. '
                           f'Available: {list(self.DATASET_CONFIG.keys())} + cifar10c, cifar100c')
        
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
        elif ood_name == 'cifar100':
            ood_dataset = torchvision.datasets.CIFAR100(
                root=ood_root, train=False, transform=transform, download=True
            )
        elif ood_name == 'tinyimagenet':
            ood_dataset = self._get_tinyimagenet(train=False, transform=transform)
        else:
            raise NotImplementedError(f'OOD dataset {ood_name} not implemented')
        
        ood_loader = torch.utils.data.DataLoader(
            ood_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=pin_memory
        )
        
        return ood_loader
    
    def _get_tinyimagenet(self, train, transform):
        """Load Tiny-ImageNet dataset from directory structure."""
        split = 'train' if train else 'val'
        data_dir = os.path.join(self.root_path, split)
        if not os.path.exists(data_dir):
            raise FileNotFoundError(
                f"Tiny-ImageNet not found at {data_dir}. "
                f"Download from http://cs231n.stanford.edu/tiny-imagenet-200.zip "
                f"and extract to {self.root_path}"
            )
        return torchvision.datasets.ImageFolder(data_dir, transform=transform)
    
    def _get_corruption_loader(
        self,
        corruption_name: str,
        batch_size: int,
        normalize_to_id: bool = True,
        num_workers: int = 8,
        pin_memory: bool = True,
        severity: int = 3,
        corruption_type: str = 'gaussian_noise'
    ):
        """
        Load CIFAR-10-C or CIFAR-100-C corruption datasets.
        
        Expected format: numpy files at {dataroot}/CIFAR-10-C/ or CIFAR-100-C/
        with files like: gaussian_noise.npy, labels.npy
        
        Download from: https://zenodo.org/record/2535967
        """
        if corruption_name == 'cifar10c':
            base_name = 'cifar10'
            c_dir = os.path.join(os.path.dirname(self.root_path), 'CIFAR-10-C')
        elif corruption_name == 'cifar100c':
            base_name = 'cifar100'
            c_dir = os.path.join(os.path.dirname(self.root_path), 'CIFAR-100-C')
        else:
            raise ValueError(f"Unknown corruption dataset: {corruption_name}")
        
        data_path = os.path.join(c_dir, f'{corruption_type}.npy')
        label_path = os.path.join(c_dir, 'labels.npy')
        
        if not os.path.exists(data_path):
            raise FileNotFoundError(
                f"Corruption data not found at {data_path}. "
                f"Download from https://zenodo.org/record/2535967"
            )
        
        # Load data: shape (50000*5, 32, 32, 3) for all severities
        images = np.load(data_path)
        labels = np.load(label_path)
        
        # Select severity level (1-5), each has 10000 images
        start_idx = (severity - 1) * 10000
        end_idx = severity * 10000
        images = images[start_idx:end_idx]
        labels = labels[start_idx:end_idx]
        
        # Normalize using ID stats
        norm_name = self.name if normalize_to_id else base_name
        config = self.DATASET_CONFIG.get(norm_name, self.DATASET_CONFIG[base_name])
        
        transform = torchvision.transforms.Compose([
            torchvision.transforms.ToTensor(),
            torchvision.transforms.Normalize(mean=config.mean, std=config.std),
        ])
        
        dataset = CorruptionDataset(images, labels, transform=transform)
        
        loader = torch.utils.data.DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=pin_memory
        )
        return loader

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


class CorruptionDataset(torch.utils.data.Dataset):
    """Dataset wrapper for CIFAR-C style corruption numpy files."""
    
    def __init__(self, images, labels, transform=None):
        """
        Args:
            images: numpy array of shape (N, H, W, 3), uint8
            labels: numpy array of shape (N,)
            transform: torchvision transform
        """
        self.images = images
        self.labels = labels.astype(np.int64)
        self.transform = transform
    
    def __len__(self):
        return len(self.labels)
    
    def __getitem__(self, idx):
        from PIL import Image
        img = Image.fromarray(self.images[idx])
        label = self.labels[idx]
        if self.transform is not None:
            img = self.transform(img)
        return img, label
            