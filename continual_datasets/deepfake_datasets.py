"""
Deepfake Dataset Support for AdaPromptCL
Separate module to avoid affecting existing code
"""

import os
import numpy as np
import torch
from torch.utils.data import Dataset, Subset
from torchvision import transforms
from torchvision.datasets.folder import default_loader
from copy import deepcopy

class DeepfakeDataset(Dataset):
    """
    Deepfake dataset for continual learning.
    Each task represents a different generator with real/fake images.
    Labels are assigned as: real = 0 + 2*task_id, fake = 1 + 2*task_id
    During evaluation, use modulo 2 to get binary real/fake classification.
    
    Directory structure expected:
    data_path/
    ├── train/
    │   ├── generator1/
    │   │   ├── real/
    │   │   └── fake/
    │   ├── generator2/
    │   │   ├── real/
    │   │   └── fake/
    │   └── ...
    └── test/
        ├── generator1/
        │   ├── real/
        │   └── fake/
        └── ...
    """
    
    def __init__(self, root, train=True, transform=None, target_transform=None, 
                 task_names=None, task_id=None, download=False):
        self.root = os.path.expanduser(root)
        self.transform = transform
        self.target_transform = target_transform
        self.train = train
        self.task_names = task_names or []
        self.task_id = task_id
        
        if not self._check_exists():
            if download:
                raise NotImplementedError("Download not implemented for Deepfake dataset")
            else:
                raise RuntimeError(f'Dataset not found at {self.root}. Please ensure the data is properly organized.')
        
        self.samples = []
        self.targets = []
        self.classes = set()
        
        # If task_id is specified, load only that task
        if self.task_id is not None:
            self._load_single_task(self.task_id)
        else:
            # Load all tasks (for evaluation across all tasks)
            self._load_all_tasks()
    
    def _load_single_task(self, task_id):
        """Load data for a single task/generator"""
        if task_id >= len(self.task_names):
            raise ValueError(f"Task ID {task_id} exceeds available tasks ({len(self.task_names)})")
        
        generator_name = self.task_names[task_id]
        split = 'train' if self.train else 'test'
        task_root = os.path.join(self.root, split, generator_name)
        
        print(f"Loading task {task_id}: {generator_name} from {task_root}")
        
        # Load real images (label = 0 + 2*task_id)
        real_path = os.path.join(task_root, 'real')
        if os.path.exists(real_path):
            real_count = 0
            for img_name in os.listdir(real_path):
                if img_name.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tiff')):
                    img_path = os.path.join(real_path, img_name)
                    label = 0 + 2 * task_id
                    self.samples.append((img_path, label))
                    self.targets.append(label)
                    self.classes.add(label)
                    real_count += 1
            print(f"  Real images: {real_count}")
        
        # Load fake images (label = 1 + 2*task_id)
        fake_path = os.path.join(task_root, 'fake')
        if os.path.exists(fake_path):
            fake_count = 0
            for img_name in os.listdir(fake_path):
                if img_name.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tiff')):
                    img_path = os.path.join(fake_path, img_name)
                    label = 1 + 2 * task_id
                    self.samples.append((img_path, label))
                    self.targets.append(label)
                    self.classes.add(label)
                    fake_count += 1
            print(f"  Fake images: {fake_count}")
        
        print(f"  Total images for task {task_id}: {len(self.samples)}")
    
    def _load_all_tasks(self):
        """Load data for all tasks (used for evaluation)"""
        total_samples = 0
        for task_id, generator_name in enumerate(self.task_names):
            split = 'train' if self.train else 'test'
            task_root = os.path.join(self.root, split, generator_name)
            
            task_samples = 0
            
            # Load real images
            real_path = os.path.join(task_root, 'real')
            if os.path.exists(real_path):
                for img_name in os.listdir(real_path):
                    if img_name.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tiff')):
                        img_path = os.path.join(real_path, img_name)
                        label = 0 + 2 * task_id
                        self.samples.append((img_path, label))
                        self.targets.append(label)
                        self.classes.add(label)
                        task_samples += 1
            
            # Load fake images
            fake_path = os.path.join(task_root, 'fake')
            if os.path.exists(fake_path):
                for img_name in os.listdir(fake_path):
                    if img_name.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tiff')):
                        img_path = os.path.join(fake_path, img_name)
                        label = 1 + 2 * task_id
                        self.samples.append((img_path, label))
                        self.targets.append(label)
                        self.classes.add(label)
                        task_samples += 1
            
            print(f"Loaded {task_samples} samples for task {task_id}: {generator_name}")
            total_samples += task_samples
        
        print(f"Total samples across all tasks: {total_samples}")
    
    def _check_exists(self):
        """Check if the dataset exists"""
        train_path = os.path.join(self.root, 'train')
        test_path = os.path.join(self.root, 'test')
        return os.path.exists(train_path) and os.path.exists(test_path)
    
    def __getitem__(self, index):
        """
        Args:
            index (int): Index
        Returns:
            tuple: (image, target) where target is the class index
        """
        img_path, target = self.samples[index]
        
        # Load image
        img = default_loader(img_path)
        
        if self.transform is not None:
            img = self.transform(img)
        
        if self.target_transform is not None:
            target = self.target_transform(target)
        
        return img, target
    
    def __len__(self):
        return len(self.samples)
    
    def get_binary_labels(self):
        """Convert labels to binary real/fake (0/1) using modulo operation"""
        return [label % 2 for label in self.targets]


def build_deepfake_dataloader(args):
    """
    Build dataloaders for deepfake continual learning.
    Separate function to avoid affecting existing code.
    """
    from datasets import build_transform, Lambda, target_transform
    import utils
    
    dataloader = list()
    # FIXED: Always create class_mask for deepfake datasets since we need it for evaluation
    class_mask = list()

    transform_train = build_transform(True, args)
    transform_val = build_transform(False, args)
    
    # Initialize classes counter
    args.nb_classes = 0
    
    print(f"Building deepfake dataloaders for {len(args.task_names)} tasks")
    print(f"Task names: {args.task_names}")
    
    for i in range(len(args.task_names)):
        print(f"\n=== Building dataloader for task {i}: {args.task_names[i]} ===")
        
        # Create datasets for this specific task
        dataset_train = DeepfakeDataset(
            root=args.data_path, 
            train=True, 
            transform=transform_train,
            task_names=args.task_names,
            task_id=i
        )
        
        dataset_val = DeepfakeDataset(
            root=args.data_path, 
            train=False, 
            transform=transform_val,
            task_names=args.task_names,
            task_id=i
        )
        
        # Create label transform for non-task-incremental learning
        transform_target = Lambda(target_transform, args.nb_classes)
        
        # FIXED: Always create class_mask for deepfake datasets
        # Each task has 2 classes (real, fake) 
        # Map local classes [0, 1] to global class space
        class_mask.append([args.nb_classes, args.nb_classes + 1])
        args.nb_classes += 2
        print(f"Class mask for task {i}: {class_mask[-1]}")
        print(f"Total classes so far: {args.nb_classes}")
        
        # FIXED: Always apply target transform for deepfake to map to global label space
        if not getattr(args, 'task_inc', False):
            dataset_train.target_transform = transform_target
            dataset_val.target_transform = transform_target
        
        # Handle small dataset scaling if needed
        if hasattr(args, 'small_dataset') and args.small_dataset:
            num_subset = int(len(dataset_train) * args.small_dataset_scale)
            subset_inds = torch.randperm(len(dataset_train))[:num_subset]
            dataset_train = Subset(dataset_train, subset_inds)
            print(f"Using small dataset: {len(dataset_train)} samples")
        
        # Create data loaders
        if args.distributed and utils.get_world_size() > 1:
            num_tasks = utils.get_world_size()
            global_rank = utils.get_rank()
            sampler_train = torch.utils.data.DistributedSampler(
                dataset_train, num_replicas=num_tasks, rank=global_rank, shuffle=True)
            sampler_val = torch.utils.data.SequentialSampler(dataset_val)
        else:
            sampler_train = torch.utils.data.RandomSampler(dataset_train)
            sampler_val = torch.utils.data.SequentialSampler(dataset_val)
        
        data_loader_train = torch.utils.data.DataLoader(
            dataset_train, sampler=sampler_train,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
            pin_memory=args.pin_mem,
        )

        data_loader_val = torch.utils.data.DataLoader(
            dataset_val, sampler=sampler_val,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
            pin_memory=args.pin_mem,
        )

        dataloader.append({'train': data_loader_train, 'val': data_loader_val})
        
        print(f"Train loader: {len(data_loader_train)} batches")
        print(f"Val loader: {len(data_loader_val)} batches")
    
    # Set num_tasks based on actual task count
    args.num_tasks = len(args.task_names)
    
    print(f"\n=== Final Configuration ===")
    print(f"Number of tasks: {args.num_tasks}")
    print(f"Total classes: {args.nb_classes}")
    print(f"Class masks: {class_mask}")
    
    return dataloader, class_mask


def accuracy_binary_deepfake(y_pred, y_true):
    """
    Calculate binary accuracy for deepfake datasets.
    Converts multi-class labels back to binary real/fake using modulo operation.
    
    Args:
        y_pred: Predicted labels (numpy array)
        y_true: True labels (numpy array)
    
    Returns:
        dict: Dictionary containing various accuracy metrics
    """
    assert len(y_pred) == len(y_true), 'Data length error.'
    
    # Convert to binary using modulo operation
    y_pred_binary = y_pred % 2
    y_true_binary = y_true % 2
    
    # Overall binary accuracy
    total_acc = np.around((y_pred_binary == y_true_binary).sum() * 100 / len(y_true), decimals=2)
    
    # Per-task accuracy (every 2 classes form a task)
    task_accs = {}
    max_label = max(np.max(y_true), np.max(y_pred))
    
    for task_start in range(0, max_label + 1, 2):
        task_mask = (y_true >= task_start) & (y_true < task_start + 2)
        if task_mask.sum() > 0:
            task_pred = y_pred_binary[task_mask]
            task_true = y_true_binary[task_mask]
            task_acc = np.around((task_pred == task_true).sum() * 100 / len(task_true), decimals=2)
            task_name = f'Task-{task_start//2}'
            task_accs[task_name] = task_acc
    
    return {
        'total': total_acc,
        'tasks': task_accs,
        'binary_pred': y_pred_binary,
        'binary_true': y_true_binary
    } 