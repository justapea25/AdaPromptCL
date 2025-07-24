import os
import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision import datasets
from torchvision.datasets.folder import default_loader
from typing import Tuple
import random

class DeepfakeDataset(torch.utils.data.Dataset):
    """
    Custom Deepfake Dataset for continual learning
    Expected structure:
    data_path/
    ├── train/
    │   ├── task1_name/
    │   │   ├── real/
    │   │   └── fake/
    │   └── task2_name/
    │       ├── real/
    │       └── fake/
    └── test/
        ├── task1_name/
        │   ├── real/
        │   └── fake/
        └── task2_name/
            ├── real/
            └── fake/
    """
    
    def __init__(self, root, task_name, task_id, train=True, transform=None, target_transform=None):
        self.root = root
        self.task_name = task_name
        self.task_id = task_id
        self.train = train
        self.transform = transform
        self.target_transform = target_transform
        self.loader = default_loader
        
        # Set up paths
        split = 'train' if train else 'test'
        self.task_path = os.path.join(root, split, task_name)
        
        # Load samples and create class mapping
        self.samples = []
        self.targets = []
        self.classes = set()
        
        self._load_data()
        
        # Convert classes to list and create class_to_idx mapping
        self.classes = ['real', 'fake']  # FIXED: Always binary classes
        self.class_to_idx = {'real': 0, 'fake': 1}  # FIXED: Always binary mapping
        
# In AdaPromptCL/continual_datasets/deepfake_dataset.py

    def _load_data(self):
        """Load data for the specific task"""
        real_count = 0
        fake_count = 0
        
        # FIXED: Keep binary labels (0=real, 1=fake) for each task
        real_label = 0  # Always 0 for real
        fake_label = 1  # Always 1 for fake
        
        # Load real images
        real_path = os.path.join(self.task_path, 'real')
        if os.path.exists(real_path):
            for imgname in os.listdir(real_path):
                if imgname.lower().endswith(('.png', '.jpg', '.jpeg')):
                    img_path = os.path.join(real_path, imgname)
                    self.samples.append(img_path)
                    self.targets.append(real_label)
                    real_count += 1
        
        # Load fake images
        fake_path = os.path.join(self.task_path, 'fake')
        if os.path.exists(fake_path):
            for imgname in os.listdir(fake_path):
                if imgname.lower().endswith(('.png', '.jpg', '.jpeg')):
                    img_path = os.path.join(fake_path, imgname)
                    self.samples.append(img_path)
                    self.targets.append(fake_label)
                    fake_count += 1
        
        split_name = 'train' if self.train else 'test'
        print(f"[{self.task_name}] {split_name} - Real: {real_count}, Fake: {fake_count}, Total: {real_count + fake_count}")
        print(f"[{self.task_name}] Labels: Real={real_label}, Fake={fake_label}")
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        img_path = self.samples[idx]
        target = self.targets[idx]
        
        try:
            image = self.loader(img_path)
        except Exception as e:
            print(f"Error loading image {img_path}: {e}")
            image = Image.new('RGB', (224, 224), (0, 0, 0))
        
        if self.transform:
            image = self.transform(image)
        
        if self.target_transform:
            target = self.target_transform(target)
        
        return image, target 