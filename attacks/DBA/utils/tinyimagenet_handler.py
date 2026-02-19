"""
TinyImageNet-200 Data Handler
Handles the specific directory structure of TinyImageNet-200:
- Train: Each class in separate subdirectories (n00000001/, n00000002/, etc.)
- Val: All images flat in val/images/ with annotations in val_annotations.txt
"""

import os
import torch
from torchvision import datasets, transforms
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import logging

logger = logging.getLogger("logger")


class TinyImageNetValidationDataset(Dataset):
    """Custom dataset for TinyImageNet validation using annotations file"""
    
    def __init__(self, val_dir, annotations_file, transform=None):
        """
        Args:
            val_dir: Path to val/ directory (contains images/ subdirectory)
            annotations_file: Path to val_annotations.txt
            transform: Transforms to apply to images
        """
        self.val_images_dir = os.path.join(val_dir, 'images')
        self.transform = transform
        self.images = []
        self.labels = []
        
        # Create mapping from class name to label
        self.class_to_label = {}
        self.label_to_class = {}
        next_label = 0
        
        # Parse annotations file
        if not os.path.exists(annotations_file):
            raise FileNotFoundError(f"Annotations file not found: {annotations_file}")
        
        with open(annotations_file, 'r') as f:
            for line in f:
                parts = line.strip().split('\t')
                if len(parts) >= 2:
                    image_name = parts[0]
                    class_name = parts[1]
                    
                    # Map class name to integer label
                    if class_name not in self.class_to_label:
                        self.class_to_label[class_name] = next_label
                        self.label_to_class[next_label] = class_name
                        next_label += 1
                    
                    label = self.class_to_label[class_name]
                    self.images.append(image_name)
                    self.labels.append(label)
        
        logger.info(f"TinyImageNet validation dataset loaded with {len(self.images)} images and {len(self.class_to_label)} classes")
        
    def __len__(self):
        return len(self.images)
    
    def __getitem__(self, idx):
        image_name = self.images[idx]
        label = self.labels[idx]
        
        # Load image
        image_path = os.path.join(self.val_images_dir, image_name)
        try:
            image = Image.open(image_path).convert('RGB')
        except Exception as e:
            logger.error(f"Error loading image {image_path}: {e}")
            raise
        
        # Apply transforms
        if self.transform:
            image = self.transform(image)
        
        return image, label


def load_tinyimagenet_train(data_dir, transform=None):
    """
    Load TinyImageNet training dataset using ImageFolder
    
    Args:
        data_dir: Path to tiny-imagenet-200/ directory
        transform: Transform to apply to images
        
    Returns:
        Dataset object with training images
    """
    train_dir = os.path.join(data_dir, 'train')
    
    if not os.path.exists(train_dir):
        raise FileNotFoundError(f"Training directory not found: {train_dir}")
    
    # Check if subdirectories exist
    subdirs = [d for d in os.listdir(train_dir) if os.path.isdir(os.path.join(train_dir, d))]
    if len(subdirs) == 0:
        raise ValueError(f"No class subdirectories found in {train_dir}. "
                        "Expected directories like n00000001/, n00000002/, etc.")
    
    logger.info(f"Loading TinyImageNet training dataset with {len(subdirs)} classes from {train_dir}")
    
    if transform is None:
        transform = transforms.Compose([
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
        ])
    
    train_dataset = datasets.ImageFolder(train_dir, transform=transform)
    logger.info(f"Training dataset loaded: {len(train_dataset)} samples, {len(train_dataset.classes)} classes")
    
    return train_dataset


def load_tinyimagenet_val(data_dir, transform=None):
    """
    Load TinyImageNet validation dataset using annotations file
    
    Args:
        data_dir: Path to tiny-imagenet-200/ directory
        transform: Transform to apply to images
        
    Returns:
        Dataset object with validation images
    """
    val_dir = os.path.join(data_dir, 'val')
    annotations_file = os.path.join(val_dir, 'val_annotations.txt')
    
    # Validate paths
    if not os.path.exists(val_dir):
        raise FileNotFoundError(f"Validation directory not found: {val_dir}")
    
    if not os.path.exists(annotations_file):
        raise FileNotFoundError(f"Annotations file not found: {annotations_file}")
    
    val_images_dir = os.path.join(val_dir, 'images')
    if not os.path.exists(val_images_dir):
        raise FileNotFoundError(f"Validation images directory not found: {val_images_dir}")
    
    logger.info(f"Loading TinyImageNet validation dataset from {val_dir}")
    
    if transform is None:
        transform = transforms.Compose([
            transforms.ToTensor(),
        ])
    
    val_dataset = TinyImageNetValidationDataset(val_dir, annotations_file, transform=transform)
    logger.info(f"Validation dataset loaded: {len(val_dataset)} samples, {len(val_dataset.class_to_label)} classes")
    
    return val_dataset


def verify_tinyimagenet_structure(data_dir):
    """
    Verify TinyImageNet directory structure
    
    Args:
        data_dir: Path to tiny-imagenet-200/ directory
        
    Returns:
        Dictionary with structure information
    """
    info = {
        'has_train': False,
        'has_val': False,
        'train_classes': 0,
        'train_images': 0,
        'val_images': 0,
        'has_annotations': False
    }
    
    train_dir = os.path.join(data_dir, 'train')
    val_dir = os.path.join(data_dir, 'val')
    annotations_file = os.path.join(val_dir, 'val_annotations.txt')
    
    if os.path.exists(train_dir):
        info['has_train'] = True
        subdirs = [d for d in os.listdir(train_dir) if os.path.isdir(os.path.join(train_dir, d))]
        info['train_classes'] = len(subdirs)
        
        for class_dir in subdirs:
            images = [f for f in os.listdir(os.path.join(train_dir, class_dir)) 
                     if f.endswith(('.jpg', '.png', '.JPEG'))]
            info['train_images'] += len(images)
    
    if os.path.exists(val_dir):
        info['has_val'] = True
        val_images_dir = os.path.join(val_dir, 'images')
        if os.path.exists(val_images_dir):
            val_images = [f for f in os.listdir(val_images_dir) 
                         if f.endswith(('.jpg', '.png', '.JPEG'))]
            info['val_images'] = len(val_images)
    
    if os.path.exists(annotations_file):
        info['has_annotations'] = True
    
    return info


if __name__ == '__main__':
    # Test the handler
    import logging
    logging.basicConfig(level=logging.INFO)
    
    data_dir = './data/tiny-imagenet-200/'
    
    # Verify structure
    print("Verifying TinyImageNet structure...")
    info = verify_tinyimagenet_structure(data_dir)
    print(f"Structure info: {info}")
    
    # Load datasets
    print("\nLoading training dataset...")
    train_dataset = load_tinyimagenet_train(data_dir)
    print(f"Train dataset size: {len(train_dataset)}")
    
    print("\nLoading validation dataset...")
    val_dataset = load_tinyimagenet_val(data_dir)
    print(f"Val dataset size: {len(val_dataset)}")
