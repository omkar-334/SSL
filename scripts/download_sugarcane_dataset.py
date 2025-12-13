#!/usr/bin/env python3
"""
Script to download and prepare the sugarcane leaf disease dataset
"""

import os
import sys
from pathlib import Path

def download_sugarcane_dataset(data_dir='./data'):
    """Download sugarcane leaf disease dataset from HuggingFace"""
    try:
        from datasets import load_dataset
        
        print("Loading dataset from HuggingFace...")
        hf_dataset = load_dataset("nirmalsankalana/sugarcane-leaf-disease-dataset")
        
        data_dir = Path(data_dir) / 'sugarcane'
        train_dir = data_dir / 'train'
        test_dir = data_dir / 'test'
        
        train_dir.mkdir(parents=True, exist_ok=True)
        test_dir.mkdir(parents=True, exist_ok=True)
        
        # Get class names
        if 'train' in hf_dataset:
            class_names = hf_dataset['train'].features['label'].names
            num_classes = len(class_names)
            
            print(f"Found {num_classes} classes: {class_names}")
            
            # Create class directories
            for cls_name in class_names:
                (train_dir / cls_name).mkdir(exist_ok=True)
                (test_dir / cls_name).mkdir(exist_ok=True)
            
            # Save train images
            print("Downloading and saving train images...")
            for idx, example in enumerate(hf_dataset['train']):
                img = example['image']
                label = example['label']
                cls_name = class_names[label]
                img_path = train_dir / cls_name / f"{idx}.jpg"
                img.save(str(img_path))
                if (idx + 1) % 100 == 0:
                    print(f"  Saved {idx + 1} train images...")
            
            # Save test/validation images
            if 'test' in hf_dataset:
                print("Downloading and saving test images...")
                for idx, example in enumerate(hf_dataset['test']):
                    img = example['image']
                    label = example['label']
                    cls_name = class_names[label]
                    img_path = test_dir / cls_name / f"{idx}.jpg"
                    img.save(str(img_path))
                    if (idx + 1) % 100 == 0:
                        print(f"  Saved {idx + 1} test images...")
            elif 'validation' in hf_dataset:
                print("Downloading and saving validation images as test...")
                for idx, example in enumerate(hf_dataset['validation']):
                    img = example['image']
                    label = example['label']
                    cls_name = class_names[label]
                    img_path = test_dir / cls_name / f"{idx}.jpg"
                    img.save(str(img_path))
                    if (idx + 1) % 100 == 0:
                        print(f"  Saved {idx + 1} validation images...")
            
            print(f"\nDataset downloaded successfully to {data_dir}")
            print(f"Number of classes: {num_classes}")
            print(f"Class names: {class_names}")
            
            # Count images per class
            print("\nTrain set distribution:")
            for cls_name in class_names:
                count = len(list((train_dir / cls_name).glob("*.jpg")))
                print(f"  {cls_name}: {count} images")
            
            print("\nTest set distribution:")
            for cls_name in class_names:
                count = len(list((test_dir / cls_name).glob("*.jpg")))
                print(f"  {cls_name}: {count} images")
            
            return num_classes, class_names
        else:
            print("Error: 'train' split not found in dataset")
            return None, None
            
    except ImportError:
        print("Error: 'datasets' library not found. Please install it with: pip install datasets")
        sys.exit(1)
    except Exception as e:
        print(f"Error downloading dataset: {e}")
        import traceback
        traceback.print_exc()
        return None, None


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Download sugarcane leaf disease dataset")
    parser.add_argument("--data_dir", type=str, default="./data", help="Data directory")
    args = parser.parse_args()
    
    num_classes, class_names = download_sugarcane_dataset(args.data_dir)
    if num_classes:
        print(f"\n✓ Dataset ready! Number of classes: {num_classes}")
        sys.exit(0)
    else:
        print("\n✗ Failed to download dataset")
        sys.exit(1)

