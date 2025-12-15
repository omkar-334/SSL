# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""
Sugarcane Leaf Disease Dataset
Dataset from: nirmalsankalana/sugarcane-leaf-disease-dataset
"""

import math
import os

import numpy as np
from PIL import Image
from torchvision import transforms
from torchvision.datasets import ImageFolder

from semilearn.datasets.augmentation import (
    RandAugment,
    RandomResizedCropAndInterpolation,
)
from semilearn.datasets.utils import split_ssl_data

from .datasetbase import BasicDataset

# ImageNet normalization (standard for most image classification tasks)
dataset_mean = (0.485, 0.456, 0.406)
dataset_std = (0.229, 0.224, 0.225)


def get_sugarcane(
    args,
    alg,
    dataset,
    num_labels,
    num_classes,
    data_dir="./data",
    include_lb_to_ulb=True,
):
    """
    Load sugarcane leaf disease dataset

    Args:
        args: arguments
        alg: algorithm name
        dataset: dataset name (should be 'sugarcane')
        num_labels: number of labeled samples
        num_classes: number of classes
        data_dir: data directory
        include_lb_to_ulb: whether to include labeled data in unlabeled set
    """
    img_size = args.img_size
    crop_ratio = args.crop_ratio

    transform_weak = transforms.Compose([
        transforms.Resize((
            int(math.floor(img_size / crop_ratio)),
            int(math.floor(img_size / crop_ratio)),
        )),
        transforms.RandomCrop((img_size, img_size)),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(dataset_mean, dataset_std),
    ])

    transform_medium = transforms.Compose([
        transforms.Resize((
            int(math.floor(img_size / crop_ratio)),
            int(math.floor(img_size / crop_ratio)),
        )),
        transforms.RandomCrop((img_size, img_size)),
        transforms.RandomHorizontalFlip(),
        RandAugment(1, 5),
        transforms.ToTensor(),
        transforms.Normalize(dataset_mean, dataset_std),
    ])

    transform_strong = transforms.Compose([
        transforms.Resize(int(math.floor(img_size / crop_ratio))),
        RandomResizedCropAndInterpolation((img_size, img_size)),
        transforms.RandomHorizontalFlip(),
        RandAugment(3, 10),
        transforms.ToTensor(),
        transforms.Normalize(dataset_mean, dataset_std),
    ])

    transform_val = transforms.Compose([
        transforms.Resize(math.floor(int(img_size / crop_ratio))),
        transforms.CenterCrop(img_size),
        transforms.ToTensor(),
        transforms.Normalize(dataset_mean, dataset_std),
    ])

    # Load dataset from local directory
    # Expected structure: data_dir/sugarcane/train/ and data_dir/sugarcane/test/
    data_dir = os.path.join(data_dir, dataset.lower())

    if not os.path.exists(data_dir):
        raise FileNotFoundError(
            f"Dataset not found at {data_dir}. "
            f"Expected structure: {data_dir}/train/ and {data_dir}/test/"
        )

    train_dir = os.path.join(data_dir, "train")
    test_dir = os.path.join(data_dir, "test")

    if not os.path.exists(train_dir):
        raise FileNotFoundError(f"Train directory not found at {train_dir}")
    if not os.path.exists(test_dir):
        raise FileNotFoundError(f"Test directory not found at {test_dir}")

    # Load using ImageFolder (standard torchvision format)
    base_dataset = ImageFolder(train_dir)

    # Get all image paths and targets
    train_data = [sample[0] for sample in base_dataset.samples]  # image paths
    train_targets = np.array([sample[1] for sample in base_dataset.samples])

    # Split into labeled and unlabeled
    lb_data, lb_targets, ulb_data, ulb_targets = split_ssl_data(
        args,
        train_data,
        train_targets,
        num_classes,
        lb_num_labels=num_labels,
        ulb_num_labels=args.ulb_num_labels,
        lb_imbalance_ratio=args.lb_imb_ratio,
        ulb_imbalance_ratio=args.ulb_imb_ratio,
        include_lb_to_ulb=include_lb_to_ulb,
    )

    lb_count = [0 for _ in range(num_classes)]
    ulb_count = [0 for _ in range(num_classes)]
    for c in lb_targets:
        lb_count[c] += 1
    for c in ulb_targets:
        ulb_count[c] += 1
    print(f"lb count: {lb_count}")
    print(f"ulb count: {ulb_count}")

    if alg == "fullysupervised":
        lb_data = train_data
        lb_targets = train_targets

    # Create datasets
    lb_dset = SugarcaneDataset(
        alg,
        data_dir,
        split="train",
        idx_list=None,
        data_list=lb_data,
        targets=lb_targets,
        transform=transform_weak,
        transform_strong=None,
        is_ulb=False,
    )

    ulb_dset = SugarcaneDataset(
        alg,
        data_dir,
        split="train",
        idx_list=None,
        data_list=ulb_data,
        targets=ulb_targets,
        transform=transform_weak,
        transform_medium=transform_medium,
        transform_strong=transform_strong,
        is_ulb=True,
    )

    eval_dset = SugarcaneDataset(alg, data_dir, split="test", transform=transform_val)

    print(
        f"#Labeled: {len(lb_dset)} #Unlabeled: {len(ulb_dset)} #Val: {len(eval_dset)}"
    )

    return lb_dset, ulb_dset, eval_dset


class SugarcaneDataset(BasicDataset):
    """
    Sugarcane Leaf Disease Dataset
    """

    def __init__(
        self,
        alg,
        root,
        split="train",
        is_ulb=False,
        idx_list=None,
        data_list=None,
        targets=None,
        transform=None,
        target_transform=None,
        transform_medium=None,
        transform_strong=None,
    ):
        """
        Args:
            alg: algorithm name
            root: root directory containing train/test folders
            split: 'train' or 'test'
            is_ulb: whether this is unlabeled data
            idx_list: list of indices to use (optional)
            data_list: list of image paths (optional)
            targets: pre-loaded targets (optional)
            transform: weak transform
            transform_medium: medium transform (for some algorithms)
            transform_strong: strong transform
        """
        self.alg = alg
        self.is_ulb = is_ulb
        self.medium_transform = transform_medium
        self.strong_transform = transform_strong

        if self.strong_transform is None:
            if self.is_ulb:
                assert self.alg not in [
                    "fullysupervised",
                    "supervised",
                    "pseudolabel",
                    "vat",
                    "pimodel",
                    "meanteacher",
                    "mixmatch",
                    "refixmatch",
                ], f"alg {self.alg} requires strong augmentation"

        if self.medium_transform is None:
            if self.is_ulb:
                assert self.alg not in ["sequencematch"], (
                    f"alg {self.alg} requires medium augmentation"
                )

        # If data_list is provided, use it directly
        if data_list is not None:
            self.data = data_list
            self.targets = np.asarray(targets) if targets is not None else None
        else:
            # Load from ImageFolder structure
            split_dir = os.path.join(root, split)
            if not os.path.exists(split_dir):
                raise FileNotFoundError(f"Split directory not found: {split_dir}")

            base_dataset = ImageFolder(
                split_dir, transform=None, target_transform=target_transform
            )
            self.data = [sample[0] for sample in base_dataset.samples]  # image paths
            self.targets = np.array([sample[1] for sample in base_dataset.samples])

            if idx_list is not None:
                self.data = [self.data[i] for i in idx_list]
                self.targets = self.targets[idx_list]

        # Initialize BasicDataset
        super().__init__(
            alg=alg,
            data=self.data,
            targets=self.targets,
            num_classes=None,  # Will be set from args
            transform=transform,
            is_ulb=is_ulb,
            medium_transform=transform_medium,
            strong_transform=transform_strong,
            onehot=False,
        )

    def __sample__(self, index):
        """Sample an image and target"""
        path = self.data[index]
        try:
            img = Image.open(path).convert("RGB")
        except Exception as e:
            raise RuntimeError(f"Error loading image from {path}: {e}")
        target = self.targets[index] if self.targets is not None else None
        return img, target
