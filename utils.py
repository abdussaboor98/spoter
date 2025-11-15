"""Dataset splitting helpers shared across training scripts."""

from collections import Counter
from typing import Tuple

import numpy as np
from sklearn.model_selection import train_test_split


def stratified_train_validation_split(dataset, validation_ratio: float = 0.0):
    targets = np.array(dataset.targets)
    train_indices, validation_indices = train_test_split(
        np.arange(targets.shape[0]),
        test_size=validation_ratio,
        stratify=targets,
    )

    training_dataset = dataset.subset(train_indices)
    validation_dataset = dataset.subset(validation_indices)

    return training_dataset, validation_dataset


def select_training_subset(dataset, train_fraction: float = 1.0):
    if train_fraction == 1:
        return dataset

    targets = np.array(dataset.targets)
    train_indices, _ = train_test_split(
        np.arange(targets.shape[0]),
        test_size=1 - train_fraction,
        stratify=targets,
    )

    training_dataset = dataset.subset(train_indices)

    return training_dataset


def log_class_distribution(dataset) -> Tuple[int, int]:
    class_counts = dict(Counter(dataset.targets))
    print(class_counts)
    return len(class_counts), sum(class_counts.values())
