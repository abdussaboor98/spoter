
import numpy as np

from collections import Counter
from sklearn.model_selection import train_test_split


def __balance_val_split(dataset, val_split=0.):
    targets = np.array(dataset.targets)
    train_indices, val_indices = train_test_split(
        np.arange(targets.shape[0]),
        test_size=val_split,
        stratify=targets
    )

    train_dataset = dataset.subset(train_indices)
    val_dataset = dataset.subset(val_indices)

    return train_dataset, val_dataset


def __split_of_train_sequence(dataset, train_split=1.0):
    if train_split == 1:
        return dataset

    targets = np.array(dataset.targets)
    train_indices, _ = train_test_split(
        np.arange(targets.shape[0]),
        test_size=1 - train_split,
        stratify=targets
    )

    train_dataset = dataset.subset(train_indices)

    return train_dataset


def __log_class_statistics(dataset):
    train_classes = dataset.targets
    print(dict(Counter(train_classes)))
