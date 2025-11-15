"""Dataset splitting helpers shared across training scripts."""

from collections import Counter, defaultdict
import csv
from typing import Tuple

import numpy as np
from sklearn.model_selection import train_test_split

import random


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


def user_gesture_repetition_validation_split(dataset, dataset_csv_path: str,
                                              seed: int | None = None):
    """Create a validation split with one repetition per user/gesture pair.

    The CSV is expected to contain the metadata columns ``user``, ``gesture``
    and ``repetition``. For every user/gesture combination a single repetition
    is selected uniformly at random. When multiple variants of the same
    repetition exist (e.g. original and flipped) exactly one of them is chosen
    at random to populate the validation split.
    """

    if not dataset_csv_path:
        raise ValueError("A dataset CSV path is required to construct the validation split.")

    metadata: list[tuple[str, str, str]] = []
    with open(dataset_csv_path, newline="") as csv_file:
        reader = csv.DictReader(csv_file)
        for row in reader:
            metadata.append((row["user"], row["gesture"], row["repetition"]))

    if len(metadata) != len(dataset):
        raise ValueError(
            "Dataset size and CSV metadata length do not match. Ensure the dataset was built from the provided CSV."
        )

    grouped_indices: dict[tuple[str, str], dict[str, list[int]]] = defaultdict(lambda: defaultdict(list))
    for index, (user, gesture, repetition) in enumerate(metadata):
        grouped_indices[(user, gesture)][repetition].append(index)

    rng = random.Random(seed)
    validation_indices: list[int] = []

    for key in sorted(grouped_indices):
        repetitions = grouped_indices[key]
        chosen_repetition = rng.choice(list(repetitions.keys()))
        candidates = repetitions[chosen_repetition]
        validation_indices.append(rng.choice(candidates))

    validation_indices.sort()
    validation_set = set(validation_indices)
    all_indices = np.arange(len(dataset))
    train_indices = [index for index in all_indices if index not in validation_set]

    training_dataset = dataset.subset(train_indices)
    validation_dataset = dataset.subset(validation_indices)

    return training_dataset, validation_dataset
