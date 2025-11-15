import ast
import random
from random import randrange
from typing import Callable, List, Optional, Sequence

import numpy as np
import pandas as pd
import tensorflow as tf

from augmentations import *
from normalization.body_normalization import BODY_IDENTIFIERS
from normalization.hand_normalization import HAND_IDENTIFIERS
from normalization.body_normalization import normalize_single_dict as normalize_single_body_dict
from normalization.hand_normalization import normalize_single_dict as normalize_single_hand_dict

HAND_IDENTIFIERS = [id + "_0" for id in HAND_IDENTIFIERS] + [id + "_1" for id in HAND_IDENTIFIERS]


def load_dataset(file_location: str):

    # Load the datset csv file
    df = pd.read_csv(file_location, encoding="utf-8")

    # TO BE DELETED
    df.columns = [item.replace("_left_", "_0_").replace("_right_", "_1_") for item in list(df.columns)]
    if "neck_X" not in df.columns:
        df["neck_X"] = [0 for _ in range(df.shape[0])]
        df["neck_Y"] = [0 for _ in range(df.shape[0])]

    # TEMP
    labels = df["labels"].to_list()
    # labels = [label + 1 for label in df["labels"].to_list()]
    data = []

    for row_index, row in df.iterrows():
        current_row = np.empty(
            shape=(len(ast.literal_eval(row["leftEar_X"])), len(BODY_IDENTIFIERS + HAND_IDENTIFIERS), 2),
            dtype=np.float32,
        )
        for index, identifier in enumerate(BODY_IDENTIFIERS + HAND_IDENTIFIERS):
            current_row[:, index, 0] = ast.literal_eval(row[identifier + "_X"])
            current_row[:, index, 1] = ast.literal_eval(row[identifier + "_Y"])

        data.append(current_row)

    return data, labels


def tensor_to_dictionary(landmarks_array: np.ndarray) -> dict:

    data_array = np.asarray(landmarks_array)
    output = {}

    for landmark_index, identifier in enumerate(BODY_IDENTIFIERS + HAND_IDENTIFIERS):
        output[identifier] = [
            (float(frame[0]), float(frame[1])) for frame in data_array[:, landmark_index]
        ]

    return output


def dictionary_to_tensor(landmarks_dict: dict) -> tf.Tensor:

    sequence_length = len(landmarks_dict["leftEar"])
    output = np.empty(
        shape=(sequence_length, len(BODY_IDENTIFIERS + HAND_IDENTIFIERS), 2), dtype=np.float32
    )

    for landmark_index, identifier in enumerate(BODY_IDENTIFIERS + HAND_IDENTIFIERS):
        output[:, landmark_index, 0] = [frame[0] for frame in landmarks_dict[identifier]]
        output[:, landmark_index, 1] = [frame[1] for frame in landmarks_dict[identifier]]

    return tf.convert_to_tensor(output, dtype=tf.float32)


class CzechSLRDataset:
    """Advanced object representation of the HPOES dataset for loading hand joints landmarks utilizing the Torch's
    built-in Dataset properties"""

    data: List[np.ndarray]
    labels: List[int]

    def __init__(
            self,
            dataset_filename: Optional[str] = None,
            num_labels: int = 5,
            transform: Optional[Callable[[tf.Tensor], tf.Tensor]] = None,
            augmentations: bool = False,
            augmentations_prob: float = 0.5,
            normalize: bool = True,
            data: Optional[Sequence[np.ndarray]] = None,
            labels: Optional[Sequence[int]] = None,
            indices: Optional[Sequence[int]] = None,
    ):
        """
        Initiates the HPOESDataset with the pre-loaded data from the h5 file.

        :param dataset_filename: Path to the h5 file
        :param transform: Any data transformation to be applied (default: None)
        """

        if data is not None and labels is not None:
            self._base_data = list(data)
            self._base_labels = list(labels)
        elif dataset_filename is not None:
            loaded_data = load_dataset(dataset_filename)
            self._base_data, self._base_labels = list(loaded_data[0]), list(loaded_data[1])
        else:
            raise ValueError("Either dataset_filename or data/labels must be provided.")

        if indices is None:
            self.indices = list(range(len(self._base_data)))
        else:
            self.indices = list(indices)

        self.data = self._base_data
        self.labels = self._base_labels
        self.targets = [self._base_labels[i] for i in self.indices]
        self.num_labels = num_labels
        self.transform = transform

        self.augmentations = augmentations
        self.augmentations_prob = augmentations_prob
        self.normalize = normalize

    def __getitem__(self, idx: int):
        """
        Allocates, potentially transforms and returns the item at the desired index.

        :param idx: Index of the item
        :return: Tuple containing both the depth map and the label
        """

        base_index = self.indices[idx]
        depth_map = np.copy(self.data[base_index])
        label = tf.convert_to_tensor(self.labels[base_index] - 1, dtype=tf.int32)

        depth_map = tensor_to_dictionary(depth_map)

        # Apply potential augmentations
        if self.augmentations and random.random() < self.augmentations_prob:

            selected_aug = randrange(4)

            if selected_aug == 0:
                depth_map = augment_rotate(depth_map, (-13, 13))

            if selected_aug == 1:
                depth_map = augment_shear(depth_map, "perspective", (0, 0.1))

            if selected_aug == 2:
                depth_map = augment_shear(depth_map, "squeeze", (0, 0.15))

            if selected_aug == 3:
                depth_map = augment_arm_joint_rotate(depth_map, 0.3, (-4, 4))

        if self.normalize:
            depth_map = normalize_single_body_dict(depth_map)
            depth_map = normalize_single_hand_dict(depth_map)

        depth_map = dictionary_to_tensor(depth_map)

        # Move the landmark position interval to improve performance
        depth_map = depth_map - 0.5

        if self.transform:
            depth_map = self.transform(depth_map)

        return depth_map, label

    def __len__(self):
        return len(self.indices)

    def subset(self, subset_indices: Sequence[int]):
        return CzechSLRDataset(
            num_labels=self.num_labels,
            transform=self.transform,
            augmentations=self.augmentations,
            augmentations_prob=self.augmentations_prob,
            normalize=self.normalize,
            data=self._base_data,
            labels=self._base_labels,
            indices=[self.indices[i] for i in subset_indices],
        )


if __name__ == "__main__":
    pass
