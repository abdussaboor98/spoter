import ast
from typing import Callable, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
import tensorflow as tf

from normalization.body_normalization import BODY_IDENTIFIERS
from normalization.hand_normalization import HAND_IDENTIFIERS
from normalization.body_normalization import normalize_single_dict as normalize_single_body_dict
from normalization.hand_normalization import normalize_single_dict as normalize_single_hand_dict

HAND_IDENTIFIERS = [identifier + "_0" for identifier in HAND_IDENTIFIERS] + [identifier + "_1" for identifier in HAND_IDENTIFIERS]
ALL_IDENTIFIERS = BODY_IDENTIFIERS + HAND_IDENTIFIERS
IDENTIFIER_TO_INDEX = {identifier: index for index, identifier in enumerate(ALL_IDENTIFIERS)}

LEFT_ARM_IDENTIFIERS = ["leftShoulder", "leftElbow", "leftWrist"]
RIGHT_ARM_IDENTIFIERS = ["rightShoulder", "rightElbow", "rightWrist"]


def load_pose_sequences(file_location: str) -> Tuple[Sequence[np.ndarray], Sequence[int]]:
    dataframe = pd.read_csv(file_location, encoding="utf-8")
    dataframe.columns = [item.replace("_left_", "_0_").replace("_right_", "_1_") for item in list(dataframe.columns)]

    if "neck_X" not in dataframe.columns:
        dataframe["neck_X"] = [0 for _ in range(dataframe.shape[0])]
        dataframe["neck_Y"] = [0 for _ in range(dataframe.shape[0])]

    labels = dataframe["labels"].to_list()
    pose_sequences = []

    for _, row in dataframe.iterrows():
        frame_count = len(ast.literal_eval(row["leftEar_X"]))
        sequence_frames = np.empty(
            shape=(frame_count, len(ALL_IDENTIFIERS), 2),
            dtype=np.float32,
        )
        for index, identifier in enumerate(ALL_IDENTIFIERS):
            sequence_frames[:, index, 0] = ast.literal_eval(row[identifier + "_X"])
            sequence_frames[:, index, 1] = ast.literal_eval(row[identifier + "_Y"])
        pose_sequences.append(sequence_frames)

    return pose_sequences, labels


def pose_tensor_to_dict(landmarks_array: np.ndarray) -> dict:
    data_array = np.asarray(landmarks_array)
    landmark_dictionary = {}

    for landmark_index, identifier in enumerate(ALL_IDENTIFIERS):
        landmark_dictionary[identifier] = [
            (float(frame[0]), float(frame[1])) for frame in data_array[:, landmark_index]
        ]

    return landmark_dictionary


def pose_dict_to_tensor(landmarks_dict: dict) -> tf.Tensor:
    sequence_length = len(landmarks_dict["leftEar"])
    pose_tensor = np.empty(
        shape=(sequence_length, len(ALL_IDENTIFIERS), 2), dtype=np.float32
    )

    for landmark_index, identifier in enumerate(ALL_IDENTIFIERS):
        pose_tensor[:, landmark_index, 0] = [frame[0] for frame in landmarks_dict[identifier]]
        pose_tensor[:, landmark_index, 1] = [frame[1] for frame in landmarks_dict[identifier]]

    return tf.convert_to_tensor(pose_tensor, dtype=tf.float32)


class SignPoseDataset:
    """Tensor-based loader for sign pose sequences with normalization, batching, and augmentation support."""

    def __init__(
        self,
        dataset_filename: Optional[str] = None,
        augmentations_prob: float = 0.5,
        normalize: bool = True,
        data: Optional[np.ndarray] = None,
        lengths: Optional[np.ndarray] = None,
        labels: Optional[Sequence[int]] = None,
    ):
        if data is not None and lengths is not None and labels is not None:
            self.pose_sequences = np.array(data, dtype=np.float32)
            self.sequence_lengths = np.array(lengths, dtype=np.int32)
            self.labels = np.array(labels, dtype=np.int32)
        elif dataset_filename is not None:
            loaded_data = load_pose_sequences(dataset_filename)
            pose_sequences, labels = list(loaded_data[0]), list(loaded_data[1])
            self.sequence_lengths = np.array([sequence.shape[0] for sequence in pose_sequences], dtype=np.int32)
            self.max_sequence_length = int(np.max(self.sequence_lengths))
            joint_count = pose_sequences[0].shape[1]
            self.pose_sequences = np.zeros(
                (len(pose_sequences), self.max_sequence_length, joint_count, 2),
                dtype=np.float32,
            )
            for index, sequence in enumerate(pose_sequences):
                length = sequence.shape[0]
                self.pose_sequences[index, :length] = sequence
            self.labels = np.asarray(labels, dtype=np.int32) - 1
        else:
            raise ValueError("Either dataset_filename or data/lengths/labels must be provided.")

        if data is not None:
            self.max_sequence_length = self.pose_sequences.shape[1]

        self.num_samples = self.pose_sequences.shape[0]
        self.num_joints = self.pose_sequences.shape[2]
        self.input_shape = (self.max_sequence_length, self.num_joints, 2)
        self.targets = self.labels.tolist()
        self.augmentations_prob = float(augmentations_prob)

        if dataset_filename is not None and normalize:
            self._normalize_sequences()

        if dataset_filename is not None:
            self.pose_sequences = self.pose_sequences - 0.5

    def _normalize_sequences(self):
        normalized_sequences = np.zeros_like(self.pose_sequences)
        for index in range(self.num_samples):
            length = self.sequence_lengths[index]
            pose_dict = pose_tensor_to_dict(self.pose_sequences[index, :length])
            pose_dict = normalize_single_body_dict(pose_dict)
            pose_dict = normalize_single_hand_dict(pose_dict)
            normalized_tensor = pose_dict_to_tensor(pose_dict).numpy()
            normalized_sequences[index, :length] = normalized_tensor
        self.pose_sequences = normalized_sequences

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx: int):
        pose = tf.convert_to_tensor(self.pose_sequences[idx], dtype=tf.float32)
        length = tf.convert_to_tensor(self.sequence_lengths[idx], dtype=tf.int32)
        label = tf.convert_to_tensor(self.labels[idx], dtype=tf.int32)
        return pose, length, label

    def subset(self, subset_indices: Sequence[int]):
        subset_indices = np.asarray(subset_indices, dtype=np.int32)
        subset_data = self.pose_sequences[subset_indices]
        subset_lengths = self.sequence_lengths[subset_indices]
        subset_labels = self.labels[subset_indices]
        return SignPoseDataset(
            data=subset_data,
            lengths=subset_lengths,
            labels=subset_labels,
            augmentations_prob=self.augmentations_prob,
            normalize=False,
        )

    def as_tf_dataset(
        self,
        batch_size: int,
        shuffle: bool = False,
        augment: bool = False,
        gaussian_noise: Optional[Callable[[tf.Tensor], tf.Tensor]] = None,
    ) -> tf.data.Dataset:
        poses = self.pose_sequences
        lengths = self.sequence_lengths
        labels = self.labels

        dataset = tf.data.Dataset.from_tensor_slices((poses, lengths, labels))
        if shuffle:
            dataset = dataset.shuffle(buffer_size=self.num_samples, reshuffle_each_iteration=True)

        def _map_sample(pose, length, label):
            pose = tf.ensure_shape(pose, self.input_shape)
            mask = tf.sequence_mask(length, maxlen=self.max_sequence_length)
            if augment:
                pose = self._maybe_augment(pose, mask)
            if gaussian_noise is not None:
                pose = gaussian_noise(pose)
            return {"pose": pose, "mask": mask}, label

        dataset = dataset.map(_map_sample, num_parallel_calls=tf.data.AUTOTUNE)
        dataset = dataset.batch(batch_size, drop_remainder=False)
        dataset = dataset.prefetch(tf.data.AUTOTUNE)
        return dataset

    def steps_per_epoch(self, batch_size: int) -> int:
        return int(np.ceil(self.num_samples / batch_size))

    def _maybe_augment(self, pose: tf.Tensor, mask: tf.Tensor) -> tf.Tensor:
        probability = tf.random.uniform([], dtype=tf.float32)
        augment_tensor = tf.constant(self.augmentations_prob, dtype=tf.float32)

        def apply_augmentation():
            choice = tf.random.uniform([], minval=0, maxval=4, dtype=tf.int32)
            augmentations = {
                0: lambda: self._rotate_frames(pose, mask, (-13.0, 13.0)),
                1: lambda: self._shear_frames(pose, mask, axis="x", magnitude_range=(0.0, 0.15)),
                2: lambda: self._shear_frames(pose, mask, axis="y", magnitude_range=(0.0, 0.1)),
                3: lambda: self._rotate_random_arm(pose, mask),
            }
            augmented_pose = tf.switch_case(choice, branch_fns=augmentations)
            return self._apply_mask(augmented_pose, mask)

        return tf.cond(probability < augment_tensor, apply_augmentation, lambda: pose)

    @staticmethod
    def _apply_mask(pose: tf.Tensor, mask: tf.Tensor) -> tf.Tensor:
        expanded_mask = tf.cast(mask, pose.dtype)[:, tf.newaxis, tf.newaxis]
        return pose * expanded_mask

    def _rotate_frames(self, pose: tf.Tensor, mask: tf.Tensor, angle_range: Tuple[float, float]) -> tf.Tensor:
        radians = np.pi / 180.0
        min_angle, max_angle = angle_range
        angle = tf.random.uniform([], minval=min_angle * radians, maxval=max_angle * radians, dtype=tf.float32)
        transform = tf.stack(
            [
                [tf.cos(angle), -tf.sin(angle)],
                [tf.sin(angle), tf.cos(angle)],
            ],
            axis=0,
        )
        return self._apply_linear_transform(pose, transform)

    def _shear_frames(self, pose: tf.Tensor, mask: tf.Tensor, axis: str, magnitude_range: Tuple[float, float]) -> tf.Tensor:
        min_magnitude, max_magnitude = magnitude_range
        magnitude = tf.random.uniform([], minval=min_magnitude, maxval=max_magnitude, dtype=tf.float32)
        direction = tf.random.uniform([], minval=0, maxval=2, dtype=tf.int32)
        magnitude = tf.where(direction == 0, magnitude, -magnitude)

        if axis == "x":
            transform = tf.stack([[1.0, magnitude], [0.0, 1.0]], axis=0)
        else:
            transform = tf.stack([[1.0, 0.0], [magnitude, 1.0]], axis=0)

        return self._apply_linear_transform(pose, transform)

    @staticmethod
    def _apply_linear_transform(pose: tf.Tensor, transform: tf.Tensor) -> tf.Tensor:
        original_shape = tf.shape(pose)
        flattened_pose = tf.reshape(pose, (-1, 2))
        transformed_pose = tf.matmul(flattened_pose, transform)
        return tf.reshape(transformed_pose, original_shape)

    def _rotate_random_arm(self, pose: tf.Tensor, mask: tf.Tensor) -> tf.Tensor:
        arm_choice = tf.random.uniform([], minval=0, maxval=2, dtype=tf.int32)
        left_arm_indices = tf.constant([IDENTIFIER_TO_INDEX[name] for name in LEFT_ARM_IDENTIFIERS], dtype=tf.int32)
        right_arm_indices = tf.constant([IDENTIFIER_TO_INDEX[name] for name in RIGHT_ARM_IDENTIFIERS], dtype=tf.int32)
        joint_indices = tf.cond(arm_choice == 0, lambda: left_arm_indices, lambda: right_arm_indices)
        return self._rotate_arm_segment(pose, joint_indices)

    def _rotate_arm_segment(self, pose: tf.Tensor, joint_indices: tf.Tensor) -> tf.Tensor:
        radians = np.pi / 180.0
        angle = tf.random.uniform([], minval=-4.0 * radians, maxval=4.0 * radians, dtype=tf.float32)
        rotation = tf.stack(
            [
                [tf.cos(angle), -tf.sin(angle)],
                [tf.sin(angle), tf.cos(angle)],
            ],
            axis=0,
        )

        anchor_index = joint_indices[0]
        anchor_positions = pose[:, anchor_index, :]
        joint_values = tf.gather(pose, joint_indices, axis=1)
        offset = joint_values - tf.expand_dims(anchor_positions, axis=1)
        transformed_offsets = tf.matmul(tf.reshape(offset, (-1, 2)), rotation)
        transformed_offsets = tf.reshape(transformed_offsets, tf.shape(offset))
        updated_values = tf.expand_dims(anchor_positions, axis=1) + transformed_offsets

        seq_len = tf.shape(pose)[0]
        repeated_frames = tf.tile(tf.range(seq_len)[:, tf.newaxis], [1, tf.size(joint_indices)])
        repeated_joints = tf.tile(joint_indices[tf.newaxis, :], [seq_len, 1])
        scatter_indices = tf.stack(
            [tf.reshape(repeated_frames, [-1]), tf.reshape(repeated_joints, [-1])],
            axis=1,
        )
        updates = tf.reshape(updated_values, (-1, 2))
        return tf.tensor_scatter_nd_update(pose, scatter_indices, updates)


if __name__ == "__main__":
    pass
