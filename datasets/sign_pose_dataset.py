import ast
from pathlib import Path
from typing import Callable, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
import tensorflow as tf

from normalization.body_normalization import BODY_IDENTIFIERS
from normalization.hand_normalization import HAND_IDENTIFIERS as BASE_HAND_IDENTIFIERS

LEFT_HAND_IDENTIFIERS = [identifier + "_0" for identifier in BASE_HAND_IDENTIFIERS]
RIGHT_HAND_IDENTIFIERS = [identifier + "_1" for identifier in BASE_HAND_IDENTIFIERS]
HAND_IDENTIFIERS = LEFT_HAND_IDENTIFIERS + RIGHT_HAND_IDENTIFIERS
ALL_IDENTIFIERS = BODY_IDENTIFIERS + HAND_IDENTIFIERS
IDENTIFIER_TO_INDEX = {identifier: index for index, identifier in enumerate(ALL_IDENTIFIERS)}

LEFT_ARM_IDENTIFIERS = ["leftShoulder", "leftElbow", "leftWrist"]
RIGHT_ARM_IDENTIFIERS = ["rightShoulder", "rightElbow", "rightWrist"]

BODY_ANCHOR_IDENTIFIERS = {
    "neck": "neck",
    "nose": "nose",
    "left_shoulder": "leftShoulder",
    "right_shoulder": "rightShoulder",
    "left_eye": "leftEye",
}


def _load_raw_pose_sequences(file_location: str) -> Tuple[Sequence[np.ndarray], Sequence[int]]:
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


def _cache_file_path(csv_path: str) -> Path:
    csv_path_obj = Path(csv_path)
    return csv_path_obj.with_suffix(csv_path_obj.suffix + ".npz")


def load_or_cache_sequences(file_location: str) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    cache_path = _cache_file_path(file_location)
    if cache_path.exists():
        cached = np.load(cache_path, allow_pickle=False)
        return cached["poses"], cached["lengths"], cached["labels"]

    pose_sequences, labels = _load_raw_pose_sequences(file_location)
    lengths = np.array([sequence.shape[0] for sequence in pose_sequences], dtype=np.int32)
    max_length = int(np.max(lengths))
    joint_count = pose_sequences[0].shape[1]
    padded_sequences = np.zeros((len(pose_sequences), max_length, joint_count, 2), dtype=np.float32)
    for index, sequence in enumerate(pose_sequences):
        length = sequence.shape[0]
        padded_sequences[index, :length] = sequence

    np.savez(
        cache_path,
        poses=padded_sequences,
        lengths=lengths,
        labels=np.asarray(labels, dtype=np.int32),
    )
    return padded_sequences, lengths, np.asarray(labels, dtype=np.int32)


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
            cached_data = load_or_cache_sequences(dataset_filename)
            self.pose_sequences, self.sequence_lengths, raw_labels = cached_data
            self.labels = np.asarray(raw_labels, dtype=np.int32) - 1
        else:
            raise ValueError("Either dataset_filename or data/lengths/labels must be provided.")

        self.max_sequence_length = self.pose_sequences.shape[1]

        self.num_samples = self.pose_sequences.shape[0]
        self.num_joints = self.pose_sequences.shape[2]
        self.input_shape = (self.max_sequence_length, self.num_joints, 2)
        self.targets = self.labels.tolist()
        self.augmentations_prob = float(augmentations_prob)
        self.normalize = normalize
        self.body_joint_mask = tf.constant(
            [identifier in BODY_IDENTIFIERS for identifier in ALL_IDENTIFIERS],
            dtype=tf.bool,
        )
        self.left_hand_indices = tf.constant(
            [IDENTIFIER_TO_INDEX[identifier] for identifier in LEFT_HAND_IDENTIFIERS],
            dtype=tf.int32,
        )
        self.right_hand_indices = tf.constant(
            [IDENTIFIER_TO_INDEX[identifier] for identifier in RIGHT_HAND_IDENTIFIERS],
            dtype=tf.int32,
        )
        self.left_arm_indices = tf.constant(
            [IDENTIFIER_TO_INDEX[identifier] for identifier in LEFT_ARM_IDENTIFIERS],
            dtype=tf.int32,
        )
        self.right_arm_indices = tf.constant(
            [IDENTIFIER_TO_INDEX[identifier] for identifier in RIGHT_ARM_IDENTIFIERS],
            dtype=tf.int32,
        )
        self.body_anchor_indices = {
            key: IDENTIFIER_TO_INDEX[value]
            for key, value in BODY_ANCHOR_IDENTIFIERS.items()
        }

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
            normalize=self.normalize,
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
            pose = self._apply_mask(pose, mask)
            if augment:
                pose = self._maybe_augment(pose, mask)
            if self.normalize:
                pose = self._normalize_pose(pose, mask)
            pose = pose - 0.5
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
        joint_indices = tf.cond(
            arm_choice == 0,
            lambda: self.left_arm_indices,
            lambda: self.right_arm_indices,
        )
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

    def _normalize_pose(self, pose: tf.Tensor, mask: tf.Tensor) -> tf.Tensor:
        normalized_pose = self._normalize_body_joints(pose, mask)
        normalized_pose = self._normalize_hands(normalized_pose, mask, self.left_hand_indices)
        normalized_pose = self._normalize_hands(normalized_pose, mask, self.right_hand_indices)
        return normalized_pose

    def _normalize_body_joints(self, pose: tf.Tensor, mask: tf.Tensor) -> tf.Tensor:
        pose_dtype = pose.dtype
        mask = tf.cast(mask, tf.bool)

        indices = self.body_anchor_indices
        left_shoulder = pose[:, indices["left_shoulder"], :]
        right_shoulder = pose[:, indices["right_shoulder"], :]
        neck = pose[:, indices["neck"], :]
        nose = pose[:, indices["nose"], :]
        left_eye = pose[:, indices["left_eye"], :]

        left_shoulder_valid = self._joint_valid_mask(left_shoulder)
        right_shoulder_valid = self._joint_valid_mask(right_shoulder)
        neck_valid = self._joint_valid_mask(neck)
        nose_valid = self._joint_valid_mask(nose)
        left_eye_valid = self._joint_valid_mask(left_eye)

        shoulder_valid = left_shoulder_valid & right_shoulder_valid
        head_vector = tf.linalg.norm(left_shoulder - right_shoulder, axis=-1)
        fallback_vector = tf.linalg.norm(neck - nose, axis=-1)
        head_metric = tf.where(shoulder_valid, head_vector, fallback_vector)
        head_metric_valid = tf.where(shoulder_valid, shoulder_valid, neck_valid & nose_valid)
        head_metric_valid = head_metric_valid & mask
        head_metric = tf.where(head_metric_valid, head_metric, tf.zeros_like(head_metric))

        neck_x = neck[:, 0]
        neck_x_valid = neck_valid & mask
        eye_y = tf.where(left_eye_valid, left_eye[:, 1], tf.where(neck_valid, neck[:, 1], tf.zeros_like(neck[:, 1])))
        eye_valid = (left_eye_valid | neck_valid) & mask

        start_x = neck_x - (3.0 * head_metric)
        start_y = eye_y + (0.5 * head_metric)
        end_x = neck_x + (3.0 * head_metric)
        end_y = start_y - (6.0 * head_metric)

        box_valid = head_metric_valid & neck_x_valid & eye_valid
        start_x = self._forward_fill(start_x, box_valid)
        start_y = self._forward_fill(start_y, box_valid)
        end_x = self._forward_fill(end_x, box_valid)
        end_y = self._forward_fill(end_y, box_valid)

        width = tf.maximum(end_x - start_x, tf.constant(1e-6, dtype=pose_dtype))
        height = tf.maximum(start_y - end_y, tf.constant(1e-6, dtype=pose_dtype))

        normalized_x = (pose[:, :, 0] - start_x[:, tf.newaxis]) / width[:, tf.newaxis]
        normalized_y = (pose[:, :, 1] - end_y[:, tf.newaxis]) / height[:, tf.newaxis]
        normalized = tf.stack([normalized_x, normalized_y], axis=-1)

        joint_presence = self._joint_presence_mask(pose)
        frame_mask = mask[:, tf.newaxis]
        body_mask = self.body_joint_mask[tf.newaxis, :]
        update_mask_bool = body_mask & joint_presence & frame_mask
        update_mask = tf.broadcast_to(
            tf.cast(update_mask_bool[..., tf.newaxis], pose_dtype),
            tf.shape(pose),
        )
        normalized_pose = pose * (1.0 - update_mask) + normalized * update_mask
        return normalized_pose

    def _normalize_hands(self, pose: tf.Tensor, mask: tf.Tensor, joint_indices: tf.Tensor) -> tf.Tensor:
        pose_dtype = pose.dtype
        mask = tf.cast(mask, tf.bool)
        hand_coords = tf.gather(pose, joint_indices, axis=1)
        hand_presence = self._joint_presence_mask(hand_coords)
        valid_frames = mask[:, tf.newaxis] & hand_presence
        if_valid = tf.reduce_any(valid_frames)
        def no_op():
            return pose

        def normalize_frames():
            large_value = tf.constant(1e6, dtype=pose_dtype)
            small_value = tf.constant(-1e6, dtype=pose_dtype)
            coords_for_min = tf.where(
                valid_frames[..., tf.newaxis],
                hand_coords,
                tf.ones_like(hand_coords) * large_value,
            )
            coords_for_max = tf.where(
                valid_frames[..., tf.newaxis],
                hand_coords,
                tf.ones_like(hand_coords) * small_value,
            )
            min_vals = tf.reduce_min(coords_for_min, axis=1)
            max_vals = tf.reduce_max(coords_for_max, axis=1)

            width = max_vals[:, 0] - min_vals[:, 0]
            height = max_vals[:, 1] - min_vals[:, 1]
            width = tf.maximum(width, tf.constant(1e-6, dtype=pose_dtype))
            height = tf.maximum(height, tf.constant(1e-6, dtype=pose_dtype))
            width_greater = width > height

            delta_x_width = 0.1 * width
            delta_y_width = delta_x_width + 0.5 * (width - height)
            delta_y_height = 0.1 * height
            delta_x_height = delta_y_height + 0.5 * (height - width)

            delta_x = tf.where(width_greater, delta_x_width, delta_x_height)
            delta_y = tf.where(width_greater, delta_y_width, delta_y_height)

            start_x = min_vals[:, 0] - delta_x
            start_y = min_vals[:, 1] - delta_y
            end_x = max_vals[:, 0] + delta_x
            end_y = max_vals[:, 1] + delta_y

            width_box = tf.maximum(end_x - start_x, tf.constant(1e-6, dtype=pose_dtype))
            height_box = tf.maximum(start_y - end_y, tf.constant(1e-6, dtype=pose_dtype))

            norm_x = (hand_coords[:, :, 0] - start_x[:, tf.newaxis]) / width_box[:, tf.newaxis]
            norm_y = (hand_coords[:, :, 1] - end_y[:, tf.newaxis]) / height_box[:, tf.newaxis]
            normalized_hand = tf.stack([norm_x, norm_y], axis=-1)
            valid_mask = valid_frames[..., tf.newaxis]
            normalized_hand = tf.where(valid_mask, normalized_hand, hand_coords)

            seq_len = tf.shape(pose)[0]
            joint_count = tf.shape(joint_indices)[0]
            frame_ids = tf.tile(tf.range(seq_len)[:, tf.newaxis], [1, joint_count])
            joint_ids = tf.tile(joint_indices[tf.newaxis, :], [seq_len, 1])
            scatter_indices = tf.stack(
                [tf.reshape(frame_ids, [-1]), tf.reshape(joint_ids, [-1])],
                axis=1,
            )
            updates = tf.reshape(normalized_hand, (-1, 2))
            return tf.tensor_scatter_nd_update(pose, scatter_indices, updates)

        return tf.cond(if_valid, normalize_frames, no_op)

    @staticmethod
    def _joint_valid_mask(joint_coordinates: tf.Tensor) -> tf.Tensor:
        return tf.reduce_any(tf.not_equal(joint_coordinates, 0.0), axis=-1)

    @staticmethod
    def _joint_presence_mask(pose: tf.Tensor) -> tf.Tensor:
        return tf.reduce_any(tf.not_equal(pose, 0.0), axis=-1)

    @staticmethod
    def _forward_fill(values: tf.Tensor, valid_mask: tf.Tensor) -> tf.Tensor:
        dtype = values.dtype
        valid_mask = tf.cast(valid_mask, tf.bool)
        length = tf.shape(values)[0]
        initial_value = tf.zeros_like(values[0])
        output = tf.TensorArray(dtype, size=length)

        def condition(i, *_):
            return i < length

        def body(i, last_value, ta):
            current_value = values[i]
            is_valid = valid_mask[i]
            new_value = tf.where(is_valid, current_value, last_value)
            ta = ta.write(i, new_value)
            return i + 1, new_value, ta

        _, _, output = tf.while_loop(
            condition,
            body,
            loop_vars=(tf.constant(0), initial_value, output),
        )
        return output.stack()


if __name__ == "__main__":
    pass
