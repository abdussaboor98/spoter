"""Utility helpers for training and evaluating the TensorFlow SPOTER model."""

import logging
from typing import Optional

import numpy as np
import tensorflow as tf


class PlateauLearningRateScheduler:
    """Manual implementation of ReduceLROnPlateau compatible with TF optimizers."""

    def __init__(self, optimizer: tf.keras.optimizers.Optimizer, factor: float = 0.1,
                 patience: int = 5, min_lr: float = 0.0):
        self.optimizer = optimizer
        self.factor = factor
        self.patience = patience
        self.min_lr = min_lr
        self.best_metric = np.inf
        self.waiting_epochs = 0

    def step(self, monitored_metric: float):
        if monitored_metric < self.best_metric:
            self.best_metric = monitored_metric
            self.waiting_epochs = 0
            return

        self.waiting_epochs += 1
        if self.waiting_epochs > self.patience:
            current_lr = float(tf.keras.backend.get_value(self.optimizer.learning_rate))
            updated_lr = max(current_lr * self.factor, self.min_lr)
            tf.keras.backend.set_value(self.optimizer.learning_rate, updated_lr)
            self.waiting_epochs = 0


def _ensure_batch_dimension(inputs: tf.Tensor) -> tf.Tensor:
    if tf.rank(inputs) == 3:
        return tf.expand_dims(inputs, axis=0)
    return inputs


def _ensure_label_batch(labels: tf.Tensor) -> tf.Tensor:
    if tf.rank(labels) == 0:
        return tf.expand_dims(labels, axis=0)
    return labels


def train_single_epoch(model: tf.keras.Model, dataset, loss_fn, optimizer,
                       scheduler: Optional[PlateauLearningRateScheduler] = None):
    correct_predictions, total_predictions = 0, 0
    cumulative_loss = 0.0

    sample_indices = np.random.permutation(len(dataset))

    for sample_index in sample_indices:
        features, labels = dataset[sample_index]
        batched_inputs = _ensure_batch_dimension(features)
        batched_labels = _ensure_label_batch(labels)

        with tf.GradientTape() as tape:
            logits = model(batched_inputs, training=True)
            logits = tf.squeeze(logits, axis=1)
            loss = loss_fn(batched_labels, logits)

        gradients = tape.gradient(loss, model.trainable_variables)
        optimizer.apply_gradients(zip(gradients, model.trainable_variables))

        cumulative_loss += float(loss.numpy())

        predictions = tf.argmax(logits, axis=-1, output_type=tf.int32)
        correct_predictions += int(tf.reduce_sum(tf.cast(tf.equal(predictions, batched_labels), tf.int32)))
        total_predictions += int(batched_labels.shape[0])

    if scheduler:
        scheduler.step(cumulative_loss / len(dataset))

    accuracy = (correct_predictions / total_predictions) if total_predictions else 0.0
    return cumulative_loss, correct_predictions, total_predictions, accuracy


def evaluate_model(model: tf.keras.Model, dataset, print_stats: bool = False):
    correct_predictions, total_predictions = 0, 0
    class_level_stats = {i: [0, 0] for i in range(101)}

    for sample_index in range(len(dataset)):
        features, labels = dataset[sample_index]
        batched_inputs = _ensure_batch_dimension(features)
        batched_labels = _ensure_label_batch(labels)

        logits = model(batched_inputs, training=False)
        logits = tf.squeeze(logits, axis=1)
        predictions = tf.argmax(logits, axis=-1, output_type=tf.int32)

        for predicted_label, target_label in zip(predictions.numpy(), batched_labels.numpy()):
            if int(predicted_label) == int(target_label):
                class_level_stats[int(target_label)][0] += 1
                correct_predictions += 1
            class_level_stats[int(target_label)][1] += 1
            total_predictions += 1

    if print_stats:
        label_statistics = {key: value[0] / value[1] for key, value in class_level_stats.items() if value[1] != 0}
        print("Label accuracies statistics:")
        print(str(label_statistics) + "\n")
        logging.info("Label accuracies statistics:")
        logging.info(str(label_statistics) + "\n")

    accuracy = (correct_predictions / total_predictions) if total_predictions else 0.0
    return correct_predictions, total_predictions, accuracy


def evaluate_top_k_accuracy(model: tf.keras.Model, dataset, k: int = 5):
    correct_predictions, total_predictions = 0, 0

    for sample_index in range(len(dataset)):
        features, labels = dataset[sample_index]
        batched_inputs = _ensure_batch_dimension(features)
        batched_labels = _ensure_label_batch(labels)

        logits = model(batched_inputs, training=False)
        logits = tf.squeeze(logits, axis=1)
        top_k_results = tf.math.top_k(logits, k=k)

        top_k_indices = top_k_results.indices.numpy()
        label_array = batched_labels.numpy()

        for row_index, target_label in enumerate(label_array):
            if int(target_label) in top_k_indices[row_index]:
                correct_predictions += 1
            total_predictions += 1

    accuracy = (correct_predictions / total_predictions) if total_predictions else 0.0
    return correct_predictions, total_predictions, accuracy
