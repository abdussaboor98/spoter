"""Utility helpers for training and evaluating the TensorFlow SPOTER model."""

import logging
from typing import Optional

import numpy as np
import tensorflow as tf
from tqdm.auto import tqdm


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
            current_lr = self.optimizer.learning_rate.numpy()
            updated_lr = max(current_lr * self.factor, self.min_lr)
            self.optimizer.learning_rate.assign(tf.cast(updated_lr, tf.float32))
            self.waiting_epochs = 0

    def get_state(self):
        return {
            "best_metric": float(self.best_metric) if np.isfinite(self.best_metric) else None,
            "waiting_epochs": int(self.waiting_epochs),
        }

    def set_state(self, state):
        if not state:
            return
        best_metric = state.get("best_metric")
        self.best_metric = float(best_metric) if best_metric is not None else np.inf
        self.waiting_epochs = int(state.get("waiting_epochs", 0))


def train_single_epoch(model: tf.keras.Model, dataset: tf.data.Dataset, loss_fn, optimizer,
                       scheduler: Optional[PlateauLearningRateScheduler] = None,
                       show_sample_progress: bool = False, epoch_description: Optional[str] = None,
                       steps_per_epoch: Optional[int] = None):
    correct_predictions, total_predictions = 0, 0
    cumulative_loss = 0.0

    progress_iterator = dataset
    batch_bar = None

    if show_sample_progress:
        batch_bar = tqdm(
            dataset,
            desc=epoch_description or "Training",
            leave=False,
            total=steps_per_epoch,
            unit="batch",
        )
        progress_iterator = batch_bar

    for batch_inputs, batch_labels in progress_iterator:
        mask = batch_inputs["mask"]
        features = batch_inputs["pose"]
        with tf.GradientTape() as tape:
            logits = model(features, training=True, attention_mask=mask)
            logits = tf.squeeze(logits, axis=1)
            loss = loss_fn(batch_labels, logits)

        gradients = tape.gradient(loss, model.trainable_variables)
        optimizer.apply_gradients(zip(gradients, model.trainable_variables))

        batch_size = int(batch_labels.shape[0])
        cumulative_loss += float(loss.numpy()) * batch_size

        predictions = tf.argmax(logits, axis=-1, output_type=tf.int32)
        correct_predictions += int(tf.reduce_sum(tf.cast(tf.equal(predictions, batch_labels), tf.int32)))
        total_predictions += batch_size

        if batch_bar is not None:
            running_loss = (cumulative_loss / total_predictions) if total_predictions else 0.0
            running_accuracy = (correct_predictions / total_predictions) if total_predictions else 0.0
            batch_bar.set_postfix({
                "loss": f"{running_loss:.4f}",
                "acc": f"{running_accuracy:.4f}",
            })

    mean_loss = (cumulative_loss / total_predictions) if total_predictions else 0.0

    if scheduler:
        scheduler.step(mean_loss)

    if batch_bar is not None:
        batch_bar.close()

    accuracy = (correct_predictions / total_predictions) if total_predictions else 0.0
    return mean_loss, correct_predictions, total_predictions, accuracy


def evaluate_model(model: tf.keras.Model, dataset: tf.data.Dataset, num_classes: int,
                   print_stats: bool = False):
    correct_predictions, total_predictions = 0, 0
    class_level_correct = np.zeros(num_classes, dtype=np.int32)
    class_level_total = np.zeros(num_classes, dtype=np.int32)

    for batch_inputs, batch_labels in dataset:
        mask = batch_inputs["mask"]
        features = batch_inputs["pose"]
        logits = model(features, training=False, attention_mask=mask)
        logits = tf.squeeze(logits, axis=1)
        predictions = tf.argmax(logits, axis=-1, output_type=tf.int32)

        predicted_array = predictions.numpy()
        labels_array = batch_labels.numpy()

        correct_predictions += int(np.sum(predicted_array == labels_array))
        total_predictions += labels_array.shape[0]

        for target_label, predicted_label in zip(labels_array, predicted_array):
            class_level_total[int(target_label)] += 1
            if int(predicted_label) == int(target_label):
                class_level_correct[int(target_label)] += 1

    if print_stats:
        label_statistics = {
            index: class_level_correct[index] / class_level_total[index]
            for index in range(num_classes)
            if class_level_total[index] > 0
        }
        print("Label accuracies statistics:")
        print(str(label_statistics) + "\n")
        logging.info("Label accuracies statistics:")
        logging.info(str(label_statistics) + "\n")

    accuracy = (correct_predictions / total_predictions) if total_predictions else 0.0
    return correct_predictions, total_predictions, accuracy


def evaluate_top_k_accuracy(model: tf.keras.Model, dataset: tf.data.Dataset, k: int = 5):
    correct_predictions, total_predictions = 0, 0

    for batch_inputs, batch_labels in dataset:
        mask = batch_inputs["mask"]
        features = batch_inputs["pose"]
        logits = model(features, training=False, attention_mask=mask)
        logits = tf.squeeze(logits, axis=1)
        top_k_results = tf.math.top_k(logits, k=k)

        top_k_indices = top_k_results.indices.numpy()
        labels_array = batch_labels.numpy()

        for row_index, target_label in enumerate(labels_array):
            if int(target_label) in top_k_indices[row_index]:
                correct_predictions += 1
            total_predictions += 1

    accuracy = (correct_predictions / total_predictions) if total_predictions else 0.0
    return correct_predictions, total_predictions, accuracy
