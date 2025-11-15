
import logging
from typing import Optional

import numpy as np
import tensorflow as tf


class ReduceLROnPlateau:
    def __init__(self, optimizer: tf.keras.optimizers.Optimizer, factor: float = 0.1,
                 patience: int = 5, min_lr: float = 0.0):
        self.optimizer = optimizer
        self.factor = factor
        self.patience = patience
        self.min_lr = min_lr
        self.best = np.inf
        self.wait = 0

    def step(self, metric: float):
        if metric < self.best:
            self.best = metric
            self.wait = 0
        else:
            self.wait += 1
            if self.wait > self.patience:
                current_lr = float(tf.keras.backend.get_value(self.optimizer.learning_rate))
                new_lr = max(current_lr * self.factor, self.min_lr)
                tf.keras.backend.set_value(self.optimizer.learning_rate, new_lr)
                self.wait = 0


def _expand_batch(inputs: tf.Tensor) -> tf.Tensor:
    if tf.rank(inputs) == 3:
        return tf.expand_dims(inputs, axis=0)
    return inputs


def _expand_labels(labels: tf.Tensor) -> tf.Tensor:
    if tf.rank(labels) == 0:
        return tf.expand_dims(labels, axis=0)
    return labels


def train_epoch(model: tf.keras.Model, dataset, loss_fn, optimizer,
                scheduler: Optional[ReduceLROnPlateau] = None):

    pred_correct, pred_all = 0, 0
    running_loss = 0.0

    indices = np.random.permutation(len(dataset))

    for idx in indices:
        inputs, labels = dataset[idx]
        inputs = _expand_batch(inputs)
        labels = _expand_labels(labels)

        with tf.GradientTape() as tape:
            outputs = model(inputs, training=True)
            logits = tf.squeeze(outputs, axis=1)
            loss = loss_fn(labels, logits)

        gradients = tape.gradient(loss, model.trainable_variables)
        optimizer.apply_gradients(zip(gradients, model.trainable_variables))

        running_loss += float(loss.numpy())

        predictions = tf.argmax(logits, axis=-1, output_type=tf.int32)
        pred_correct += int(tf.reduce_sum(tf.cast(tf.equal(predictions, labels), tf.int32)))
        pred_all += int(labels.shape[0])

    if scheduler:
        scheduler.step(running_loss / len(dataset))

    return running_loss, pred_correct, pred_all, (pred_correct / pred_all if pred_all else 0)


def evaluate(model: tf.keras.Model, dataset, print_stats: bool = False):

    pred_correct, pred_all = 0, 0
    stats = {i: [0, 0] for i in range(101)}

    for idx in range(len(dataset)):
        inputs, labels = dataset[idx]
        inputs = _expand_batch(inputs)
        labels = _expand_labels(labels)

        outputs = model(inputs, training=False)
        logits = tf.squeeze(outputs, axis=1)
        predictions = tf.argmax(logits, axis=-1, output_type=tf.int32)

        for pred, label in zip(predictions.numpy(), labels.numpy()):
            if int(pred) == int(label):
                stats[int(label)][0] += 1
                pred_correct += 1
            stats[int(label)][1] += 1
            pred_all += 1

    if print_stats:
        label_stats = {key: value[0] / value[1] for key, value in stats.items() if value[1] != 0}
        print("Label accuracies statistics:")
        print(str(label_stats) + "\n")
        logging.info("Label accuracies statistics:")
        logging.info(str(label_stats) + "\n")

    return pred_correct, pred_all, (pred_correct / pred_all if pred_all else 0)


def evaluate_top_k(model: tf.keras.Model, dataset, k: int = 5):

    pred_correct, pred_all = 0, 0

    for idx in range(len(dataset)):
        inputs, labels = dataset[idx]
        inputs = _expand_batch(inputs)
        labels = _expand_labels(labels)

        outputs = model(inputs, training=False)
        logits = tf.squeeze(outputs, axis=1)
        topk = tf.math.top_k(logits, k=k)

        topk_indices = topk.indices.numpy()
        label_array = labels.numpy()

        for row_idx, label in enumerate(label_array):
            if int(label) in topk_indices[row_idx]:
                pred_correct += 1
            pred_all += 1

    return pred_correct, pred_all, (pred_correct / pred_all if pred_all else 0)
