from typing import Optional, Sequence

import tensorflow as tf


@tf.keras.utils.register_keras_serializable(package="spoter")
class TemporalConvBlock(tf.keras.layers.Layer):
    """Quantization-friendly temporal convolution block with separable convolutions."""

    def __init__(
        self,
        channels: int,
        kernel_size: int = 5,
        dilation_rate: int = 1,
        dropout: float = 0.1,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.channels = channels
        self.kernel_size = kernel_size
        self.dilation_rate = dilation_rate
        self.dropout_rate = dropout

        self.conv1 = tf.keras.layers.SeparableConv1D(
            filters=channels,
            kernel_size=kernel_size,
            dilation_rate=dilation_rate,
            padding="same",
            depth_multiplier=1,
            activation=None,
        )
        self.bn1 = tf.keras.layers.BatchNormalization()
        self.conv2 = tf.keras.layers.SeparableConv1D(
            filters=channels,
            kernel_size=kernel_size,
            dilation_rate=dilation_rate,
            padding="same",
            depth_multiplier=1,
            activation=None,
        )
        self.bn2 = tf.keras.layers.BatchNormalization()
        self.activation = tf.keras.layers.ReLU()
        self.dropout = tf.keras.layers.Dropout(dropout)
        self.residual_projection = tf.keras.layers.Conv1D(
            filters=channels,
            kernel_size=1,
            padding="same",
        )

    def call(self, inputs: tf.Tensor, training: bool = False) -> tf.Tensor:
        residual = self.residual_projection(inputs)
        x = self.conv1(inputs)
        x = self.bn1(x, training=training)
        x = self.activation(x)
        x = self.dropout(x, training=training)
        x = self.conv2(x)
        x = self.bn2(x, training=training)
        x = self.activation(x)
        x = self.dropout(x, training=training)
        return x + residual

    def get_config(self):
        config = super().get_config()
        config.update(
            {
                "channels": self.channels,
                "kernel_size": self.kernel_size,
                "dilation_rate": self.dilation_rate,
                "dropout": self.dropout_rate,
            }
        )
        return config


@tf.keras.utils.register_keras_serializable(package="spoter")
class SPOTER(tf.keras.Model):
    """Temporal Convolutional Network architecture for pose-based sign recognition."""

    def __init__(
        self,
        num_classes: int,
        tcn_channels: Sequence[int],
        dropout: float = 0.1,
        activation: str = "relu",
        kernel_size: int = 5,
        dilation_base: int = 2,
        apply_fake_quant: bool = False,
        quantization_bits: int = 8,
        **kwargs,
    ):
        super().__init__(**kwargs)
        if dilation_base < 1:
            raise ValueError("dilation_base must be a positive integer.")
        if not tcn_channels:
            raise ValueError("tcn_channels must contain at least one block width.")

        self.num_classes = num_classes
        self.tcn_channels = list(tcn_channels)
        self.dropout_rate = dropout
        self.activation_name = activation
        self.kernel_size = kernel_size
        self.dilation_base = dilation_base
        self.apply_fake_quant = apply_fake_quant
        self.quantization_bits = int(quantization_bits)

        self.input_projection = tf.keras.layers.Dense(
            self.tcn_channels[0],
            activation=None,
        )
        self.input_activation = tf.keras.layers.Activation(activation)

        self.temporal_blocks = [
            TemporalConvBlock(
                channels=channels,
                kernel_size=kernel_size,
                dilation_rate=(dilation_base ** layer_index) if dilation_base > 1 else 1,
                dropout=dropout,
            )
            for layer_index, channels in enumerate(self.tcn_channels)
        ]
        self.global_pool = tf.keras.layers.GlobalAveragePooling1D()
        self.classifier = tf.keras.layers.Dense(num_classes)

    def _maybe_fake_quant(self, tensor: tf.Tensor) -> tf.Tensor:
        if not self.apply_fake_quant:
            return tensor
        min_val = tf.reduce_min(tensor)
        max_val = tf.reduce_max(tensor)
        return tf.quantization.fake_quant_with_min_max_vars(
            tensor,
            min=min_val,
            max=max_val,
            num_bits=self.quantization_bits,
        )

    def call(self, inputs: tf.Tensor, training: bool = False, mask: Optional[tf.Tensor] = None) -> tf.Tensor:
        inputs = tf.cast(inputs, tf.float32)
        dynamic_shape = tf.shape(inputs)
        batch_size = dynamic_shape[0]
        sequence_length = dynamic_shape[1]
        per_frame_dim = dynamic_shape[2] * dynamic_shape[3]
        flattened_shape = tf.stack([batch_size, sequence_length, per_frame_dim])
        flattened_inputs = tf.reshape(inputs, flattened_shape)
        flattened_inputs = self._maybe_fake_quant(flattened_inputs)

        x = self.input_projection(flattened_inputs)
        x = self.input_activation(x)
        x = self._maybe_fake_quant(x)

        for block in self.temporal_blocks:
            x = block(x, training=training)
            x = self._maybe_fake_quant(x)

        if mask is not None:
            float_mask = tf.cast(mask[:, :, tf.newaxis], x.dtype)
            x = x * float_mask
            summed = tf.reduce_sum(x, axis=1)
            valid_counts = tf.reduce_sum(float_mask, axis=1)
            pooled = tf.math.divide_no_nan(summed, valid_counts)
        else:
            pooled = self.global_pool(x)
        pooled = self._maybe_fake_quant(pooled)

        logits = self.classifier(pooled)
        logits = tf.reshape(logits, (batch_size, 1, self.num_classes))
        logits = self._maybe_fake_quant(logits)
        return logits

    def get_config(self):
        config = super().get_config()
        config.update(
            {
                "num_classes": self.num_classes,
                "tcn_channels": self.tcn_channels,
                "dropout": self.dropout_rate,
                "activation": self.activation_name,
                "kernel_size": self.kernel_size,
                "dilation_base": self.dilation_base,
                "apply_fake_quant": self.apply_fake_quant,
                "quantization_bits": self.quantization_bits,
            }
        )
        return config
