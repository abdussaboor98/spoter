from typing import Optional

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
    """Temporal Convolutional Network replacement for the original Transformer architecture."""

    def __init__(
        self,
        num_classes: int,
        hidden_dim: int = 55,
        num_heads: int = 9,  # Unused but kept for compatibility with existing CLI flags/configs.
        num_encoder_layers: int = 6,
        num_decoder_layers: int = 6,  # Not used but preserved for backwards compatibility.
        dim_feedforward: int = 2048,  # Represents internal projection size.
        dropout: float = 0.1,
        activation: str = "relu",
        kernel_size: int = 5,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.num_classes = num_classes
        self.hidden_dim = hidden_dim
        self.num_encoder_layers = num_encoder_layers
        self.dim_feedforward = dim_feedforward
        self.dropout_rate = dropout
        self.activation_name = activation
        self.kernel_size = kernel_size

        self.input_projection = tf.keras.layers.Dense(
            dim_feedforward,
            activation=None,
        )
        self.input_activation = tf.keras.layers.Activation(activation)

        self.temporal_blocks = [
            TemporalConvBlock(
                channels=dim_feedforward,
                kernel_size=kernel_size,
                dilation_rate=2 ** layer_index,
                dropout=dropout,
            )
            for layer_index in range(num_encoder_layers)
        ]
        self.global_pool = tf.keras.layers.GlobalAveragePooling1D()
        self.classifier = tf.keras.layers.Dense(num_classes)

    def call(self, inputs: tf.Tensor, training: bool = False, attention_mask: Optional[tf.Tensor] = None) -> tf.Tensor:
        inputs = tf.cast(inputs, tf.float32)
        batch_size = tf.shape(inputs)[0]
        sequence_length = tf.shape(inputs)[1]
        flattened_inputs = tf.reshape(inputs, (batch_size, sequence_length, -1))

        frame_feature_dim = tf.shape(flattened_inputs)[-1]
        expected_dim = tf.constant(self.hidden_dim, dtype=frame_feature_dim.dtype)
        with tf.control_dependencies(
            [
                tf.debugging.assert_equal(
                    frame_feature_dim,
                    expected_dim,
                    message="Per-frame flattened dimension must equal hidden_dim.",
                ),
            ]
        ):
            temporal_input = tf.reshape(flattened_inputs, (batch_size, sequence_length, self.hidden_dim))

        x = self.input_projection(temporal_input)
        x = self.input_activation(x)

        for block in self.temporal_blocks:
            x = block(x, training=training)

        pooled = self.global_pool(x)
        logits = self.classifier(pooled)
        logits = tf.reshape(logits, (batch_size, 1, self.num_classes))
        return logits

    def get_config(self):
        config = super().get_config()
        config.update(
            {
                "num_classes": self.num_classes,
                "hidden_dim": self.hidden_dim,
                "num_encoder_layers": self.num_encoder_layers,
                "dim_feedforward": self.dim_feedforward,
                "dropout": self.dropout_rate,
                "activation": self.activation_name,
                "kernel_size": self.kernel_size,
            }
        )
        return config
