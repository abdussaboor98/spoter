from typing import Optional

import tensorflow as tf


@tf.keras.utils.register_keras_serializable(package="spoter")
class SpoterEncoderLayer(tf.keras.layers.Layer):
    def __init__(self, d_model: int, num_heads: int, dim_feedforward: int = 2048,
                 dropout: float = 0.1, activation: str = "relu", **kwargs):
        super().__init__(**kwargs)
        if d_model % num_heads != 0:
            raise ValueError("d_model must be divisible by num_heads for multi-head attention.")
        self.d_model = d_model
        self.num_heads = num_heads
        self.dim_feedforward = dim_feedforward
        self.dropout_rate = dropout
        self.activation_name = activation

        self.self_attention = tf.keras.layers.MultiHeadAttention(
            num_heads=num_heads,
            key_dim=d_model // num_heads,
            dropout=dropout,
        )
        self.dropout_after_attention = tf.keras.layers.Dropout(dropout)
        self.layer_norm_after_attention = tf.keras.layers.LayerNormalization(epsilon=1e-6)

        self.feedforward_projection = tf.keras.layers.Dense(dim_feedforward, activation=tf.keras.activations.get(activation))
        self.feedforward_dropout = tf.keras.layers.Dropout(dropout)
        self.feedforward_output_projection = tf.keras.layers.Dense(d_model)
        self.dropout_after_feedforward = tf.keras.layers.Dropout(dropout)
        self.layer_norm_after_feedforward = tf.keras.layers.LayerNormalization(epsilon=1e-6)

    def call(self, encoder_input: tf.Tensor, training: bool = False,
             mask: Optional[tf.Tensor] = None) -> tf.Tensor:
        attention_output = self.self_attention(
            encoder_input,
            encoder_input,
            encoder_input,
            attention_mask=mask,
            training=training,
        )
        encoder_input = self.layer_norm_after_attention(
            encoder_input + self.dropout_after_attention(attention_output, training=training)
        )

        feedforward_output = self.feedforward_projection(encoder_input)
        feedforward_output = self.feedforward_dropout(feedforward_output, training=training)
        feedforward_output = self.feedforward_output_projection(feedforward_output)
        encoder_input = self.layer_norm_after_feedforward(
            encoder_input + self.dropout_after_feedforward(feedforward_output, training=training)
        )
        return encoder_input

    def get_config(self):
        config = super().get_config()
        config.update({
            "d_model": self.d_model,
            "num_heads": self.num_heads,
            "dim_feedforward": self.dim_feedforward,
            "dropout": self.dropout_rate,
            "activation": self.activation_name,
        })
        return config


@tf.keras.utils.register_keras_serializable(package="spoter")
class SpoterDecoderLayer(tf.keras.layers.Layer):
    def __init__(self, d_model: int, num_heads: int, dim_feedforward: int = 2048,
                 dropout: float = 0.1, activation: str = "relu", **kwargs):
        super().__init__(**kwargs)
        if d_model % num_heads != 0:
            raise ValueError("d_model must be divisible by num_heads for multi-head attention.")
        self.d_model = d_model
        self.num_heads = num_heads
        self.dim_feedforward = dim_feedforward
        self.dropout_rate = dropout
        self.activation_name = activation

        self.initial_dropout = tf.keras.layers.Dropout(dropout)
        self.initial_norm = tf.keras.layers.LayerNormalization(epsilon=1e-6)

        self.cross_attention = tf.keras.layers.MultiHeadAttention(
            num_heads=num_heads,
            key_dim=d_model // num_heads,
            dropout=dropout,
        )
        self.dropout_after_cross_attention = tf.keras.layers.Dropout(dropout)
        self.layer_norm_after_cross_attention = tf.keras.layers.LayerNormalization(epsilon=1e-6)

        self.feedforward_projection = tf.keras.layers.Dense(dim_feedforward, activation=tf.keras.activations.get(activation))
        self.feedforward_dropout = tf.keras.layers.Dropout(dropout)
        self.feedforward_output_projection = tf.keras.layers.Dense(d_model)
        self.dropout_after_feedforward = tf.keras.layers.Dropout(dropout)
        self.layer_norm_after_feedforward = tf.keras.layers.LayerNormalization(epsilon=1e-6)

    def call(self, decoder_query: tf.Tensor, memory: tf.Tensor, training: bool = False,
             memory_mask: Optional[tf.Tensor] = None) -> tf.Tensor:
        decoder_query = self.initial_norm(decoder_query + self.initial_dropout(decoder_query, training=training))
        attention_output = self.cross_attention(decoder_query, memory, memory, attention_mask=memory_mask, training=training)
        decoder_query = self.layer_norm_after_cross_attention(
            decoder_query + self.dropout_after_cross_attention(attention_output, training=training)
        )

        feedforward_output = self.feedforward_projection(decoder_query)
        feedforward_output = self.feedforward_dropout(feedforward_output, training=training)
        feedforward_output = self.feedforward_output_projection(feedforward_output)
        decoder_query = self.layer_norm_after_feedforward(
            decoder_query + self.dropout_after_feedforward(feedforward_output, training=training)
        )
        return decoder_query

    def get_config(self):
        config = super().get_config()
        config.update({
            "d_model": self.d_model,
            "num_heads": self.num_heads,
            "dim_feedforward": self.dim_feedforward,
            "dropout": self.dropout_rate,
            "activation": self.activation_name,
        })
        return config


@tf.keras.utils.register_keras_serializable(package="spoter")
class SpoterEncoderStack(tf.keras.layers.Layer):
    def __init__(self, num_layers: int, d_model: int, num_heads: int,
                 dim_feedforward: int = 2048, dropout: float = 0.1,
                 activation: str = "relu", **kwargs):
        super().__init__(**kwargs)
        self.num_layers = num_layers
        self.d_model = d_model
        self.num_heads = num_heads
        self.dim_feedforward = dim_feedforward
        self.dropout_rate = dropout
        self.activation_name = activation

        self.layers = [
            SpoterEncoderLayer(d_model, num_heads, dim_feedforward, dropout, activation)
            for _ in range(num_layers)
        ]

    def call(self, encoder_input: tf.Tensor, training: bool = False,
             attention_mask: Optional[tf.Tensor] = None) -> tf.Tensor:
        output = encoder_input
        for layer in self.layers:
            output = layer(output, training=training, mask=attention_mask)
        return output

    def get_config(self):
        config = super().get_config()
        config.update({
            "num_layers": self.num_layers,
            "d_model": self.d_model,
            "num_heads": self.num_heads,
            "dim_feedforward": self.dim_feedforward,
            "dropout": self.dropout_rate,
            "activation": self.activation_name,
        })
        return config


@tf.keras.utils.register_keras_serializable(package="spoter")
class SpoterDecoderStack(tf.keras.layers.Layer):
    def __init__(self, num_layers: int, d_model: int, num_heads: int,
                 dim_feedforward: int = 2048, dropout: float = 0.1,
                 activation: str = "relu", **kwargs):
        super().__init__(**kwargs)
        self.num_layers = num_layers
        self.d_model = d_model
        self.num_heads = num_heads
        self.dim_feedforward = dim_feedforward
        self.dropout_rate = dropout
        self.activation_name = activation

        self.layers = [
            SpoterDecoderLayer(d_model, num_heads, dim_feedforward, dropout, activation)
            for _ in range(num_layers)
        ]

    def call(self, decoder_query: tf.Tensor, memory: tf.Tensor, training: bool = False,
             memory_mask: Optional[tf.Tensor] = None) -> tf.Tensor:
        output = decoder_query
        for layer in self.layers:
            output = layer(output, memory, training=training, memory_mask=memory_mask)
        return output

    def get_config(self):
        config = super().get_config()
        config.update({
            "num_layers": self.num_layers,
            "d_model": self.d_model,
            "num_heads": self.num_heads,
            "dim_feedforward": self.dim_feedforward,
            "dropout": self.dropout_rate,
            "activation": self.activation_name,
        })
        return config


@tf.keras.utils.register_keras_serializable(package="spoter")
class SPOTER(tf.keras.Model):
    """TensorFlow replica of the original PyTorch SPOTER architecture."""

    def __init__(self, num_classes: int, hidden_dim: int = 55,
                 num_heads: int = 9, num_encoder_layers: int = 6,
                 num_decoder_layers: int = 6, dim_feedforward: int = 2048,
                 dropout: float = 0.1, activation: str = "relu", **kwargs):
        super().__init__(**kwargs)
        if hidden_dim % num_heads != 0:
            raise ValueError("hidden_dim must be divisible by num_heads.")
        self.num_classes = num_classes
        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        self.num_encoder_layers = num_encoder_layers
        self.num_decoder_layers = num_decoder_layers
        self.dim_feedforward = dim_feedforward
        self.dropout_rate = dropout
        self.activation_name = activation

        self.positional_embedding = self.add_weight(
            name="positional_embedding",
            shape=(1, 1, hidden_dim),
            initializer=tf.keras.initializers.RandomUniform(),
            trainable=True,
        )
        self.classification_token = self.add_weight(
            name="class_query",
            shape=(1, hidden_dim),
            initializer=tf.keras.initializers.RandomUniform(),
            trainable=True,
        )

        self.encoder = SpoterEncoderStack(
            num_layers=num_encoder_layers,
            d_model=hidden_dim,
            num_heads=num_heads,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            activation=activation,
        )
        self.decoder = SpoterDecoderStack(
            num_layers=num_decoder_layers,
            d_model=hidden_dim,
            num_heads=num_heads,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            activation=activation,
        )
        self.classifier = tf.keras.layers.Dense(num_classes)

    def call(self, inputs: tf.Tensor, training: bool = False,
             attention_mask: Optional[tf.Tensor] = None) -> tf.Tensor:
        inputs = tf.cast(inputs, tf.float32)
        batch_size = tf.shape(inputs)[0]
        sequence_length = tf.shape(inputs)[1]
        flattened_inputs = tf.reshape(inputs, (batch_size, sequence_length, -1))

        frame_feature_dim = tf.shape(flattened_inputs)[-1]
        expected_dim = tf.constant(self.hidden_dim, dtype=frame_feature_dim.dtype)
        with tf.control_dependencies([
            tf.debugging.assert_equal(
                frame_feature_dim,
                expected_dim,
                message="Per-frame flattened dimension must equal hidden_dim.",
            ),
        ]):
            encoder_input = tf.reshape(flattened_inputs, (batch_size, sequence_length, self.hidden_dim))

        if attention_mask is not None:
            attention_mask = tf.cast(attention_mask, tf.bool)
            attention_mask = attention_mask[:, tf.newaxis, tf.newaxis, :]

        positional_encoding = tf.broadcast_to(self.positional_embedding,
                                              (batch_size, sequence_length, self.hidden_dim))
        encoder_memory = self.encoder(
            encoder_input + positional_encoding,
            training=training,
            attention_mask=attention_mask,
        )

        class_token = tf.broadcast_to(self.classification_token, (batch_size, self.hidden_dim))
        class_token = tf.expand_dims(class_token, axis=1)
        decoder_output = self.decoder(
            class_token,
            encoder_memory,
            training=training,
            memory_mask=attention_mask,
        )
        class_logits = self.classifier(decoder_output)
        return class_logits

    def get_config(self):
        config = super().get_config()
        config.update({
            "num_classes": self.num_classes,
            "hidden_dim": self.hidden_dim,
            "num_heads": self.num_heads,
            "num_encoder_layers": self.num_encoder_layers,
            "num_decoder_layers": self.num_decoder_layers,
            "dim_feedforward": self.dim_feedforward,
            "dropout": self.dropout_rate,
            "activation": self.activation_name,
        })
        return config
