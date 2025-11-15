from typing import Optional

import tensorflow as tf

try:
    import tensorflow_model_optimization as tfmot
except ImportError:  # pragma: no cover - optional dependency
    tfmot = None


@tf.keras.utils.register_keras_serializable(package="spoter")
class ScaledDotProductAttentionLayer(tf.keras.layers.Layer):
    """Scaled dot-product attention composed of quantizable primitives."""

    def __init__(self, dropout: float = 0.1, **kwargs):
        super().__init__(**kwargs)
        self.dropout_layer = tf.keras.layers.Dropout(dropout)

    def call(self, query: tf.Tensor, key: tf.Tensor, value: tf.Tensor,
             scale: float, attention_mask: Optional[tf.Tensor], training: bool) -> tf.Tensor:
        attention_scores = tf.matmul(query, key, transpose_b=True)
        attention_scores = attention_scores * scale
        if attention_mask is not None:
            attention_mask = tf.cast(attention_mask, attention_scores.dtype)
            attention_scores += attention_mask
        attention_weights = tf.nn.softmax(attention_scores, axis=-1)
        attention_weights = self.dropout_layer(attention_weights, training=training)
        return tf.matmul(attention_weights, value)

    def get_config(self):
        base_config = super().get_config()
        base_config.update({"dropout": self.dropout_layer.rate})
        return base_config


@tf.keras.utils.register_keras_serializable(package="spoter")
class QuantizationReadyMultiHeadAttention(tf.keras.layers.Layer):
    """
    Custom multi-head attention that uses tf.keras primitives supported by
    TensorFlow Lite quantization aware training.
    """

    def __init__(self, num_heads: int, key_dim: int, dropout: float = 0.1,
                 **kwargs):
        super().__init__(**kwargs)
        if key_dim * num_heads <= 0:
            raise ValueError("key_dim and num_heads must be positive integers.")
        self.num_heads = num_heads
        self.key_dim = key_dim
        self.dropout_rate = dropout
        self.scale = tf.cast(1.0 / tf.math.sqrt(tf.cast(key_dim, tf.float32)), tf.float32)

        if tfmot is not None:
            annotate = tfmot.quantization.keras.quantize_annotate_layer
            self.query_dense = annotate(tf.keras.layers.Dense(num_heads * key_dim, use_bias=True, name="query_projection"))
            self.key_dense = annotate(tf.keras.layers.Dense(num_heads * key_dim, use_bias=True, name="key_projection"))
            self.value_dense = annotate(tf.keras.layers.Dense(num_heads * key_dim, use_bias=True, name="value_projection"))
            self.out_dense = annotate(tf.keras.layers.Dense(num_heads * key_dim, use_bias=True, name="output_projection"))
        else:
            self.query_dense = tf.keras.layers.Dense(num_heads * key_dim, use_bias=True, name="query_projection")
            self.key_dense = tf.keras.layers.Dense(num_heads * key_dim, use_bias=True, name="key_projection")
            self.value_dense = tf.keras.layers.Dense(num_heads * key_dim, use_bias=True, name="value_projection")
            self.out_dense = tf.keras.layers.Dense(num_heads * key_dim, use_bias=True, name="output_projection")
        self.attention_layer = ScaledDotProductAttentionLayer(dropout)

    def _split_heads(self, projection: tf.Tensor) -> tf.Tensor:
        batch_size = tf.shape(projection)[0]
        sequence_length = tf.shape(projection)[1]
        reshaped = tf.reshape(projection, (batch_size, sequence_length, self.num_heads, self.key_dim))
        return tf.transpose(reshaped, perm=(0, 2, 1, 3))

    def _combine_heads(self, attention_output: tf.Tensor) -> tf.Tensor:
        attention_output = tf.transpose(attention_output, perm=(0, 2, 1, 3))
        batch_size = tf.shape(attention_output)[0]
        sequence_length = tf.shape(attention_output)[1]
        return tf.reshape(attention_output, (batch_size, sequence_length, self.num_heads * self.key_dim))

    def call(self, query: tf.Tensor, value: tf.Tensor, key: tf.Tensor,
             attention_mask: Optional[tf.Tensor] = None, training: bool = False) -> tf.Tensor:
        query_projection = self._split_heads(self.query_dense(query))
        key_projection = self._split_heads(self.key_dense(key))
        value_projection = self._split_heads(self.value_dense(value))

        attention_output = self.attention_layer(
            query_projection,
            key_projection,
            value_projection,
            self.scale,
            attention_mask,
            training,
        )
        attention_output = self._combine_heads(attention_output)
        return self.out_dense(attention_output)

    def get_config(self):
        base_config = super().get_config()
        base_config.update({
            "num_heads": self.num_heads,
            "key_dim": self.key_dim,
            "dropout": self.dropout_rate,
        })
        return base_config


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

        self.self_attention = QuantizationReadyMultiHeadAttention(num_heads=num_heads, key_dim=d_model // num_heads,
                                                                  dropout=dropout)
        self.dropout_after_attention = tf.keras.layers.Dropout(dropout)
        self.layer_norm_after_attention = tf.keras.layers.LayerNormalization(epsilon=1e-6)

        self.feedforward_projection = tf.keras.layers.Dense(dim_feedforward, activation=tf.keras.activations.get(activation))
        self.feedforward_dropout = tf.keras.layers.Dropout(dropout)
        self.feedforward_output_projection = tf.keras.layers.Dense(d_model)
        self.dropout_after_feedforward = tf.keras.layers.Dropout(dropout)
        self.layer_norm_after_feedforward = tf.keras.layers.LayerNormalization(epsilon=1e-6)

    def call(self, encoder_input: tf.Tensor, training: bool = False,
             mask: Optional[tf.Tensor] = None) -> tf.Tensor:
        attention_output = self.self_attention(encoder_input, encoder_input, encoder_input,
                                               attention_mask=mask, training=training)
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

        self.cross_attention = QuantizationReadyMultiHeadAttention(num_heads=num_heads, key_dim=d_model // num_heads,
                                                                   dropout=dropout)
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

    def call(self, encoder_input: tf.Tensor, training: bool = False) -> tf.Tensor:
        output = encoder_input
        for layer in self.layers:
            output = layer(output, training=training)
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

    def call(self, decoder_query: tf.Tensor, memory: tf.Tensor, training: bool = False) -> tf.Tensor:
        output = decoder_query
        for layer in self.layers:
            output = layer(output, memory, training=training)
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

    def call(self, inputs: tf.Tensor, training: bool = False) -> tf.Tensor:
        inputs = tf.cast(inputs, tf.float32)
        batch_size = tf.shape(inputs)[0]
        flattened_inputs = tf.reshape(inputs, (batch_size, 1, -1))

        hidden_dim = tf.shape(flattened_inputs)[-1]
        expected_dim = tf.constant(self.hidden_dim, dtype=hidden_dim.dtype)
        with tf.control_dependencies([
            tf.debugging.assert_equal(hidden_dim, expected_dim,
                                      message="Flattened input dimension must equal hidden_dim."),
        ]):
            encoder_input = tf.reshape(flattened_inputs, (batch_size, 1, self.hidden_dim))

        positional_encoding = tf.broadcast_to(self.positional_embedding, (batch_size, 1, self.hidden_dim))
        encoder_memory = self.encoder(encoder_input + positional_encoding, training=training)

        class_token = tf.broadcast_to(self.classification_token, (batch_size, self.hidden_dim))
        class_token = tf.expand_dims(class_token, axis=1)
        decoder_output = self.decoder(class_token, encoder_memory, training=training)
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


def create_quantization_aware_spoter(example_input_shape, **model_kwargs) -> tf.keras.Model:
    """Create a SPOTER model wrapped for TensorFlow Lite QAT."""
    if tfmot is None:
        raise ImportError("tensorflow_model_optimization must be installed for quantization aware training.")

    base_model = SPOTER(**model_kwargs)
    dummy_input = tf.zeros((1,) + tuple(example_input_shape), dtype=tf.float32)
    base_model(dummy_input, training=False)

    annotated_model = tfmot.quantization.keras.quantize_annotate_model(base_model)
    with tfmot.quantization.keras.quantize_scope({
        "QuantizationReadyMultiHeadAttention": QuantizationReadyMultiHeadAttention,
        "SpoterEncoderLayer": SpoterEncoderLayer,
        "SpoterDecoderLayer": SpoterDecoderLayer,
        "SpoterEncoderStack": SpoterEncoderStack,
        "SpoterDecoderStack": SpoterDecoderStack,
        "SPOTER": SPOTER,
    }):
        qat_model = tfmot.quantization.keras.quantize_apply(annotated_model)

    qat_model(dummy_input, training=False)
    return qat_model
