from typing import Optional

import tensorflow as tf

try:
    import tensorflow_model_optimization as tfmot
except ImportError:  # pragma: no cover - optional dependency
    tfmot = None


@tf.keras.utils.register_keras_serializable(package="spoter")
class _ScaledDotProductAttention(tf.keras.layers.Layer):
    """Scaled dot-product attention composed of quantizable primitives."""

    def __init__(self, dropout: float = 0.1, **kwargs):
        super().__init__(**kwargs)
        self.dropout = tf.keras.layers.Dropout(dropout)

    def call(self, query: tf.Tensor, key: tf.Tensor, value: tf.Tensor,
             scale: float, attention_mask: Optional[tf.Tensor], training: bool) -> tf.Tensor:
        scores = tf.matmul(query, key, transpose_b=True)
        scores = scores * scale
        if attention_mask is not None:
            attention_mask = tf.cast(attention_mask, scores.dtype)
            scores += attention_mask
        weights = tf.nn.softmax(scores, axis=-1)
        weights = self.dropout(weights, training=training)
        return tf.matmul(weights, value)

    def get_config(self):
        base_config = super().get_config()
        base_config.update({"dropout": self.dropout.rate})
        return base_config


@tf.keras.utils.register_keras_serializable(package="spoter")
class QuantizableMultiHeadAttention(tf.keras.layers.Layer):
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
            self.query_dense = annotate(tf.keras.layers.Dense(num_heads * key_dim, use_bias=True, name="q"))
            self.key_dense = annotate(tf.keras.layers.Dense(num_heads * key_dim, use_bias=True, name="k"))
            self.value_dense = annotate(tf.keras.layers.Dense(num_heads * key_dim, use_bias=True, name="v"))
            self.out_dense = annotate(tf.keras.layers.Dense(num_heads * key_dim, use_bias=True, name="out"))
        else:
            self.query_dense = tf.keras.layers.Dense(num_heads * key_dim, use_bias=True, name="q")
            self.key_dense = tf.keras.layers.Dense(num_heads * key_dim, use_bias=True, name="k")
            self.value_dense = tf.keras.layers.Dense(num_heads * key_dim, use_bias=True, name="v")
            self.out_dense = tf.keras.layers.Dense(num_heads * key_dim, use_bias=True, name="out")
        self.attention = _ScaledDotProductAttention(dropout)

    def _split_heads(self, tensor: tf.Tensor) -> tf.Tensor:
        batch_size = tf.shape(tensor)[0]
        seq_len = tf.shape(tensor)[1]
        new_shape = (batch_size, seq_len, self.num_heads, self.key_dim)
        tensor = tf.reshape(tensor, new_shape)
        return tf.transpose(tensor, perm=(0, 2, 1, 3))

    def _combine_heads(self, tensor: tf.Tensor) -> tf.Tensor:
        tensor = tf.transpose(tensor, perm=(0, 2, 1, 3))
        batch_size = tf.shape(tensor)[0]
        seq_len = tf.shape(tensor)[1]
        return tf.reshape(tensor, (batch_size, seq_len, self.num_heads * self.key_dim))

    def call(self, query: tf.Tensor, value: tf.Tensor, key: tf.Tensor,
             attention_mask: Optional[tf.Tensor] = None, training: bool = False) -> tf.Tensor:
        q = self._split_heads(self.query_dense(query))
        k = self._split_heads(self.key_dense(key))
        v = self._split_heads(self.value_dense(value))

        attn = self.attention(q, k, v, self.scale, attention_mask, training)
        attn = self._combine_heads(attn)
        return self.out_dense(attn)

    def get_config(self):
        base_config = super().get_config()
        base_config.update({
            "num_heads": self.num_heads,
            "key_dim": self.key_dim,
            "dropout": self.dropout_rate,
        })
        return base_config


@tf.keras.utils.register_keras_serializable(package="spoter")
class TransformerEncoderLayer(tf.keras.layers.Layer):
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

        self.self_attn = QuantizableMultiHeadAttention(num_heads=num_heads, key_dim=d_model // num_heads,
                                                       dropout=dropout)
        self.dropout1 = tf.keras.layers.Dropout(dropout)
        self.norm1 = tf.keras.layers.LayerNormalization(epsilon=1e-6)

        self.linear1 = tf.keras.layers.Dense(dim_feedforward, activation=tf.keras.activations.get(activation))
        self.dropout_ff = tf.keras.layers.Dropout(dropout)
        self.linear2 = tf.keras.layers.Dense(d_model)
        self.dropout2 = tf.keras.layers.Dropout(dropout)
        self.norm2 = tf.keras.layers.LayerNormalization(epsilon=1e-6)

    def call(self, src: tf.Tensor, training: bool = False,
             mask: Optional[tf.Tensor] = None) -> tf.Tensor:
        attn_output = self.self_attn(src, src, src, attention_mask=mask, training=training)
        src = self.norm1(src + self.dropout1(attn_output, training=training))

        ff_output = self.linear1(src)
        ff_output = self.dropout_ff(ff_output, training=training)
        ff_output = self.linear2(ff_output)
        src = self.norm2(src + self.dropout2(ff_output, training=training))
        return src

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
class TransformerDecoderLayer(tf.keras.layers.Layer):
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

        self.cross_attn = QuantizableMultiHeadAttention(num_heads=num_heads, key_dim=d_model // num_heads,
                                                        dropout=dropout)
        self.dropout1 = tf.keras.layers.Dropout(dropout)
        self.norm1 = tf.keras.layers.LayerNormalization(epsilon=1e-6)

        self.linear1 = tf.keras.layers.Dense(dim_feedforward, activation=tf.keras.activations.get(activation))
        self.dropout_ff = tf.keras.layers.Dropout(dropout)
        self.linear2 = tf.keras.layers.Dense(d_model)
        self.dropout2 = tf.keras.layers.Dropout(dropout)
        self.norm2 = tf.keras.layers.LayerNormalization(epsilon=1e-6)

    def call(self, tgt: tf.Tensor, memory: tf.Tensor, training: bool = False,
             memory_mask: Optional[tf.Tensor] = None) -> tf.Tensor:
        tgt = self.initial_norm(tgt + self.initial_dropout(tgt, training=training))
        attn_output = self.cross_attn(tgt, memory, memory, attention_mask=memory_mask, training=training)
        tgt = self.norm1(tgt + self.dropout1(attn_output, training=training))

        ff_output = self.linear1(tgt)
        ff_output = self.dropout_ff(ff_output, training=training)
        ff_output = self.linear2(ff_output)
        tgt = self.norm2(tgt + self.dropout2(ff_output, training=training))
        return tgt

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
class TransformerEncoder(tf.keras.layers.Layer):
    def __init__(self, num_layers: int, d_model: int, num_heads: int,
                 dim_feedforward: int = 2048, dropout: float = 0.1,
                 activation: str = "relu", **kwargs):
        super().__init__(**kwargs)
        self.num_layers = num_layers
        self.d_model = d_model
        self.num_heads = num_heads
        self.dim_feedforward = dim_feedforward
        self.dropout = dropout
        self.activation = activation

        self.layers = [
            TransformerEncoderLayer(d_model, num_heads, dim_feedforward, dropout, activation)
            for _ in range(num_layers)
        ]

    def call(self, src: tf.Tensor, training: bool = False) -> tf.Tensor:
        output = src
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
            "dropout": self.dropout,
            "activation": self.activation,
        })
        return config


@tf.keras.utils.register_keras_serializable(package="spoter")
class TransformerDecoder(tf.keras.layers.Layer):
    def __init__(self, num_layers: int, d_model: int, num_heads: int,
                 dim_feedforward: int = 2048, dropout: float = 0.1,
                 activation: str = "relu", **kwargs):
        super().__init__(**kwargs)
        self.num_layers = num_layers
        self.d_model = d_model
        self.num_heads = num_heads
        self.dim_feedforward = dim_feedforward
        self.dropout = dropout
        self.activation = activation

        self.layers = [
            TransformerDecoderLayer(d_model, num_heads, dim_feedforward, dropout, activation)
            for _ in range(num_layers)
        ]

    def call(self, tgt: tf.Tensor, memory: tf.Tensor, training: bool = False) -> tf.Tensor:
        output = tgt
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
            "dropout": self.dropout,
            "activation": self.activation,
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
        self.dropout = dropout
        self.activation = activation

        self.pos = self.add_weight(
            name="positional_embedding",
            shape=(1, 1, hidden_dim),
            initializer=tf.keras.initializers.RandomUniform(),
            trainable=True,
        )
        self.class_query = self.add_weight(
            name="class_query",
            shape=(1, hidden_dim),
            initializer=tf.keras.initializers.RandomUniform(),
            trainable=True,
        )

        self.encoder = TransformerEncoder(
            num_layers=num_encoder_layers,
            d_model=hidden_dim,
            num_heads=num_heads,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            activation=activation,
        )
        self.decoder = TransformerDecoder(
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
        flattened = tf.reshape(inputs, (batch_size, 1, -1))

        hidden_dim = tf.shape(flattened)[-1]
        expected_dim = tf.constant(self.hidden_dim, dtype=hidden_dim.dtype)
        with tf.control_dependencies([
            tf.debugging.assert_equal(hidden_dim, expected_dim,
                                      message="Flattened input dimension must equal hidden_dim."),
        ]):
            encoder_input = tf.reshape(flattened, (batch_size, 1, self.hidden_dim))

        positional = tf.broadcast_to(self.pos, (batch_size, 1, self.hidden_dim))
        memory = self.encoder(encoder_input + positional, training=training)

        class_query = tf.broadcast_to(self.class_query, (batch_size, self.hidden_dim))
        class_query = tf.expand_dims(class_query, axis=1)
        decoder_output = self.decoder(class_query, memory, training=training)
        logits = self.classifier(decoder_output)
        return logits

    def get_config(self):
        config = super().get_config()
        config.update({
            "num_classes": self.num_classes,
            "hidden_dim": self.hidden_dim,
            "num_heads": self.num_heads,
            "num_encoder_layers": self.num_encoder_layers,
            "num_decoder_layers": self.num_decoder_layers,
            "dim_feedforward": self.dim_feedforward,
            "dropout": self.dropout,
            "activation": self.activation,
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
        "QuantizableMultiHeadAttention": QuantizableMultiHeadAttention,
        "TransformerEncoderLayer": TransformerEncoderLayer,
        "TransformerDecoderLayer": TransformerDecoderLayer,
        "TransformerEncoder": TransformerEncoder,
        "TransformerDecoder": TransformerDecoder,
        "SPOTER": SPOTER,
    }):
        qat_model = tfmot.quantization.keras.quantize_apply(annotated_model)

    qat_model(dummy_input, training=False)
    return qat_model
