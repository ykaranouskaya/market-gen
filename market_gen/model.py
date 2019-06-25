"""
Implement Transformer [	arXiv:1706.03762 ] decoder only model with self-attention
that will learn the input time series data.
"""
import numpy as np
import tensorflow as tf

from market_gen import utils


EPSILON = 1e-9


# Scaled dot-product attention
def scaled_dot_product_attention(q, k, v, masks=None):
    """
    :param
      q: query shape == (..., seq_len_q, depth)
      k: key shape == (..., seq_len_k, depth)
      v: value shape == (..., seq_len_v, depth)
      mask: Float tensor with shape broadcastable
            to (..., seq_len_q, seq_len_k). Defaults to None.
    :return: tuple (output, attention_weights)
    """
    matmult_qk = tf.matmul(q, k, transpose_b=True)

    # Scale
    dk = tf.cast(tf.shape(k)[-1], tf.float32)
    scaled_qk = matmult_qk / tf.math.sqrt(dk)

    # Add mask to the scaled level
    if masks is not None:
        scaled_qk += (masks * EPSILON)

    # Softmax normalized on the last axis
    attention_weights = tf.nn.softmax(scaled_qk, axis=-1)
    output = tf.matmul(attention_weights, v)

    return output, attention_weights


# Multi-head attention
class MultiHeadAttention:
    def __init__(self, num_heads, d_model):
        self.num_heads = num_heads
        self.d_model = d_model

        assert d_model % num_heads == 0, "Number of heads must divide d_model"
        self.depth = d_model // num_heads

        self.wq = tf.keras.layers.Dense(d_model)
        self.wk = tf.keras.layers.Dense(d_model)
        self.wv = tf.keras.layers.Dense(d_model)

        self.dense = tf.keras.layers.Dense(d_model)

    def split_heads(self, x, batch_size):
        x = tf.reshape(x, (batch_size, -1, self.num_heads, self.depth))
        x = tf.transpose(x, perm=[0, 2, 1, 3])

        return x

    def __call__(self, q, k, v, mask):
        batch_size = tf.shape(q)[0]

        q = self.wq(q)
        k = self.wk(k)
        v = self.wv(v)

        q = self.split_heads(q, batch_size)
        k = self.split_heads(k, batch_size)
        v = self.split_heads(v, batch_size)

        scaled_attention, attention_weights = scaled_dot_product_attention(q, k,
                                                                           v, mask)
        scaled_attention = tf.transpose(scaled_attention, perm=[0, 2, 1, 3])
        concat_attention = tf.reshape(scaled_attention, (batch_size, -1, self.d_model))

        output = self.dense(concat_attention)

        return output, attention_weights


# Point wise FFN
def pointwise_feedforward(d_model, dff):
    ffn = tf.keras.Sequential([
          tf.keras.layers.Dense(dff, activation='relu'),
          tf.keras.layers.Dense(d_model)
          ])

    return ffn


class DecoderLayer(tf.keras.layers.Layer):
    def __init__(self, d_model, num_heads, dff, rate=0.1):
        super(DecoderLayer, self).__init__()

        self.mha1 = MultiHeadAttention(num_heads, d_model)
        self.mha2 = MultiHeadAttention(num_heads, d_model)

        self.ffn = pointwise_feedforward(d_model, dff)

        self.layernorm1 = tf.keras.layers.LayerNormalization(epsilon=1e-6)
        self.layernorm2 = tf.keras.layers.LayerNormalization(epsilon=1e-6)
        self.layernorm3 = tf.keras.layers.LayerNormalization(epsilon=1e-6)

        self.dropout1 = tf.keras.layers.Dropout(rate=rate)
        self.dropout2 = tf.keras.layers.Dropout(rate=rate)
        self.dropout3 = tf.keras.layers.Dropout(rate=rate)

    def __call__(self, x,
                 training, look_ahead_mask):
        attn1, attn1_weights = self.mha1(x, x, x, look_ahead_mask)
        attn1 = self.dropout1(attn1, training=training)
        out1 = self.layernorm1(x + attn1)

        ffn = self.ffn(out1)
        ffn = self.dropout3(ffn, training=training)

        out3 = self.layernorm3(out1 + ffn)

        return out3, attn1_weights


class Decoder(tf.keras.layers.Layer):
    def __init__(self, num_layers, d_model, num_heads, dff, pos_period, rate=0.1):
        super().__init__()
        self.d_model = d_model
        self.num_layers = num_layers

        self.cnn_input = tf.keras.layers.Conv1D(d_model, 1, activation='relu')
        self.positional_encoder = utils.positional_encoding(pos_period, d_model)

        self.decoder_layers = [DecoderLayer(d_model, num_heads, dff, rate)
                               for _ in range(num_layers)]

        self.dropout = tf.keras.layers.Dropout(rate)

    def __call__(self, x,
                 training, look_ahead_mask):
        seq_len = tf.shape(x)[1]
        attn_weights = {}

        x = tf.expand_dims(x, -1)
        x = tf.cast(x, dtype=tf.float32)
        x = self.cnn_input(x)
        x = tf.cast(x, dtype=tf.float32)
        x += self.positional_encoder[:, :seq_len, :]

        x = self.dropout(x, training=training)

        for i in range(self.num_layers):
            x, attn1 = self.decoder_layers[i](x, training,
                                              look_ahead_mask)
            attn_weights['decoder_layer{}_block'.format(i + 1)] = attn1

        return x, attn_weights


# Actual Transformer model
class Transformer(tf.keras.Model):
    def __init__(self, num_layers, d_model, num_heads, dff, pos_period, rate=0.1):
        super().__init__()

        self.decoder = Decoder(num_layers, d_model, num_heads, dff,
                               pos_period, rate)
        self.final_layer = tf.keras.layers.Dense(1)

    def __call__(self, inp, training, look_ahead_mask,
                 enc_padding_mask, dec_padding_mask):
        dec_output, attn_weights = self.decoder(inp,
                                                training,
                                                look_ahead_mask)

        final_output = self.final_layer(dec_output)
        final_output = tf.squeeze(final_output)

        return final_output, attn_weights
