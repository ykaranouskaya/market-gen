import numpy as np
import tensorflow as tf


def create_look_ahead_mask(size):
    """
    Look ahead mask - array of shape (size, size) as
    an upper triangle matrix of 1. This is required to force decoder
    look at earlier step in a sequence only.
    :param size: sequence length
    :return: array (size, size)
    """
    mask = 1 - tf.linalg.band_part(tf.ones((size, size)), -1, 0)
    return mask


def get_angles(pos, i, d_model):
    angle_rates = 1 / np.power(10000, (2 * (i // 2)) / np.float(d_model))
    return pos * angle_rates


def positional_encoding(position, d_model):
    """
    Cosine positional encoding.
    Encodes spatial information from the sequence to feed into transformer.
    :param position: cosine period - how
    :param d_model: dimension of encoded vector
    :return: tensor (1, position, d_model) of encoded positions
    """
    angle_rads = get_angles(np.arange(position)[:, np.newaxis],
                            np.arange(d_model)[np.newaxis, :],
                            d_model)

    sines = np.sin(angle_rads[:, 0::2])
    cosines = np.cos(angle_rads[:, 1::2])

    pos_encoding = np.concatenate([sines, cosines], axis=-1)
    pos_encoding = pos_encoding[np.newaxis, :]

    return tf.cast(pos_encoding, dtype=tf.float32)


