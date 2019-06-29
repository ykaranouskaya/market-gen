import numpy as np
import pandas as pd
import tensorflow as tf
import matplotlib.pyplot as plt


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


def adjust_for_splits(df, splits):
    for t in splits:
        coef = splits[t]
        indx = df['date'] < t

        df.loc[indx, 'open'] /= coef
        df.loc[indx, 'close'] /= coef
        df.loc[indx, 'high'] /= coef
        df.loc[indx, 'low'] /= coef
        df.loc[indx, 'volume'] *= coef

    return df


def json_to_csv(json_data):
    dict_data = {'date': [],
                 'open': [],
                 'close': [],
                 'high': [],
                 'low': [],
                 'volume': []}
    json_data = json_data['Time Series (Daily)']
    for date in json_data:
        dict_data['date'].append(date)
        for item in json_data[date]:
            key = item[3:]
            dict_data[key].append(float(json_data[date][item]))
    csv_data = pd.DataFrame(data=dict_data)
    return csv_data


# Fractional differentiation - fixed width window
# Reference: Marcos Lopez de Prado 'Advances in Financial Machine Learning'
def get_weights_ffd(d, thresh):
    k = 1
    w = [1.]
    while abs(w[-1]) > thresh:
        w_ = -w[-1] / k * (d - k + 1)
        w.append(w_)
        k += 1
    w = np.array(w[::-1]).reshape(-1, 1)
    return w


def frac_diff_ffd(series, d, thresh=1e-5):
    # Compute weights for the longest series
    w = get_weights_ffd(d, thresh)
    width = len(w) - 1

    # Apply weights to values
    df = {}
    for name in series.columns:
        series_f, df_ = series[[name]].fillna(method='ffill').dropna(), pd.Series()
        for iloc1 in range(width, series_f.shape[0]):
            loc0, loc1 = series_f.index[iloc1 - width], series_f.index[iloc1]
            if not np.isfinite(series.loc[loc1, name]):
                continue

            df_.loc[loc1] = np.dot(w.T, series_f.loc[loc0:loc1])[0][0]
        df[name] = df_.copy(deep=True)

    df = pd.concat(df, axis=1)

    return df


def plot_data(data, time_data, save_path=None):
    close = data[:, 1]
    close_frac = data[:, -1]
    df = pd.DataFrame(data={'close': close, 'close_frac_diff': close_frac},
                      index=time_data)
    fig = plt.figure(figsize=(16, 8))
    ax = fig.add_subplot(1, 2, 1)
    df.plot(y='close', ax=ax)

    ax = fig.add_subplot(1, 2, 2)
    df.plot(y='close_frac_diff', ax=ax)

    if save_path:
        plt.savefig(save_path)
