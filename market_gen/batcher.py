import json

import numpy as np

from market_gen import utils


def price_generator(data, lookback, delay, min_index=0, max_index=None,
                    shuffle=True, batch_size=128, step=1, column=-1):
    """
    Time series input batcher. Generates (samples, targets) tuple
    :param data: sequence of data
    :param lookback: length of samples sequence
    :param delay: length of target sequence
    :param min_index: starting index in the data
    :param max_index: ending index in the data
    :param shuffle: if True, generate non sequential batches
    :param batch_size: batch size
    :param step: stride to use for data sampling
    :param column: index of feature choose from data vector
    """
    if max_index is None:
        max_index = len(data) - delay - 1

    i = min_index + lookback

    while True:
        if shuffle:
            data_idx = np.random.randint(min_index + lookback, max_index,
                                         size=batch_size)

        else:
            if i + batch_size >= len(data):
                i = min_index + lookback
            data_idx = np.arange(i, min(i + batch_size, max_index))
            i += len(data_idx)

        samples = np.zeros((len(data_idx), lookback // step))
        targets = np.zeros((len(data_idx), delay // step))
        for j, row in enumerate(data_idx):
            indices = range(data_idx[j] - lookback, data_idx[j], step)
            target_indices = range(data_idx[j], data_idx[j] + delay, step)
            samples[j] = data[indices][:, column]
            targets[j] = data[target_indices][:, column]
        yield samples.astype('float32'), targets.astype('float32')


def load_daily_data(data_path, split=None, d=0.5, thresh=1e-2,
                    column='close'):
    with open(data_path, 'r') as f:
        data = json.load(f)

    df = utils.json_to_csv(data)

    if split:
        with open(split, 'r') as f:
            split_data = json.load(f)
        df = utils.adjust_for_splits(df, split_data)

    df = df[::-1]
    df_frac = utils.frac_diff_ffd(df[[column]], d, thresh)
    df[f'{column}_frac'] = df_frac[column]

    df.dropna(inplace=True)

    data_vector = df[df.columns[1:]].values
    data_vector = data_vector[1:, :]

    return data_vector


class Batcher:
    def __init__(self, data_path, split=None, frac_d=0.5, frac_thr=1e-2, feature='close'):
        self.data = load_daily_data(data_path, split, frac_d, frac_thr, feature)

    def batch_gen(self, lookback=10, delay=5,
                  min_index=0, max_index=None, shuffle=True,
                  batch_size=128, step=1, column=-1):

        return price_generator(self.data, lookback, delay, min_index,
                               max_index, shuffle, batch_size, step, column)

