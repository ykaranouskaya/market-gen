import argparse
import json
import time
import sys

import tensorflow as tf

from market_gen import utils
from market_gen.model import Transformer
from market_gen.batcher import Batcher


def parse_arguments(argv):
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint_dir', help="Path to the checkpoint directory")

    # Data parameters
    parser.add_argument('--data_path', help="Path to the data")
    parser.add_argument('--splits_data', default=None, help="Path to json file with splits info")

    # Model parameters
    parser.add_argument('--num_layers', type=int, default=4, help="Number of decoder MHA + FFN layers")
    parser.add_argument('--d_model', type=int, default=128, help="Internal dimension of the model")
    parser.add_argument('--feedforward_units', type=int, default=512,
                        help="Number of hidden units in feedforward layer")
    parser.add_argument('--num_heads', type=int, default=8,
                        help="Number of heads for multihead attention")
    parser.add_argument('--pos_period', type=int, default=100,
                        help="Size of sequence to use for positional encoding")
    parser.add_argument('--dropout_rate', type=float, default=0.1,
                        help="Fraction of the units to drop in dropout layer")

    # Train parameters
    parser.add_argument('--epochs', type=int, default=10, help="Number of training epochs")
    parser.add_argument('--epoch_len', type=int, default=1000, help="Number of steps in epoch")

    return parser.parse_args(argv)


class CustomSchedule(tf.keras.optimizers.schedules.LearningRateSchedule):
    """
    LR Schedule from [arXiv:1706.03762]
    """
    def __init__(self, d_model, warmup_steps=4000):
        super().__init__()

        self.d_model = d_model
        self.warmup_steps = warmup_steps

    def __call__(self, step):
        step = tf.cast(step, dtype=tf.float32)
        arg1 = step ** (-0.5)
        arg2 = step * (self.warmup_steps ** -1.5)

        return (self.d_model ** -0.5) * tf.math.minimum(arg1, arg2)


def loss_function(real, pred):
    """
    MAE loss.
    """
    loss = tf.keras.losses.mean_absolute_error(real, pred)

    return tf.reduce_mean(loss)


class TrainModel:

    def __init__(self, model, learning_rate):
        self.model = model
        self.optimizer = tf.keras.optimizers.Adam(learning_rate, beta_1=0.9, beta_2=0.98,
                                                  epsilon=1e-9)
        self.train_loss = tf.keras.metrics.Mean(name='train_loss')

    @classmethod
    def decoder_from_params(cls, learning_rate, **kwargs):
        decoder = Transformer(**kwargs)
        return cls(decoder, learning_rate)

    @tf.function
    def train_step(self, inp):
        tar_inp = inp[:, :-1]  # Decoder input
        tar_real = inp[:, 1:]  # Decoder target

        look_ahead_mask = utils.create_look_ahead_mask(tf.shape(tar_inp)[1])

        with tf.GradientTape() as tape:
            pred, attn_weights = self.model(tar_inp, True,
                                            look_ahead_mask)
            loss = loss_function(pred, tar_real)

        gradients = tape.gradient(loss, self.model.trainable_variables)
        self.optimizer.apply_gradients(zip(gradients, self.model.trainable_variables))

        self.train_loss(loss)

    def train_model(self, data_gen, checkpoint_dir, epochs=10,
                    save_ckpt_every=5, epoch_len=1000):

        checkpoint = tf.train.Checkpoint(model=self.model,
                                         optimizer=self.optimizer)
        ckpt_manager = tf.train.CheckpointManager(checkpoint, checkpoint_dir,
                                                  max_to_keep=5)

        if ckpt_manager.latest_checkpoint:
            checkpoint.restore(ckpt_manager.latest_checkpoint).expect_partial()
            print('Latest checkpoint restored!')

        for epoch in range(epochs):
            start = time.time()

            self.train_loss.reset_states()

            b = 0
            while b < epoch_len:
                inp, tar = next(data_gen)
                self.train_step(inp)

                if b % 100 == 0:
                    print('Epoch {} Batch {} Loss {:.4f}'.format(
                        epoch + 1, b, self.train_loss.result()))
                b += 1

            if (epoch + 1) % save_ckpt_every == 0:
                ckpt_save_path = ckpt_manager.save()
                print('Saving checkpoint for epoch {} at {}'.format(epoch+1,
                                                                    ckpt_save_path))

            print('Epoch {} Loss {:.4f}'.format(epoch + 1, self.train_loss.result()))
            print('Time taken for 1 epoch: {} secs\n'.format(time.time() - start))


def main(argv=None):
    argv = argv or sys.argv[1:]
    args = parse_arguments(argv)

    learning_rate = CustomSchedule(args.d_model)
    train_params = {
        "num_layers": args.num_layers,
        "d_model": args.d_model,
        "num_heads": args.num_heads,
        "dff": args.feedforward_units,
        "pos_period": args.pos_period,
        "rate": args.dropout_rate,
        }
    model = TrainModel.decoder_from_params(learning_rate, **train_params)
    batcher = Batcher(args.data_path, args.splits_data)
    train_gen = batcher.batch_gen(min_index=0, max_index=1500)

    ckpt_dir = args.checkpoint_dir

    model.train_model(train_gen, ckpt_dir, epochs=args.epochs,
                      epoch_len=args.epoch_len)
