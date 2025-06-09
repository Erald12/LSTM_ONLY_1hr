# custom_objects.py
import tensorflow as tf
from tensorflow.keras.layers import Layer


def heteroscedastic_loss(true, pred):
    mu = pred[:, 0]         # predicted mean
    raw_log_var = pred[:, 1]  # predicted log variance

    # Ensure numerical stability and positivity
    log_var = tf.math.softplus(raw_log_var)
    log_var = tf.clip_by_value(log_var, 1e-6, 50.0)

    precision = tf.exp(-log_var)

    # Heteroscedastic loss: negative log-likelihood of Gaussian
    loss = precision * tf.square(true[:, 0] - mu) + log_var

    return tf.reduce_mean(loss)


class ClipByValueLayer(Layer):
    def __init__(self, clip_min, clip_max, **kwargs):
        super().__init__(**kwargs)
        self.clip_min = clip_min
        self.clip_max = clip_max

    def call(self, inputs):
        return tf.clip_by_value(inputs, self.clip_min, self.clip_max)