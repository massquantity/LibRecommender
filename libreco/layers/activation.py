from ..tfops import tf


def gelu(x):
    return 0.5 * x * (1.0 + tf.math.erf(x / tf.cast(1.4142135623730951, tf.float32)))


def swish(x):
    return x * tf.sigmoid(x)
