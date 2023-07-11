from functools import partial

from ..tfops import get_tf_version, tf


def conv_nn(
    filters, kernel_size, strides, padding, activation, dilation_rate=1, version=None
):
    tf_version = get_tf_version(version)
    if tf_version >= "2.0.0":
        net = tf.keras.layers.Conv1D(
            filters=filters,
            kernel_size=kernel_size,
            strides=strides,
            padding=padding,
            activation=activation,
            dilation_rate=dilation_rate,
        )
    else:
        net = partial(
            tf.layers.conv1d,
            filters=filters,
            kernel_size=kernel_size,
            strides=strides,
            padding=padding,
            activation=activation,
        )
    return net


def max_pool(pool_size, strides, padding, version=None):
    tf_version = get_tf_version(version)
    if tf_version >= "2.0.0":
        net = tf.keras.layers.MaxPool1D(
            pool_size=pool_size, strides=strides, padding=padding
        )
    else:
        net = partial(
            tf.layers.max_pooling1d,
            pool_size=pool_size,
            strides=strides,
            padding=padding,
        )
    return net
