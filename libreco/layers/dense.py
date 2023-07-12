from functools import partial

import numpy as np

from ..tfops import get_tf_version, tf


# It turns out that the position of `batch normalization` layer matters in neural networks, see discussions in:
# https://stackoverflow.com/questions/39691902/ordering-of-batch-normalization-and-dropout
# https://www.zhihu.com/question/283715823
# Also according to the discussions, it is generally NOT recommended to use `batch normalization` and `dropout` simultaneously.
def dense_nn(
    net,
    hidden_units,
    activation=tf.nn.relu,
    use_bn=True,
    bn_after_activation=True,
    dropout_rate=None,
    is_training=True,
    reuse_layer=False,
    name="mlp",
):
    if activation is None:
        activation = tf.identity
    if np.isscalar(hidden_units):
        hidden_units = [hidden_units]

    reuse = tf.AUTO_REUSE if reuse_layer else None
    with tf.variable_scope(name, reuse=reuse):
        if use_bn:
            net = tf.layers.batch_normalization(net, training=is_training)
        for i, units in enumerate(hidden_units, start=1):
            layer_name = name + "_layer" + str(i)
            net = tf_dense(units, reuse=reuse_layer, name=layer_name)(net)
            if i != len(hidden_units):
                if use_bn:
                    if bn_after_activation:
                        net = activation(net)
                        net = tf.layers.batch_normalization(net, training=is_training)
                    else:
                        net = tf.layers.batch_normalization(net, training=is_training)
                        net = activation(net)
                else:
                    net = activation(net)

                if dropout_rate:
                    net = tf.layers.dropout(net, dropout_rate, training=is_training)

    return net


def tf_dense(
    units,
    activation=None,
    kernel_initializer="glorot_uniform",
    use_bias=True,
    reuse=False,
    name=None,
    version=None,
):
    tf_version = get_tf_version(version)
    # only tf1 layers can be reused
    if tf_version >= "2.0.0" and not reuse:
        net = tf.keras.layers.Dense(
            units=units,
            activation=activation,
            kernel_initializer=kernel_initializer,
            use_bias=use_bias,
            name=name,
        )
    else:
        net = partial(
            tf.layers.dense,
            units=units,
            activation=activation,
            kernel_initializer=kernel_initializer,
            use_bias=use_bias,
            name=name,
        )
    return net


def shared_dense(
    inputs,
    units,
    activation=None,
    use_bias=True,
    kernel_initializer="glorot_uniform",
    name=None,
    scope_name="shared_dense",
):
    with tf.variable_scope(scope_name, reuse=tf.AUTO_REUSE):
        return tf.layers.dense(
            inputs,
            units,
            activation=activation,
            use_bias=use_bias,
            kernel_initializer=kernel_initializer,
            name=name,
        )
