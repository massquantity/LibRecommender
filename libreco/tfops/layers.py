from functools import partial

import numpy as np

from .version import TF_VERSION, tf


# It turns out that the position of `batch normalization` layer matters in neural networks, see discussions in:
# https://stackoverflow.com/questions/39691902/ordering-of-batch-normalization-and-dropout
# https://www.zhihu.com/question/283715823
# Also according to the discussions, it is generally NOT recommended to use
# `batch normalization` and `dropout` simultaneously.
def dense_nn(
    net,
    hidden_units,
    activation=tf.nn.relu,
    use_bn=True,
    bn_after_activation=True,
    dropout_rate=None,
    is_training=True,
    name="mlp",
):
    if activation is None:
        activation = tf.identity
    if np.isscalar(hidden_units):
        hidden_units = [hidden_units]

    with tf.variable_scope(name):
        if use_bn:
            net = tf.layers.batch_normalization(net, training=is_training)
        for i, units in enumerate(hidden_units, start=1):
            layer_name = name + "_layer" + str(i)
            net = tf_dense(units, activation=None, name=layer_name)(net)
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
    name=None,
    version=None,
):
    tf_version = _get_tf_version(version)
    if tf_version >= "2.0.0":
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


def conv_nn(
    filters, kernel_size, strides, padding, activation, dilation_rate=1, version=None
):
    tf_version = _get_tf_version(version)
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
    tf_version = _get_tf_version(version)
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


def tf_rnn(
    inputs,
    rnn_type,
    lengths,
    maxlen,
    hidden_units,
    dropout_rate,
    use_ln,
    is_training,
    version=None,
):
    tf_version = _get_tf_version(version)
    if tf_version >= "2.0.0":
        # cell_type = (
        #    tf.keras.layers.LSTMCell
        #    if self.rnn_type.endswith("lstm")
        #    else tf.keras.layers.GRUCell
        # )
        # cells = [cell_type(size) for size in self.hidden_units]
        # masks = tf.sequence_mask(self.user_interacted_len, self.max_seq_len)
        # tf2_rnn = tf.keras.layers.RNN(cells, return_state=True)
        # output, *state = tf2_rnn(seq_item_embed, mask=masks)

        rnn_layer = (
            tf.keras.layers.LSTM if rnn_type.endswith("lstm") else tf.keras.layers.GRU
        )
        output = inputs
        masks = tf.sequence_mask(lengths, maxlen)
        for units in hidden_units:
            output = rnn_layer(
                units,
                return_sequences=True,
                dropout=dropout_rate,
                recurrent_dropout=dropout_rate,
                activation=None if use_ln else "tanh",
            )(output, mask=masks, training=is_training)

            if use_ln:
                output = tf.keras.layers.LayerNormalization()(output)
                output = tf.keras.activations.get("tanh")(output)

        return output[:, -1, :]

    else:
        cell_type = (
            tf.nn.rnn_cell.LSTMCell
            if rnn_type.endswith("lstm")
            else tf.nn.rnn_cell.GRUCell
        )
        cells = [cell_type(size) for size in hidden_units]
        stacked_cells = tf.nn.rnn_cell.MultiRNNCell(cells)
        zero_state = stacked_cells.zero_state(tf.shape(inputs)[0], dtype=tf.float32)
        _, state = tf.nn.dynamic_rnn(
            cell=stacked_cells,
            inputs=inputs,
            sequence_length=lengths,
            initial_state=zero_state,
            time_major=False,
        )
        return state[-1][1] if rnn_type == "lstm" else state[-1]


def _get_tf_version(version):
    if version is not None:
        assert isinstance(version, str)
        return version
    else:
        return TF_VERSION
