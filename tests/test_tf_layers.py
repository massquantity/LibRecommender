import numpy as np
import pytest

from libreco.layers import conv_nn, dense_nn, max_pool, shared_dense, tf_dense, tf_rnn
from libreco.tfops import dropout_config, reg_config, tf


@pytest.fixture
def random_data(dim):
    if dim == 2:
        return tf.random.normal([100, 10], seed=2042)
    elif dim == 3:
        return tf.random.normal([100, 20, 10], seed=2042)


@pytest.mark.parametrize("dim", [2])
def test_dense_layer(random_data, dim):
    with tf.Session() as sess:
        output = dense_nn(
            random_data,
            [10, 3],
            activation=None,
            use_bn=True,
            bn_after_activation=False,
        )
        output2 = tf_dense(7, version="1.15")(random_data)
        sess.run(tf.global_variables_initializer())
        assert sess.run(output).shape == (100, 3)
        assert sess.run(output2).shape == (100, 7)


@pytest.mark.parametrize("dim", [2])
def test_shared_dense(random_data, dim):
    with tf.Session() as sess:
        output = shared_dense(random_data, 3, name="layer1")
        output2 = shared_dense(random_data, 3, name="layer1")
        sess.run(tf.global_variables_initializer())
        assert sess.run(output).shape == (100, 3)
        assert sess.run(output2).shape == (100, 3)

        with tf.variable_scope("shared_dense", reuse=tf.AUTO_REUSE):
            v1 = tf.get_variable("layer1", shape=(10, 3))
            v2 = tf.get_variable("layer1", shape=(10, 3))
            assert v1 is v2

        with pytest.raises(ValueError), tf.variable_scope("shared_dense"):
            tf.get_variable("layer1", shape=(10, 3))
            tf.get_variable("layer1", shape=(10, 3))

        with pytest.raises(ValueError):
            tf.get_variable("shared_dense/layer1", shape=(10, 3))


@pytest.mark.parametrize("dim", [3])
def test_conv_layer(random_data, dim):
    with tf.Session() as sess:
        output = conv_nn(
            filters=2,
            kernel_size=3,
            strides=2,
            padding="valid",
            activation="relu",
            version="1.15",
        )(random_data)
        sess.run(tf.global_variables_initializer())
        assert sess.run(output).shape == (100, 9, 2)


@pytest.mark.parametrize("dim", [3])
def test_max_pool_layer(random_data, dim):
    with tf.Session() as sess:
        output = max_pool(pool_size=3, strides=2, padding="valid", version="1.15")(
            random_data
        )
        sess.run(tf.global_variables_initializer())
        assert sess.run(output).shape == (100, 9, 10)


@pytest.mark.parametrize("dim", [3])
def test_rnn_layer(random_data, dim):
    np_rng = np.random.default_rng(42)
    with tf.Session() as sess:
        output = tf_rnn(
            inputs=random_data,
            rnn_type="lstm",
            lengths=np_rng.integers(0, 20, [100]),
            maxlen=20,
            hidden_units=[16, 8],
            dropout_rate=0.1,
            use_ln=True,
            is_training=True,
            version="1.15",
        )
        output2 = tf_rnn(
            inputs=random_data,
            rnn_type="gru",
            lengths=np_rng.integers(0, 20, [100]),
            maxlen=20,
            hidden_units=[16],
            dropout_rate=0.1,
            use_ln=True,
            is_training=True,
            version="2.10",
        )
        sess.run(tf.global_variables_initializer())
        assert sess.run(output).shape == (100, 8)
        assert sess.run(output2).shape == (100, 16)


def test_config():
    with pytest.raises(ValueError):
        reg_config(1)

    with pytest.raises(ValueError):
        dropout_config(1.1)
