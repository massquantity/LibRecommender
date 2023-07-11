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
        output3 = tf_dense(3, name="layer2")(random_data)
        output4 = tf_dense(3, name="layer2")(random_data)
        sess.run(tf.global_variables_initializer())

        assert sess.run(output).shape == (100, 3)
        assert sess.run(output2).shape == (100, 3)
        assert sess.run(output3).shape == (100, 3)
        assert sess.run(output4).shape == (100, 3)

        with tf.variable_scope("shared_dense", reuse=True):
            with tf.variable_scope("layer1"):
                v1 = tf.get_variable("kernel")
                v2 = tf.get_variable("kernel")
        with tf.variable_scope("", reuse=True):
            v3 = tf.get_variable("shared_dense/layer1/kernel")
            assert v1 is v2 is v3

        msg = "Variable {} does not exist, or was not created with tf.get_variable()*"
        with pytest.raises(ValueError, match=msg.format("shared_dense/layer1_1/kernel")):  # fmt: skip
            with tf.variable_scope("shared_dense", reuse=True):
                with tf.variable_scope("layer1_1"):
                    tf.get_variable("kernel")

        with pytest.raises(ValueError, match=msg.format("layer2/kernel")):
            with tf.variable_scope("layer2", reuse=True):
                tf.get_variable("kernel")

        var_names = [v.name for v in tf.trainable_variables()]
        assert "layer2/kernel:0" in var_names
        assert "layer2_1/kernel:0" in var_names


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
