import sys

import numpy as np
import pytest
from numpy.testing import assert_allclose, assert_array_equal

from libreco.layers import (
    conv_nn,
    dense_nn,
    layer_normalization,
    max_pool,
    multi_head_attention,
    rms_norm,
    shared_dense,
    tf_dense,
    tf_rnn,
)
from libreco.layers.activation import gelu, swish
from libreco.layers.transformer import (
    positional_encoding,
    transformer_decoder_layer,
    transformer_encoder_layer,
)
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


def test_layer_norm():
    with tf.Session() as sess:
        inputs = tf.constant(np.arange(10).reshape(5, 2) * 10, dtype=tf.float32)
        outputs = layer_normalization(inputs)
        sess.run(tf.global_variables_initializer())
        labels = np.array([-1, 1] * 5, dtype=np.float32).reshape(5, 2)
        assert_array_equal(sess.run(outputs), labels)


def test_rms_norm():
    with tf.Session() as sess:
        inputs = np.arange(10).reshape(5, 2) * 10
        outputs = rms_norm(tf.constant(inputs, dtype=tf.float32))
        sess.run(tf.global_variables_initializer())
        labels = inputs / np.sqrt(np.mean(np.square(inputs), axis=-1, keepdims=True))
        assert_allclose(sess.run(outputs), labels, rtol=1e-4)


@pytest.mark.skipif(
    sys.version_info[:2] < (3, 7),
    reason="Tensorflow doesn't have `MultiHeadAttention` layer before py3.7",
)
def test_multi_head_attention():
    with tf.Session() as sess:
        queries = tf.ones([2, 3, 4], dtype=tf.float32)
        keys = tf.reshape(tf.range(30, dtype=tf.float32), (2, 5, 3))
        rng = np.random.default_rng()
        mask = tf.constant(rng.integers(0, 2, (2, 3, 5), dtype=np.bool_))
        output1 = multi_head_attention(
            queries, keys, num_heads=4, head_dim=4, mask=mask, version="2.11"
        )
        output2 = multi_head_attention(
            queries, keys, num_heads=4, head_dim=4, mask=mask, version="1.15"
        )
        sess.run(tf.global_variables_initializer())
        assert sess.run(output1).shape == sess.run(output2).shape == (2, 3, 4)


def test_positional_encoding():
    tf.reset_default_graph()
    with tf.Session() as sess:
        pe = positional_encoding(3, 3)
        sess.run(tf.global_variables_initializer())
        assert sess.run(pe).shape == (3, 3)

    tf.reset_default_graph()
    with tf.Session() as sess:
        pe2 = positional_encoding(10, 10)
        sess.run(tf.global_variables_initializer())
        assert sess.run(pe2).shape == (10, 10)


def test_transformer_encoder():
    tf.reset_default_graph()
    with tf.Session() as sess:
        batch_size = 2
        max_seq_len = 3
        embed_size = 4
        num_heads = 2
        head_dim = 2
        seqs = tf.random.normal((batch_size, max_seq_len, embed_size))
        seq_lens = tf.constant([1, 2])
        output = transformer_encoder_layer(
            seqs, seq_lens, max_seq_len, num_heads, head_dim, embed_size
        )
        sess.run(tf.global_variables_initializer())
        assert sess.run(output).shape == (batch_size, max_seq_len, embed_size)


def test_transformer_decoder():
    tf.reset_default_graph()
    with tf.Session() as sess:
        batch_size = 2
        max_seq_len = 3
        embed_size = 4
        num_heads = 2
        head_dim = 2
        seqs = tf.random.normal((batch_size, max_seq_len, embed_size))
        seq_lens = tf.constant([3, 1])
        encoder_output = tf.random.normal((batch_size, max_seq_len, embed_size))
        output = transformer_decoder_layer(
            encoder_output, seqs, seq_lens, max_seq_len, num_heads, head_dim, embed_size
        )
        sess.run(tf.global_variables_initializer())
        assert sess.run(output).shape == (batch_size, max_seq_len, embed_size)


def test_gelu():
    with tf.Session() as sess:
        inputs = tf.constant([-3.0, -1.0, 0.0, 1.0, 3.0], dtype=tf.float32)
        output = gelu(inputs)
        assert_allclose(
            sess.run(output),
            [-0.00404951, -0.15865529, 0.0, 0.8413447, 2.9959507],
            rtol=1e-4,
        )


def test_swish():
    with tf.Session() as sess:
        inputs = tf.constant([-3.0, -1.0, 0.0, 1.0, 3.0], dtype=tf.float32)
        output = swish(inputs)
        assert_allclose(
            sess.run(output), [-0.142278, -0.268941, 0.0, 0.731059, 2.857722], rtol=1e-4
        )


def test_config():
    with pytest.raises(ValueError):
        reg_config(1)

    with pytest.raises(ValueError):
        dropout_config(1.1)
