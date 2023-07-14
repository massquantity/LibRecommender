from .dense import dense_nn, tf_dense
from ..tfops import get_tf_version, tf


def tf_attention(queries, keys, key_masks):
    """Dot-product attention from tensorflow.

    Parameters
    ----------
    queries : tf.Tensor
        Typically target items, shape: (batch_size, embed_size)
    keys : tf.Tensor
        Typically behavior sequences, shape: (batch_size, seq_len, embed_size)
    key_masks : tf.Tensor
        Typically sequence mask, shape: (batch_size, seq_len)

    Returns
    -------
    attention outputs
        Shape: (batch_size, embed_size)
    """
    queries = tf.expand_dims(queries, axis=1)
    attention = tf.keras.layers.Attention(use_scale=False)
    outputs = attention(inputs=[queries, keys], mask=[None, key_masks])
    return tf.squeeze(outputs, axis=1)


def din_attention(queries, keys, key_masks):
    """Attention proposed in DIN paper.

    Parameters
    ----------
    queries : tf.Tensor
        Typically target items, shape: (batch_size, embed_size)
    keys : tf.Tensor
        Typically behavior sequences, shape: (batch_size, seq_len, embed_size)
    key_masks : tf.Tensor
        Typically sequence mask, shape: (batch_size, seq_len)

    Returns
    -------
    attention outputs
        Shape: (batch_size, embed_size)
    """
    queries = tf.tile(queries[:, tf.newaxis, :], [1, keys.shape[1], 1])
    queries_keys_cross = tf.concat(
        [queries, keys, queries - keys, queries * keys], axis=2
    )
    mlp_layer = dense_nn(
        queries_keys_cross,
        (16, 1),
        use_bn=False,
        activation=tf.nn.sigmoid,
        name="attention",
    )
    attention_weights = tf.squeeze(mlp_layer, axis=2)
    attention_weights *= tf.math.rsqrt(tf.cast(keys.shape[-1], tf.float32))
    paddings = tf.ones_like(attention_weights) * (-(2**32) + 1)
    attention_scores = tf.where(key_masks, attention_weights, paddings)
    # B * 1 * seq
    attention_scores = tf.nn.softmax(attention_scores)[:, tf.newaxis, :]
    # B * 1 * K
    outputs = attention_scores @ keys
    return tf.squeeze(outputs, axis=1)


def multi_head_attention(
    queries, keys, num_heads, head_dim, attention_mask=None, version=None
):
    """Multi-Head Attention proposed in `Attention Is All You Need` paper.

    Parameters
    ----------
    queries : tf.Tensor
        Shape: (batch_size, embed_size) or (batch_size, seq_q_len, embed_size)
    keys : tf.Tensor
        Shape: (batch_size, seq_k_len, embed_size), we will use keys for both key and value
    num_heads : int
        Number of attention heads.
    head_dim : int
        Dimension of each attention head.
    attention_mask : tf.Tensor
        Shape: (batch_size, num_heads, seq_q_len, seq_k_len), boolean mask to prevent attention to certain positions.
    version : str
        Specified tf version, mainly used for testing.

    Returns
    -------
    attention outputs
        Shape: (batch_size, embed_size) or (batch_size, seq_q_len, embed_size)
    """
    # if len(queries.get_shape().as_list()) == 2:
    tf_version = get_tf_version(version)
    if tf_version >= "2.10.0":
        att_layer = tf.keras.layers.MultiHeadAttention(num_heads, head_dim)
        return att_layer(queries, keys)

    output_dim = queries.shape[-1]
    mh_emb_size = num_heads * head_dim
    queries = tf_dense(mh_emb_size)(queries)
    keys = tf_dense(mh_emb_size)(keys)
    values = tf_dense(mh_emb_size)(keys)
    # B * H * T * E
    queries = _split_heads(queries, num_heads, head_dim)
    keys = _split_heads(keys, num_heads, head_dim)
    values = _split_heads(values, num_heads, head_dim)
    # B * H * Tq * Tk
    att_weights = tf.matmul(queries, keys, transpose_b=True)
    att_weights *= tf.math.rsqrt(tf.cast(head_dim, tf.float32))
    if attention_mask is not None:
        paddings = -1e9 * tf.ones_like(att_weights)
        att_weights = tf.where(attention_mask, att_weights, paddings)

    att_scores = tf.nn.softmax(att_weights)
    # B * H * Tq * E
    outputs = att_scores @ values
    # B * Tq * (E*H)
    outputs = _combine_heads(outputs, num_heads, head_dim)
    outputs = tf_dense(units=output_dim)(outputs)
    return outputs


def _split_heads(x, num_heads, head_dim):
    """Split (B, T, E) to (B, H, T, E)"""
    x = tf.reshape(x, (*x.shape[:-1], num_heads, head_dim))
    return tf.transpose(x, (0, 2, 1, 3))


def _combine_heads(x, num_heads, head_dim):
    """Combine (B * H * Tq * E) to (B * T * EH)"""
    x = tf.transpose(x, (0, 2, 1, 3))
    return tf.reshape(x, (*x.shape[:-2], num_heads * head_dim))


def compute_seq_mask(query_len, key_lens, max_key_len, num_heads):
    """Compute sequence masks for multi-head attention.

    Parameters
    ----------
    query_len : int
    key_lens : tf.Tensor
        Shape: (batch_size,)
    max_key_len : int
    num_heads : int

    Returns
    -------
    Output shape: (batch_size, num_heads, query_len, max_key_len)
    """
    # B * 1 * 1 * Tk
    seq_mask = tf.sequence_mask(key_lens, max_key_len)[:, tf.newaxis, tf.newaxis, :]
    # should repeat within batch to get (batch_size * num_heads)
    # seq_mask = tf.repeat(seq_mask, num_heads, axis=0)
    return tf.tile(seq_mask, (1, num_heads, query_len, 1))


def compute_causal_mask(batch_size, num_heads, seq_len):
    """Compute causal mask used in transformer decoder.

    Causal mask will only attend items before current item.

    Parameters
    ----------
    batch_size : int
    num_heads : int
    seq_len : int

    Returns
    -------
    Output shape: (batch_size, num_heads, seq_len, seq_len)
    """
    inputs = tf.ones((seq_len, seq_len), dtype=tf.bool)
    causal_mask = tf.linalg.band_part(inputs, -1, 0)[tf.newaxis, tf.newaxis, :, :]
    return tf.tile(causal_mask, (batch_size, num_heads, 1, 1))
