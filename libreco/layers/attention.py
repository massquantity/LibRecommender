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
        Typically sequence mask: shape: (batch_size, seq_len)

    Returns
    -------
    attention outputs : (batch_size, embed_size)
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
        Typically sequence mask: shape: (batch_size, seq_len)

    Returns
    -------
    attention outputs : (batch_size, embed_size)
    """
    from ..layers import dense_nn

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


def multi_head_attention(queries, keys, num_heads, head_dim, version=None):
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
    version : str
        Specified tf version, mainly used for testing.

    Returns
    -------
    attention outputs : (batch_size, embed_size) or (batch_size, seq_q_len, embed_size)
    """
    from ..layers import tf_dense

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
    # H * B * F * K
    queries = tf.stack(tf.split(queries, num_heads, axis=2))
    keys = tf.stack(tf.split(keys, num_heads, axis=2))
    values = tf.stack(tf.split(values, num_heads, axis=2))
    # H * B * F * F
    att_weights = queries @ tf.transpose(keys, [0, 1, 3, 2])
    att_weights *= tf.math.rsqrt(tf.cast(head_dim, tf.float32))
    att_weights = tf.nn.softmax(att_weights)
    # H * B * F * K
    outputs = att_weights @ values
    # 1 * B * F * (K*H)
    outputs = tf.concat(tf.split(outputs, num_heads, axis=0), axis=-1)
    outputs = tf_dense(units=output_dim)(tf.squeeze(outputs, axis=0))
    return outputs
