from ..tfops import tf


def embedding_lookup(
    indices,
    var_name=None,
    var_shape=None,
    initializer=None,
    regularizer=None,
    reuse_layer=None,
    embed_var=None,
    scope_name="embedding",
):
    reuse = tf.AUTO_REUSE if reuse_layer else None
    with tf.variable_scope(scope_name, reuse=reuse):
        if embed_var is None:
            embed_var = tf.get_variable(
                name=var_name,
                shape=var_shape,
                initializer=initializer,
                regularizer=regularizer,
            )
        return tf.nn.embedding_lookup(embed_var, indices)


def sparse_embeds_pooling(
    sparse_indices,
    var_name,
    var_shape,
    initializer,
    regularizer=None,
    reuse_layer=None,
    combiner="sqrtn",
    scope_name="sparse_embeds_pooling",
):
    reuse = tf.AUTO_REUSE if reuse_layer else None
    with tf.variable_scope(scope_name, reuse=reuse):
        embed_var = tf.get_variable(
            name=var_name,
            shape=var_shape,
            initializer=initializer,
            regularizer=regularizer,
        )
        # unknown user will return 0-vector in `safe_embedding_lookup_sparse`
        return tf.nn.safe_embedding_lookup_sparse(
            embed_var,
            sparse_indices,
            sparse_weights=None,
            combiner=combiner,
            default_id=None,
        )


def seq_embeds_pooling(
    seq_indices,
    seq_lens,
    n_items,
    var_name,
    var_shape,
    initializer=None,
    regularizer=None,
    reuse_layer=None,
    scope_name="seq_embeds_pooling",
):
    reuse = tf.AUTO_REUSE if reuse_layer else None
    with tf.variable_scope(scope_name, reuse=reuse):
        embed_var = tf.get_variable(
            name=var_name,
            shape=var_shape,
            initializer=initializer,
            regularizer=regularizer,
        )
        # unknown items are padded to 0-vector
        embed_size = var_shape[1]
        zero_padding_op = tf.scatter_update(
            embed_var, n_items, tf.zeros([embed_size], dtype=tf.float32)
        )
        with tf.control_dependencies([zero_padding_op]):
            # B * seq * K
            multi_item_embed = tf.nn.embedding_lookup(embed_var, seq_indices)

        return tf.div_no_nan(
            tf.reduce_sum(multi_item_embed, axis=1),
            tf.expand_dims(tf.sqrt(tf.cast(seq_lens, tf.float32)), axis=1),
        )
