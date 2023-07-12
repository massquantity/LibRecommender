from .version import tf


def compute_sparse_feats(
    data_info,
    multi_sparse_combiner,
    all_sparse_indices,
    var_name,
    var_shape,
    initializer=None,
    regularizer=None,
    reuse_layer=None,
    scope_name="embedding",
    flatten=False,
):
    reuse = tf.AUTO_REUSE if reuse_layer else None
    with tf.variable_scope(scope_name, reuse=reuse):
        embed_var = tf.get_variable(
            name=var_name,
            shape=var_shape,
            initializer=initializer,
            regularizer=regularizer,
        )

    if (
        data_info.multi_sparse_combine_info
        and multi_sparse_combiner in ("sum", "mean", "sqrtn")
    ):  # fmt: skip
        embed_size = var_shape[1] if len(var_shape) == 2 else 1
        sparse_embeds = multi_sparse_combine_embedding(
            data_info,
            embed_var,
            all_sparse_indices,
            multi_sparse_combiner,
            embed_size,
        )
    else:
        sparse_embeds = tf.nn.embedding_lookup(embed_var, all_sparse_indices)

    if flatten:
        sparse_embeds = tf.keras.layers.Flatten()(sparse_embeds)
    return sparse_embeds


def multi_sparse_combine_embedding(
    data_info, embed_var, all_sparse_indices, combiner, embed_size
):
    field_offsets = data_info.multi_sparse_combine_info.field_offset
    field_lens = data_info.multi_sparse_combine_info.field_len
    feat_oovs = data_info.multi_sparse_combine_info.feat_oov
    sparse_end = field_offsets[0]

    # only one multi_sparse feature and no sparse features
    if sparse_end == 0 and len(field_offsets) == 1:
        result = multi_sparse_alone(
            embed_var,
            all_sparse_indices,
            combiner,
            embed_size,
            field_offsets[0],
            field_lens[0],
            feat_oovs[0],
        )
    else:
        if sparse_end > 0:
            sparse_indices = all_sparse_indices[:, :sparse_end]
            sparse_embedding = tf.nn.embedding_lookup(embed_var, sparse_indices)
            result = [sparse_embedding]
        else:
            result = []

        for offset, length, oov in zip(field_offsets, field_lens, feat_oovs):
            result.append(
                multi_sparse_alone(
                    embed_var,
                    all_sparse_indices,
                    combiner,
                    embed_size,
                    offset,
                    length,
                    oov,
                )
            )
        result = tf.concat(result, axis=1)
    return result


def multi_sparse_alone(
    embed_var, all_sparse_indices, combiner, embed_size, offset, length, oov
):
    variable_dim = len(embed_var.get_shape().as_list())
    # oov feats are padded to 0-vector
    oov_indices = [oov] if variable_dim == 1 else oov
    zero_padding_op = tf.scatter_update(
        embed_var, oov_indices, tf.zeros([embed_size], dtype=tf.float32)
    )
    multi_sparse_indices = all_sparse_indices[:, offset : offset + length]

    with tf.control_dependencies([zero_padding_op]):
        multi_sparse_embed = tf.nn.embedding_lookup(embed_var, multi_sparse_indices)

    res_embed = tf.reduce_sum(multi_sparse_embed, axis=1, keepdims=True)
    if combiner in ("mean", "sqrtn"):
        multi_sparse_lens = tf.reduce_sum(
            tf.cast(tf.not_equal(multi_sparse_indices, oov), tf.float32),
            axis=1,
            keepdims=True,
        )
        if combiner == "sqrtn":
            multi_sparse_lens = tf.sqrt(multi_sparse_lens)
        if variable_dim == 2:
            multi_sparse_lens = tf.expand_dims(multi_sparse_lens, axis=1)

        res_embed = tf.div_no_nan(res_embed, multi_sparse_lens)

    return res_embed


def compute_dense_feats(
    dense_values,
    var_name,
    var_shape,
    initializer=None,
    regularizer=None,
    reuse_layer=None,
    scope_name="embedding",
    flatten=False,
):
    if len(var_shape) == 2:
        dense_values = dense_values[:, :, tf.newaxis]
    reuse = tf.AUTO_REUSE if reuse_layer else None
    with tf.variable_scope(scope_name, reuse=reuse):
        embed_var = tf.get_variable(
            name=var_name,
            shape=var_shape,
            initializer=initializer,
            regularizer=regularizer,
        )

    batch_size = tf.shape(dense_values)[0]
    multiple = [batch_size, 1] if len(var_shape) == 1 else [batch_size, 1, 1]
    embed_var = tf.tile(tf.expand_dims(embed_var, axis=0), multiple)
    dense_embeds = embed_var * dense_values
    if flatten:
        dense_embeds = tf.keras.layers.Flatten()(dense_embeds)
    return dense_embeds


def get_feed_dict(
    model,
    user_indices=None,
    item_indices=None,
    labels=None,
    sparse_indices=None,
    user_sparse_indices=None,
    item_sparse_indices=None,
    dense_values=None,
    user_dense_values=None,
    item_dense_values=None,
    user_interacted_seq=None,
    user_interacted_len=None,
    is_training=False,
):
    feed_dict = dict()
    if hasattr(model, "user_indices") and user_indices is not None:
        feed_dict.update({model.user_indices: user_indices})
    if hasattr(model, "item_indices") and item_indices is not None:
        feed_dict.update({model.item_indices: item_indices})
    if hasattr(model, "labels") and labels is not None:
        feed_dict.update({model.labels: labels})
    if hasattr(model, "is_training"):
        feed_dict.update({model.is_training: is_training})
    if hasattr(model, "sparse_indices") and sparse_indices is not None:
        feed_dict.update({model.sparse_indices: sparse_indices})
    if hasattr(model, "user_sparse_indices") and user_sparse_indices is not None:
        feed_dict.update({model.user_sparse_indices: user_sparse_indices})
    if hasattr(model, "item_sparse_indices") and item_sparse_indices is not None:
        feed_dict.update({model.item_sparse_indices: item_sparse_indices})
    if hasattr(model, "dense_values") and dense_values is not None:
        feed_dict.update({model.dense_values: dense_values})
    if hasattr(model, "user_dense_values") and user_dense_values is not None:
        feed_dict.update({model.user_dense_values: user_dense_values})
    if hasattr(model, "item_dense_values") and item_dense_values is not None:
        feed_dict.update({model.item_dense_values: item_dense_values})
    if user_interacted_seq is not None:
        feed_dict.update(
            {
                model.user_interacted_seq: user_interacted_seq,
                model.user_interacted_len: user_interacted_len,
            }
        )
    return feed_dict


def get_sparse_feed_dict(
    model,
    sparse_tensor_indices,
    sparse_tensor_values,
    user_sparse_indices=None,
    user_dense_values=None,
    batch_size=1,
    is_training=False,
):
    feed_dict = {
        model.item_interaction_indices: sparse_tensor_indices,
        model.item_interaction_values: sparse_tensor_values,
        model.modified_batch_size: batch_size,
        model.is_training: is_training,
    }
    if hasattr(model, "user_sparse_indices") and user_sparse_indices is not None:
        feed_dict.update({model.user_sparse_indices: user_sparse_indices})
    if hasattr(model, "user_dense_values") and user_dense_values is not None:
        feed_dict.update({model.user_dense_values: user_dense_values})
    return feed_dict
