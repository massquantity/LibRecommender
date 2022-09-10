from .version import tf


def multi_sparse_combine_embedding(
    data_info, variables, all_sparse_indices, combiner, embed_size
):
    field_offsets = data_info.multi_sparse_combine_info.field_offset
    field_lens = data_info.multi_sparse_combine_info.field_len
    feat_oovs = data_info.multi_sparse_combine_info.feat_oov
    sparse_end = field_offsets[0]

    # only one multi_sparse feature and no sparse features
    if sparse_end == 0 and len(field_offsets) == 1:
        result = multi_sparse_alone(
            variables,
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
            sparse_embedding = tf.nn.embedding_lookup(variables, sparse_indices)
            result = [sparse_embedding]
        else:
            result = []

        for offset, length, oov in zip(field_offsets, field_lens, feat_oovs):
            result.append(
                multi_sparse_alone(
                    variables,
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
    variables, all_sparse_indices, combiner, embed_size, offset, length, oov
):
    variable_dim = len(variables.get_shape().as_list())
    # oov feats are padded to 0-vector
    oov_indices = [oov] if variable_dim == 1 else oov
    zero_padding_op = tf.scatter_update(
        variables, oov_indices, tf.zeros([embed_size], dtype=tf.float32)
    )
    multi_sparse_indices = all_sparse_indices[:, offset : offset + length]

    with tf.control_dependencies([zero_padding_op]):
        multi_sparse_embed = tf.nn.embedding_lookup(variables, multi_sparse_indices)

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


def get_feed_dict(
    model,
    user_indices,
    item_indices,
    labels=None,
    sparse_indices=None,
    dense_values=None,
    user_interacted_seq=None,
    user_interacted_len=None,
    is_training=False,
):
    feed_dict = {model.user_indices: user_indices, model.item_indices: item_indices}
    if labels is not None:
        feed_dict.update({model.labels: labels})
    if hasattr(model, "is_training"):
        feed_dict.update({model.is_training: is_training})
    if hasattr(model, "sparse") and model.sparse:
        feed_dict.update({model.sparse_indices: sparse_indices})
    if hasattr(model, "dense") and model.dense:
        feed_dict.update({model.dense_values: dense_values})
    if model.model_category == "sequence":
        feed_dict.update(
            {
                model.user_interacted_seq: user_interacted_seq,
                model.user_interacted_len: user_interacted_len,
            }
        )
    return feed_dict
