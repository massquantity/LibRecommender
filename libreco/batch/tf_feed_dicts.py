from .batch_unit import PairwiseBatch, PointwiseBatch, SparseBatch


def get_tf_feeds(model, data, is_training):
    if model.model_name == "YouTubeRetrieval":
        return _sparse_feed_dict(model, data, is_training)
    elif isinstance(data, PairwiseBatch):
        return _pairwise_feed_dict(model, data, is_training)
    else:
        return _pointwise_feed_dict(model, data, is_training)


def _sparse_feed_dict(model, data: SparseBatch, is_training):
    feed_dict = {
        model.item_interaction_indices: data.seqs.interacted_indices,
        model.item_interaction_values: data.seqs.interacted_values,
        model.modified_batch_size: data.seqs.modified_batch_size,
        model.item_indices: data.items,
        model.is_training: is_training,
    }
    if hasattr(model, "sparse") and model.sparse:
        feed_dict.update({model.sparse_indices: data.sparse_indices})
    if hasattr(model, "dense") and model.dense:
        feed_dict.update({model.dense_values: data.dense_values})
    return feed_dict


def _pairwise_feed_dict(model, data: PairwiseBatch, is_training):
    if model.model_name == "BPR":
        feed_dict = {
            model.user_indices: data.queries,
            model.item_indices_pos: data.item_pairs[0],
            model.item_indices_neg: data.item_pairs[1],
        }
    elif model.model_name == "RNN4Rec":
        feed_dict = {
            model.user_interacted_seq: data.seqs.interacted_seq,
            model.user_interacted_len: data.seqs.interacted_len,
            model.item_indices_pos: data.item_pairs[0],
            model.item_indices_neg: data.item_pairs[1],
        }
    else:
        raise ValueError("Only `BPR` and `RNN4Rec` use `bpr` loss in tf models")
    if hasattr(model, "is_training"):
        feed_dict.update({model.is_training: is_training})
    return feed_dict


def _pointwise_feed_dict(model, data: PointwiseBatch, is_training):
    feed_dict = {
        model.user_indices: data.users,
        model.item_indices: data.items,
        model.labels: data.labels,
    }
    if hasattr(model, "is_training"):
        feed_dict.update({model.is_training: is_training})
    if hasattr(model, "sparse") and model.sparse:
        feed_dict.update({model.sparse_indices: data.sparse_indices})
    if hasattr(model, "dense") and model.dense:
        feed_dict.update({model.dense_values: data.dense_values})
    if model.model_category == "sequence":
        feed_dict.update(
            {
                model.user_interacted_seq: data.seqs.interacted_seq,
                model.user_interacted_len: data.seqs.interacted_len,
            }
        )
    return feed_dict
