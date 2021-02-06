from .column import (
    get_user_item_sparse_indices,
    merge_sparse_indices,
    merge_sparse_col,
    merge_offset,
    get_oov_pos,
    interaction_consumed
)
from .column_mapping import col_name2index
from .unique_features import (
    construct_unique_feat,
    get_predict_indices_and_values,
    get_recommend_indices_and_values,
    features_from_dict,
    features_from_batch_data,
    add_item_features,
    compute_sparse_feat_indices,
    _check_oov
)
