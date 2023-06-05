import os
from typing import Union

import numpy as np

from libreco.bases import DynEmbedBase, TfBase
from libreco.data import DataInfo
from libreco.tfops import tf
from libreco.utils.constants import UserEmbedModels
from libreco.utils.misc import colorize

from .common import (
    check_model_exists,
    check_path_exists,
    save_features,
    save_id_mapping,
    save_model_name,
    save_to_json,
    save_user_consumed,
)


def save_online(path: str, model: Union[DynEmbedBase, TfBase], version: int = 1):
    """Save online computing model to disk.

    Parameters
    ----------
    path : str
        Model saving path.
    model : Union[DynEmbedBase, TfBase]
        Model to save.
    version : int, default: 1
        Version number used in ``tf.saved_model``.
    """
    check_path_exists(path)
    save_model_name(path, model)
    save_id_mapping(path, model.data_info)
    save_user_consumed(path, model.data_info)
    save_features(path, model.data_info, model)
    save_user_sparse_mapping(path, model.data_info)
    save_user_dense_mapping(path, model.data_info)
    save_tf_serving_model(path, model, version)


def save_user_sparse_mapping(path: str, data_info: DataInfo):
    user_sparse_fields, user_sparse_idx_mapping = dict(), dict()
    col_mapping = data_info.col_name_mapping
    user_sparse_cols = data_info.user_sparse_col.name
    if col_mapping and user_sparse_cols:
        sparse_idx_mapping = data_info.sparse_idx_mapping  # {col: {val: idx}}
        for field_idx, col in enumerate(user_sparse_cols):
            user_sparse_fields[col] = field_idx
            if "multi_sparse" in col_mapping and col in col_mapping["multi_sparse"]:
                main_col = col_mapping["multi_sparse"][col]
                idx_mapping = sparse_idx_mapping[main_col]
            else:
                idx_mapping = sparse_idx_mapping[col]

            all_field_idx = col_mapping["sparse_col"][col]
            feat_offset = data_info.sparse_offset[all_field_idx]
            val_idx_mapping = dict()
            for val, idx in idx_mapping.items():
                val = val.item() if isinstance(val, np.integer) else val
                idx = int(idx + feat_offset)
                val_idx_mapping[val] = idx
            user_sparse_idx_mapping.update({col: val_idx_mapping})

        save_to_json(os.path.join(path, "user_sparse_fields.json"), user_sparse_fields)
        save_to_json(
            os.path.join(path, "user_sparse_idx_mapping.json"), user_sparse_idx_mapping
        )


def save_user_dense_mapping(path: str, data_info: DataInfo):
    user_dense_fields = dict()
    col_mapping = data_info.col_name_mapping
    user_dense_cols = data_info.user_dense_col.name
    if col_mapping and user_dense_cols:
        for field_idx, col in enumerate(user_dense_cols):
            user_dense_fields[col] = field_idx

        save_to_json(os.path.join(path, "user_dense_fields.json"), user_dense_fields)


def save_tf_serving_model(path: str, model: Union[DynEmbedBase, TfBase], version: int):
    model_name = model.model_name.lower()
    export_path = os.path.join(path, f"{model_name}", f"{version}")
    if os.path.isdir(export_path):
        check_model_exists(export_path)

    builder = tf.saved_model.builder.SavedModelBuilder(export_path)
    inputs, outputs = build_inputs_outputs(model)
    topk_signature = tf.saved_model.build_signature_def(
        inputs=inputs,
        outputs=outputs,
        method_name=tf.saved_model.signature_constants.PREDICT_METHOD_NAME,
    )
    builder.add_meta_graph_and_variables(
        sess=model.sess,
        tags=[tf.saved_model.tag_constants.SERVING],
        signature_def_map={"topk": topk_signature},
        clear_devices=True,
        strip_default_attrs=True,
    )

    builder.save()
    print(f"\n{colorize('Done tf exporting!', 'green', highlight=True)}\n")


def build_inputs_outputs(model):
    build_tensor = tf.saved_model.build_tensor_info
    model_name = model.model_name
    input_dict = {"k": build_tensor(model.k)}
    if hasattr(model, "user_indices"):
        input_dict.update({"user_indices": build_tensor(model.user_indices)})
    # `UserEmbedModels` would use item embeds directly
    if hasattr(model, "item_indices") and not UserEmbedModels.contains(model_name):
        input_dict.update({"item_indices": build_tensor(model.item_indices)})
    if hasattr(model, "sparse_indices"):
        input_dict.update({"sparse_indices": build_tensor(model.sparse_indices)})
    if hasattr(model, "dense_values"):
        input_dict.update({"dense_values": build_tensor(model.dense_values)})
    if hasattr(model, "user_sparse_indices"):
        input_dict.update(
            {"user_sparse_indices": build_tensor(model.user_sparse_indices)}
        )
    if hasattr(model, "item_sparse_indices"):
        input_dict.update(
            {"item_sparse_indices": build_tensor(model.item_sparse_indices)}
        )
    if hasattr(model, "user_dense_values"):
        input_dict.update({"user_dense_values": build_tensor(model.user_dense_values)})
    if hasattr(model, "item_dense_values"):
        input_dict.update({"item_dense_values": build_tensor(model.item_dense_values)})
    if hasattr(model, "user_interacted_seq"):
        input_dict.update(
            {
                "user_interacted_seq": build_tensor(model.user_interacted_seq),
                "user_interacted_len": build_tensor(model.user_interacted_len),
            }
        )
    if hasattr(model, "item_interaction_indices"):
        input_dict.update(
            {
                "item_interaction_indices": build_tensor(model.item_interaction_indices),  # fmt: skip
                "item_interaction_values": build_tensor(model.item_interaction_values),
                "modified_batch_size": build_tensor(model.modified_batch_size),
            }
        )

    output_dict = {"topk": tf.saved_model.build_tensor_info(model.serving_topk)}
    return input_dict, output_dict
