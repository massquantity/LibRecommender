import os
import shutil
import sys

from libreco.bases import TfBase
from libreco.data import DataInfo
from libreco.data.data_info import EmptyFeature
from libreco.tfops import tf
from libreco.utils.misc import colorize

from .common import (
    check_path_exists,
    save_id_mapping,
    save_model_name,
    save_to_json,
    save_user_consumed,
)


def save_tf(path: str, model: TfBase, version: int = 1):
    """Save TF model to disk.

    Parameters
    ----------
    path : str
        Model saving path.
    model : TfBase
        Model to save.
    version : int, default: 1
        Version number used in ``tf.saved_model``.
    """
    check_path_exists(path)
    save_model_name(path, model)
    save_id_mapping(path, model.data_info)
    save_user_consumed(path, model.data_info)
    save_features(path, model.data_info, model)
    save_tf_serving_model(path, model, version)


def save_features(path: str, data_info: DataInfo, model: TfBase):
    feats = {"n_items": data_info.n_items}
    if data_info.col_name_mapping:
        if data_info.user_sparse_col != EmptyFeature:
            _check_num_match(data_info.user_sparse_unique, data_info.n_users)
            feats["user_sparse_col_index"] = data_info.user_sparse_col.index
            feats["user_sparse_values"] = data_info.user_sparse_unique[:-1].tolist()
        if data_info.item_sparse_col != EmptyFeature:
            _check_num_match(data_info.item_sparse_unique, data_info.n_items)
            feats["item_sparse_col_index"] = data_info.item_sparse_col.index
            feats["item_sparse_values"] = data_info.item_sparse_unique[:-1].tolist()
        if data_info.user_dense_col != EmptyFeature:
            _check_num_match(data_info.user_dense_unique, data_info.n_users)
            feats["user_dense_col_index"] = data_info.user_dense_col.index
            feats["user_dense_values"] = data_info.user_dense_unique[:-1].tolist()
        if data_info.item_dense_col != EmptyFeature:
            _check_num_match(data_info.item_dense_unique, data_info.n_items)
            feats["item_dense_col_index"] = data_info.item_dense_col.index
            feats["item_dense_values"] = data_info.item_dense_unique[:-1].tolist()

    if hasattr(model, "max_seq_len"):
        feats["max_seq_len"] = model.max_seq_len
    feature_path = os.path.join(path, "features.json")
    save_to_json(feature_path, feats)


# include oov
def _check_num_match(v, num):
    assert len(v) == num + 1, f"feature sizes don't match, got {len(v)} and {num + 1}"


def save_tf_serving_model(path: str, model: TfBase, version: int):
    model_name = model.model_name.lower()
    if not path:  # pragma: no cover
        model_base_path = os.path.realpath("..")
        export_path = os.path.join(
            model_base_path, "serving", "models", f"{model_name}", f"{version}"
        )
    else:
        export_path = os.path.join(path, f"{model_name}", f"{version}")

    if os.path.isdir(export_path):
        check_model_exists(export_path)

    builder = tf.saved_model.builder.SavedModelBuilder(export_path)
    inputs, outputs = build_inputs_outputs(model)
    prediction_signature = tf.saved_model.build_signature_def(
        inputs=inputs,
        outputs=outputs,
        method_name=tf.saved_model.signature_constants.PREDICT_METHOD_NAME,
    )
    builder.add_meta_graph_and_variables(
        sess=model.sess,
        tags=[tf.saved_model.tag_constants.SERVING],
        signature_def_map={"predict": prediction_signature},
        clear_devices=True,
    )

    builder.save()
    print(f"{colorize('Done tf exporting!', 'green', highlight=True)}")


def check_model_exists(export_path: str):  # pragma: no cover
    answered = False
    while not answered:
        print_str = (
            f"Could not export model because '{export_path}' "
            f"already exists, would you like to remove it? [Y/n]"
        )
        print(f"{colorize(print_str, 'red')}", end="")
        choice = input().lower()
        if choice in ["yes", "y"]:
            shutil.rmtree(export_path)
            answered = True
        elif choice in ["no", "n"]:
            print(f"{colorize('refused to remove, then exit...', 'red')}")
            sys.exit(0)


# noinspection PyUnresolvedReferences
def build_inputs_outputs(model: TfBase):
    input_dict = {
        "user_indices": tf.saved_model.build_tensor_info(model.user_indices),
        "item_indices": tf.saved_model.build_tensor_info(model.item_indices),
    }
    if hasattr(model, "sparse"):
        input_dict.update(
            {"sparse_indices": tf.saved_model.build_tensor_info(model.sparse_indices)}
        )
    if hasattr(model, "dense"):
        input_dict.update(
            {"dense_values": tf.saved_model.build_tensor_info(model.dense_values)}
        )
    if model.model_name in ("YouTubeRanking", "DIN"):
        input_dict.update(
            {
                "user_interacted_seq": tf.saved_model.build_tensor_info(
                    model.user_interacted_seq
                ),
                "user_interacted_len": tf.saved_model.build_tensor_info(
                    model.user_interacted_len
                ),
            }
        )
    output_dict = {"logits": tf.saved_model.build_tensor_info(model.output)}
    return input_dict, output_dict
