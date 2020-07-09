from collections import defaultdict
import json
import os
import shutil
import sys
import numpy as np
import tensorflow as tf
from .misc import colorize


def convert_similarity_to_json(sim_csr_matrix, k=20):
    res = dict()
    num = len(sim_csr_matrix.indptr) - 1
    indices = sim_csr_matrix.indices.tolist()
    indptr = sim_csr_matrix.indptr.tolist()
    data = sim_csr_matrix.data.tolist()
    for i in range(num):
        i_slice = slice(indptr[i], indptr[i+1])
        res[i] = sorted(zip(indices[i_slice], data[i_slice]),
                        key=lambda x: -x[1])[:k]
    return res


def convert_vector_to_json(vec):
    res = dict()
    for i, v in enumerate(vec):
        res[i] = vec[i].tolist()
    return res


def convert_user_consumed_to_json(sparse_interacted_matrix):
    consumed = defaultdict(dict)
    num = len(sparse_interacted_matrix.indptr) - 1
    indices = sparse_interacted_matrix.indices.tolist()
    indptr = sparse_interacted_matrix.indptr.tolist()
    data = sparse_interacted_matrix.data.tolist()
    for u in range(num):
        user_slice = slice(indptr[u], indptr[u+1])
        for i, r in zip(indices[user_slice], data[user_slice]):
            consumed[u][i] = r
    return consumed


def convert_data_info_to_json(data_info):
    res = dict()

    # sparse part
    user_sparse_col = data_info.user_sparse_col.index
    item_sparse_col = data_info.item_sparse_col.index
    if user_sparse_col and item_sparse_col:
        orig_cols = user_sparse_col + item_sparse_col
        # keep column names in original order
        sparse_col_reindex = np.arange(len(orig_cols))[np.argsort(orig_cols)]
        res["sparse_col_reindex"] = sparse_col_reindex.tolist()
        res["user_sparse_unique"] = data_info.user_sparse_unique.tolist()
        res["item_sparse_unique"] = data_info.item_sparse_unique.tolist()
    elif user_sparse_col:
        res["user_sparse_unique"] = data_info.user_sparse_unique.tolist()
    elif item_sparse_col:
        res["item_sparse_unique"] = data_info.item_sparse_unique.tolist()

    # dense part
    user_dense_col = data_info.user_dense_col.index
    item_dense_col = data_info.item_dense_col.index
    if user_dense_col and item_dense_col:
        orig_cols = user_dense_col + item_dense_col
        # keep column names in original order
        dense_col_reindex = np.arange(len(orig_cols))[np.argsort(orig_cols)]
        res["dense_col_reindex"] = dense_col_reindex.tolist()
        res["user_dense_unique"] = data_info.user_dense_unique.tolist()
        res["item_dense_unique"] = data_info.item_dense_unique.tolist()
    elif user_dense_col:
        res["user_dense_unique"] = data_info.user_dense_unique.tolist()
    elif item_dense_col:
        res["item_dense_unique"] = data_info.item_dense_unique.tolist()

    return res


def save_to_json(path, data, convert_func):
    json_data = convert_func(data)
    with open(path, 'w') as f:
        json.dump(json_data, f, separators=(',', ':'))


def save_model_tf_serving(path, model, model_name, version, simple_save=False):
    if not path:
        model_base_path = os.path.realpath("..")
        export_path = os.path.join(model_base_path,
                                   f"serving/models/{model_name}/{version}")
    else:
        export_path = os.path.join(path, f"{model_name}/{version}")

    if os.path.isdir(export_path):
        answered = False
        while not answered:
            print_str = (f"Could not export model because '{export_path}' "
                         f"already exists, would you like to remove it? [Y/n]")
            print(f"{colorize(print_str, 'red')}", end='')
            choice = input().lower()
            if choice in ["yes", "y"]:
                shutil.rmtree(export_path)
                answered = True
            elif choice in ["no", "n"]:
                answered = True
                print(f"{colorize('refused to remove, then exit...', 'red')}")
                sys.exit(0)

    if simple_save:
        input_dict = {"user_indices": model.user_indices,
                      "item_indices": model.item_indices}
        if model.sparse:
            input_dict.update({"sparse_indices": model.sparse_indices})
        if model.dense:
            input_dict.update({"dense_values": model.dense_values})
        output_dict = {"logits": model.output}
        tf.saved_model.simple_save(model.sess,
                                   export_path,
                                   inputs=input_dict,
                                   outputs=output_dict)

    else:
        builder = tf.saved_model.builder.SavedModelBuilder(export_path)
        input_dict = {
            "user_indices": tf.saved_model.build_tensor_info(
                model.user_indices),
            "item_indices": tf.saved_model.build_tensor_info(
                model.item_indices)
        }
        if model.sparse:
            input_dict.update(
                {"sparse_indices": tf.saved_model.build_tensor_info(
                    model.sparse_indices)}
            )
        if model.dense:
            input_dict.update(
                {"dense_values": tf.saved_model.build_tensor_info(
                    model.dense_values)}
            )
        output_dict = {
            "logits": tf.saved_model.build_tensor_info(model.output)
        }

        prediction_signature = tf.saved_model.build_signature_def(
            inputs=input_dict, outputs=output_dict,
            method_name=tf.saved_model.signature_constants.PREDICT_METHOD_NAME
        )

        builder.add_meta_graph_and_variables(
            model.sess, [tf.saved_model.tag_constants.SERVING],
            signature_def_map={"predict": prediction_signature})

        builder.save()
        print(f"{colorize('Done tf exporting!', 'green')}")

