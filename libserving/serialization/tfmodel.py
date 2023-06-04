import os

from libreco.bases import TfBase
from libreco.tfops import tf
from libreco.utils.misc import colorize

from .common import (
    check_model_exists,
    check_path_exists,
    save_features,
    save_id_mapping,
    save_model_name,
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
        strip_default_attrs=True,
    )

    builder.save()
    print(f"\n{colorize('Done tf exporting!', 'green', highlight=True)}\n")


def build_inputs_outputs(model):
    input_dict = {
        "user_indices": tf.saved_model.build_tensor_info(model.user_indices),
        "item_indices": tf.saved_model.build_tensor_info(model.item_indices),
    }
    if hasattr(model, "sparse") and model.sparse:
        input_dict.update(
            {"sparse_indices": tf.saved_model.build_tensor_info(model.sparse_indices)}
        )
    if hasattr(model, "dense") and model.dense:
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
