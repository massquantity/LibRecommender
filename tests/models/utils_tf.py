from libreco.tfops import tf


def ptest_tf_variables(model):
    var_names = [v.name for v in tf.trainable_variables()]
    if hasattr(model, "user_variables"):
        for v in model.user_variables:
            assert f"{v}:0" in var_names
    if hasattr(model, "item_variables"):
        for v in model.item_variables:
            assert f"{v}:0" in var_names
    if hasattr(model, "sparse_variables"):
        for v in model.sparse_variables:
            assert f"{v}:0" in var_names
    if hasattr(model, "dense_variables"):
        for v in model.dense_variables:
            assert f"{v}:0" in var_names
