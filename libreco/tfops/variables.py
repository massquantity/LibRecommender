from .version import tf


def var_list_by_name(names):
    assert isinstance(names, (list, tuple)), "names must be list or tuple"
    var_dict = dict()
    for name in names:
        matched_vars = [var for var in tf.trainable_variables() if name in var.name]
        var_dict[name] = matched_vars
    return var_dict


def modify_variable_names(model, trainable):
    user_var, item_var, sparse_var, dense_var = None, None, None, None
    manual_var = None
    if trainable:
        if hasattr(model, "user_variables"):
            user_var = [v + ":0" for v in model.user_variables]
        if hasattr(model, "item_variables"):
            item_var = [v + ":0" for v in model.item_variables]
        if hasattr(model, "sparse_variables"):
            sparse_var = [v + ":0" for v in model.sparse_variables]
        if hasattr(model, "dense_variables"):
            dense_var = [v + ":0" for v in model.dense_variables]

        manual_var = []
        if user_var is not None:
            manual_var.extend(user_var)
        if item_var is not None:
            manual_var.extend(item_var)
        if sparse_var is not None:
            manual_var.extend(sparse_var)
        if dense_var is not None:
            manual_var.extend(dense_var)

    else:
        if hasattr(model, "user_variables"):
            user_var = []
            for v in model.user_variables:
                user_var.append(v + "/Adam:0")
                user_var.append(v + "/Adam_1:0")
                user_var.append(v + "/Ftrl:0")
                user_var.append(v + "/Ftrl_1:0")
        if hasattr(model, "item_variables"):
            item_var = []
            for v in model.item_variables:
                item_var.append(v + "/Adam:0")
                item_var.append(v + "/Adam_1:0")
                item_var.append(v + "/Ftrl:0")
                item_var.append(v + "/Ftrl_1:0")
        if hasattr(model, "sparse_variables"):
            sparse_var = []
            for v in model.sparse_variables:
                sparse_var.append(v + "/Adam:0")
                sparse_var.append(v + "/Adam_1:0")
                sparse_var.append(v + "/Ftrl:0")
                sparse_var.append(v + "/Ftrl_1:0")
        if hasattr(model, "dense_variables"):
            dense_var = []
            for v in model.dense_variables:
                dense_var.append(v + "/Adam:0")
                dense_var.append(v + "/Adam_1:0")
                dense_var.append(v + "/Ftrl:0")
                dense_var.append(v + "/Ftrl_1:0")

    return user_var, item_var, sparse_var, dense_var, manual_var


def match_adam(v_tf, v_model):
    return v_tf.name.startswith(v_model + "/Adam:0") or v_tf.name.startswith(
        v_model + "/Adam_1:0"
    )
