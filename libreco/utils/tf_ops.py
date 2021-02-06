from functools import partial
import numpy as np
import tensorflow as tf2
tf = tf2.compat.v1
tf.disable_v2_behavior()


# It turns out that the position of batch normalization layer matters in
# neural networks, see discussions in:
# https://stackoverflow.com/questions/39691902/ordering-of-batch-normalization-and-dropout
# https://www.zhihu.com/question/283715823
# Also according to the discussions, it is generally NOT recommended to use
# batch normalization and dropout simultaneously.
def dense_nn(net, hidden_units, activation=tf.nn.elu, use_bn=True,
             bn_after_activation=True, dropout_rate=None, is_training=True,
             name="mlp"):
    hidden_length = len(hidden_units)
    if activation is None:
        activation = tf.identity

    with tf.variable_scope(name):
        if use_bn:
            net = tf.layers.batch_normalization(net, training=is_training)
        for i, units in enumerate(hidden_units, start=1):
            # if i < hidden_length:
            net = tf.layers.dense(inputs=net,
                                  units=units,
                                  activation=None,
                                  name=name+"_layer"+str(i),
                                  reuse=tf.AUTO_REUSE)

            if use_bn and bn_after_activation:
                net = activation(net)
                net = tf.layers.batch_normalization(net, training=is_training)
            elif use_bn and not bn_after_activation:
                net = tf.layers.batch_normalization(net, training=is_training)
                net = activation(net)
            else:
                net = activation(net)

            if dropout_rate:
                net = tf.layers.dropout(net, dropout_rate,
                                        training=is_training)

        #    else:
        #        net = tf.layers.dense(inputs=net,
        #                              units=units,
        #                              activation=activation)

    return net


def var_list_by_name(names):
    assert isinstance(names, (list, tuple)), "names must be list or tuple"
    var_dict = dict()
    for name in names:
        matched_vars = [
            var for var in tf.trainable_variables() if name in var.name
        ]
        var_dict[name] = matched_vars
    return var_dict


def reg_config(reg):
    if not reg:
        return None
    elif isinstance(reg, float) and reg > 0.0:
        return tf.keras.regularizers.l2(reg)
    else:
        raise ValueError("reg must be float and positive...")


def dropout_config(dropout_rate):
    if not dropout_rate:
        return 0.0
    elif dropout_rate <= 0.0 or dropout_rate >= 1.0:
        raise ValueError("dropout_rate must be in (0.0, 1.0)")
    else:
        return dropout_rate


def lr_decay_config(initial_lr, default_decay_steps, **kwargs):
    decay_steps = kwargs.get("decay_steps", default_decay_steps)
    decay_rate = kwargs.get("decay_rate", 0.96)
    global_steps = tf.Variable(0, trainable=False, name="global_steps")
    learning_rate = tf.train.exponential_decay(initial_lr, global_steps,
                                               decay_steps, decay_rate,
                                               staircase=True)

    return learning_rate, global_steps


def sparse_tensor_interaction(data, recent_num=None, random_sample_rate=None):
    sparse_data = data.sparse_interaction.tocoo()
    row = sparse_data.row.reshape(-1, 1)
    indices = np.concatenate([row, np.zeros_like(row)], axis=1)
    values = sparse_data.col

#    user_interacted_num = np.diff(data.sparse_interaction.indptr)
    if recent_num is not None:
        indices, values = user_recent_interact(recent_num, indices, values)
    elif random_sample_rate is not None:
        indices, values = random_sample(random_sample_rate, indices, values)

    sparse_tensor = tf.SparseTensor(
        indices=indices, values=values, dense_shape=sparse_data.shape)
    return sparse_tensor


def random_sample(sample_rate, indices, values):
    assert 0.0 < sample_rate < 1.0, "sample_rate must be in (0.0, 1.0)"
    total_length = len(values)
    sample_num = int(total_length * sample_rate)
    sampled_indices = np.random.choice(
        range(total_length), size=sample_num, replace=False)
    indices = indices[sampled_indices]
    values = values[sampled_indices]
    return indices, values


def user_recent_interact(num, indices, values):
    assert isinstance(num, int), "recent_interact_num must be int"
    (users,
     user_position,
     user_counts) = np.unique(indices[:, 0],
                              return_inverse=True,
                              return_counts=True)

    user_split_indices = np.split(
        np.argsort(user_position, kind="mergesort"),
        np.cumsum(user_counts)[:-1]
    )

    n_users = len(users)
    recent_indices = list()
    for u in range(n_users):
        # assume user interactions have already been sorted by time.
        u_data = user_split_indices[u][-num:]
        recent_indices.extend(u_data)
    indices = indices[recent_indices]
    values = values[recent_indices]
    return indices, values


def conv_nn(tf_version, filters, kernel_size, strides, padding, activation,
            dilation_rate=1):
    if tf_version >= "2.0.0":
        net = tf.keras.layers.Conv1D(
            filters=filters,
            kernel_size=kernel_size,
            strides=strides,
            padding=padding,
            activation=activation,
            dilation_rate=dilation_rate
        )
    else:
        net = partial(
            tf.layers.conv1d,
            filters=filters,
            kernel_size=kernel_size,
            strides=strides,
            padding=padding,
            activation=activation
        )
    return net


def max_pool(tf_version, pool_size, strides, padding):
    if tf_version >= "2.0.0":
        net = tf.keras.layers.MaxPool1D(
            pool_size=pool_size,
            strides=strides,
            padding=padding
        )
    else:
        net = partial(
            tf.layers.max_pooling1d,
            pool_size=pool_size,
            strides=strides,
            padding=padding
        )
    return net


def match_adam(v_tf, v_model):
    return (v_tf.name.startswith(v_model + "/Adam:0") or
            v_tf.name.startswith(v_model + "/Adam_1:0"))


def modify_variable_names(model, trainable):
    user_var, item_var, sparse_var, dense_var = None, None, None, None
    manual_var = None
    if trainable:
        if hasattr(model, "user_variables"):
            user_var = [v+":0" for v in model.user_variables]
        if hasattr(model, "item_variables"):
            item_var = [v+":0" for v in model.item_variables]
        if hasattr(model, "sparse_variables"):
            sparse_var = [v+":0" for v in model.sparse_variables]
        if hasattr(model, "dense_variables"):
            dense_var = [v+":0" for v in model.dense_variables]

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
