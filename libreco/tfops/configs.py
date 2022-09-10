from .version import tf


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
    learning_rate = tf.train.exponential_decay(
        initial_lr, global_steps, decay_steps, decay_rate, staircase=True
    )
    return learning_rate, global_steps
