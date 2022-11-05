import multiprocessing

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


def sess_config(tf_sess_config=None):
    if not tf_sess_config:
        # Session config based on:
        # https://software.intel.com/content/www/us/en/develop/articles/tips-to-improve-performance-for-popular-deep-learning-frameworks-on-multi-core-cpus.html
        # https://github.com/tensorflow/tensorflow/blob/v2.10.0/tensorflow/core/protobuf/config.proto#L452
        tf_sess_config = {
            "intra_op_parallelism_threads": 0,
            "inter_op_parallelism_threads": 0,
            "allow_soft_placement": True,
            "device_count": {"CPU": multiprocessing.cpu_count()},
        }
        # os.environ["OMP_NUM_THREADS"] = f"{self.cpu_num}"

    config = tf.ConfigProto(**tf_sess_config)
    return tf.Session(config=config)
