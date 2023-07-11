import tensorflow

tf = tensorflow.compat.v1
tf.disable_v2_behavior()

TF_VERSION = tf.__version__


def get_tf_version(version):
    if version is not None:
        assert isinstance(version, str)
        return version
    else:
        return TF_VERSION
