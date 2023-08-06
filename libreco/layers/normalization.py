import numpy as np
import torch.linalg

from ..tfops import tf


def layer_normalization(inputs, reuse_layer=False, scope_name="layer_norm"):
    reuse = tf.AUTO_REUSE if reuse_layer else None
    with tf.variable_scope(scope_name, reuse=reuse):
        dim = inputs.get_shape().as_list()[-1]
        scale = tf.get_variable("scale", shape=[dim], initializer=tf.ones_initializer())
        bias = tf.get_variable("bias", shape=[dim], initializer=tf.zeros_initializer())
        mean = tf.reduce_mean(inputs, axis=-1, keepdims=True)
        variance = tf.reduce_mean(
            tf.squared_difference(inputs, mean), axis=-1, keepdims=True
        )
        outputs = (inputs - mean) * tf.rsqrt(variance + 1e-8)
        return outputs * scale + bias


def rms_norm(inputs, reuse_layer=False, scope_name="rms_norm"):
    """Root mean square layer normalization."""
    reuse = tf.AUTO_REUSE if reuse_layer else None
    with tf.variable_scope(scope_name, reuse=reuse):
        dim = inputs.get_shape().as_list()[-1]
        scale = tf.get_variable("scale", shape=[dim], initializer=tf.ones_initializer())
        mean_square = tf.reduce_mean(tf.square(inputs), axis=-1, keepdims=True)
        outputs = inputs * tf.rsqrt(mean_square + 1e-8)
        return outputs * scale


def normalize_embeds(*embeds, backend):
    normed_embeds = []
    for e in embeds:
        if backend == "tf":
            ne = tf.linalg.l2_normalize(e, axis=1)
        elif backend == "torch":
            norms = torch.linalg.norm(e, dim=1, keepdim=True)
            ne = e / norms
        else:
            norms = np.linalg.norm(e, axis=1, keepdims=True)
            ne = e / norms
        normed_embeds.append(ne)
    return normed_embeds[0] if len(embeds) == 1 else normed_embeds
