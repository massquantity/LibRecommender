import numpy as np

from .activation import gelu
from .attention import compute_causal_mask, compute_seq_mask, multi_head_attention
from .dense import tf_dense
from .normalization import layer_normalization
from ..tfops import tf


def transformer_encoder_layer(
    seqs, seq_lens, max_seq_len, num_heads, head_dim, embed_size
):
    """Encoder layer in transformer.

    positional encoding + global self-attention + feed forward network

    Parameters
    ----------
    seqs : tf.Tensor
        Shape: (batch_size, seq_len, embed_size)
    seq_lens : tf.Tensor
        Shape: (batch_size,)
    max_seq_len : int
        Max sequence length.
    num_heads : int
        Number of attention heads.
    head_dim : int
        Dimension of each attention head.
    embed_size : int
        Input and output dimension.

    Returns
    -------
    Output shape: (batch_size, seq_len, embed_size)
    """
    with tf.variable_scope("transformer_encoder"):
        att_mask = compute_seq_mask(seq_lens, max_seq_len)

        pe = positional_encoding(max_seq_len, embed_size)[tf.newaxis, :, :]
        # scale the inputs before adding pe
        scaling_factor = tf.math.sqrt(tf.cast(embed_size, dtype=tf.float32))
        att_inputs = seqs * scaling_factor + pe
        att_outputs = multi_head_attention(
            att_inputs, att_inputs, num_heads, head_dim, att_mask
        )
        att_outputs = att_inputs + att_outputs
        att_outputs = layer_normalization(att_outputs, scope_name="ln_att")

        ffn_inputs = att_outputs
        ffn_outputs = ffn(ffn_inputs, embed_size)
        ffn_outputs = ffn_inputs + ffn_outputs
        ffn_outputs = layer_normalization(ffn_outputs, scope_name="ln_ffn")
        return ffn_outputs


def transformer_decoder_layer(
    encoder_out, seqs, seq_lens, max_seq_len, num_heads, head_dim, embed_size
):
    """Transformer decoder layer.

    positional encoding + causal self-attention + cross encoder attention + feed forward network

    Parameters
    ----------
    encoder_out : tf.Tensor
        Output from encoder layer, shape: (batch_size, seq_len_encoder, embed_size)
    seqs : tf.Tensor
        Shape: (batch_size, seq_len, embed_size)
    seq_lens : tf.Tensor
        Shape: (batch_size,)
    max_seq_len : int
        Max sequence length.
    num_heads : int
        Number of attention heads.
    head_dim : int
        Dimension of each attention head.
    embed_size : int
        Input and output dimension.

    Returns
    -------
    Output shape: (batch_size, seq_len, embed_size)
    """
    with tf.variable_scope("transformer_decoder"):
        seq_mask = compute_seq_mask(seq_lens, max_seq_len)
        causal_mask = compute_causal_mask(tf.shape(seqs)[0], max_seq_len)
        att_mask = seq_mask & causal_mask

        pe = positional_encoding(max_seq_len, embed_size)[tf.newaxis, :, :]
        # scale the inputs before adding pe
        scaling_factor = tf.math.sqrt(tf.cast(embed_size, dtype=tf.float32))
        cau_att_inputs = seqs * scaling_factor + pe
        cau_att_outputs = multi_head_attention(
            cau_att_inputs, cau_att_inputs, num_heads, head_dim, att_mask
        )
        cau_att_outputs = cau_att_inputs + cau_att_outputs
        cau_att_outputs = layer_normalization(cau_att_outputs, scope_name="ln_cau_att")

        enc_att_inputs = cau_att_outputs
        enc_att_outputs = multi_head_attention(
            enc_att_inputs, encoder_out, num_heads, head_dim, seq_mask
        )
        enc_att_outputs = enc_att_inputs + enc_att_outputs
        enc_att_outputs = layer_normalization(enc_att_outputs, scope_name="ln_enc_att")

        ffn_inputs = enc_att_outputs
        ffn_outputs = ffn(ffn_inputs, embed_size)
        ffn_outputs = ffn_inputs + ffn_outputs
        ffn_outputs = layer_normalization(ffn_outputs, scope_name="ln_ffn")
        return ffn_outputs


def positional_encoding(seq_len, d_model, trainable=False, scope_name="transformer"):
    """Positional encoding for providing sequence information in transformer.

    Parameters
    ----------
    seq_len : int
        Sequence length.
    d_model : int
        Input dimension.
    trainable : bool
        Whether the variable is trainable.
    scope_name : str

    Returns
    -------
    Output shape: (seq_len, d_model), can be added to input by expanding the first dimension.
    """
    with tf.variable_scope(scope_name, reuse=tf.AUTO_REUSE):
        pos = np.arange(seq_len)[:, np.newaxis]  # (seq, 1)
        dim = np.arange(d_model) / d_model
        dim[1::2] = dim[0::2] if d_model % 2 == 0 else dim[0::2][:-1]  # 2i+1 == 2i
        dim = dim[np.newaxis, :]  # (1, d_model)
        pe = pos / (10000**dim)
        pe[:, 0::2] = np.sin(pe[:, 0::2])
        pe[:, 1::2] = np.cos(pe[:, 1::2])
        pe_var = tf.get_variable(
            "positional_encoding",
            shape=(seq_len, d_model),
            initializer=tf.constant_initializer(pe, dtype=tf.float32),
            trainable=trainable,
        )
        return pe_var


def ffn(inputs, output_dim):
    """Feed forward network.

    Parameters
    ----------
    inputs : tf.Tensor
        Shape: (batch_size, seq_len, embed_size)
    output_dim : int
        Output model size.

    Returns
    -------
    Output shape: (batch_size, seq_len, embed_size)
    """
    outputs = tf_dense(output_dim * 4, activation=gelu, use_bias=False)(inputs)
    outputs = tf_dense(output_dim, use_bias=False)(outputs)
    return outputs
