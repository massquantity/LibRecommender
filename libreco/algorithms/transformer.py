"""Implementation of Transformer."""
from ..bases import ModelMeta, TfBase
from ..batch.sequence import get_recent_seqs
from ..layers import (
    compute_causal_mask,
    compute_seq_mask,
    dense_nn,
    embedding_lookup,
    multi_head_attention,
    rms_norm,
    tf_attention,
    tf_dense,
)
from ..layers.activation import swish
from ..layers.transformer import ffn, positional_encoding
from ..tfops import dropout_config, reg_config, tf
from ..tfops.features import (
    combine_seq_features,
    compute_dense_feats,
    compute_sparse_feats,
)
from ..torchops import hidden_units_config
from ..utils.misc import count_params
from ..utils.validate import (
    check_dense_values,
    check_multi_sparse,
    check_seq_mode,
    check_sparse_indices,
    dense_field_size,
    sparse_feat_size,
    sparse_field_size,
)


class Transformer(TfBase, metaclass=ModelMeta):
    """*Transformer* algorithm.

    Parameters
    ----------
    task : {'rating', 'ranking'}
        Recommendation task. See :ref:`Task`.
    data_info : :class:`~libreco.data.DataInfo` object
        Object that contains useful information for training and inference.
    loss_type : {'cross_entropy', 'focal'}, default: 'cross_entropy'
        Loss for model training.
    embed_size: int, default: 16
        Vector size of embeddings.
    n_epochs: int, default: 1
        Number of epochs for training.
    lr : float, default 0.001
        Learning rate for training.
    lr_decay : bool, default: False
        Whether to use learning rate decay.
    epsilon : float, default: 1e-5
        A small constant added to the denominator to improve numerical stability in
        Adam optimizer.
        According to the `official comment <https://github.com/tensorflow/tensorflow/blob/v1.15.0/tensorflow/python/training/adam.py#L64>`_,
        default value of `1e-8` for `epsilon` is generally not good, so here we choose `1e-5`.
        Users can try tuning this hyperparameter if the training is unstable.
    reg : float or None, default: None
        Regularization parameter, must be non-negative or None.
    batch_size : int, default: 256
        Batch size for training.
    sampler : {'random', 'unconsumed', 'popular'}, default: 'random'
        Negative sampling strategy.

        - ``'random'`` means random sampling.
        - ``'unconsumed'`` samples items that the target user did not consume before.
        - ``'popular'`` has a higher probability to sample popular items as negative samples.

    num_neg : int, default: 1
        Number of negative samples for each positive sample, only used in `ranking` task.
    use_bn : bool, default: True
        Whether to use batch normalization in MLP layers.
    dropout_rate : float or None, default: None
        Probability of an element to be zeroed. If it is None, dropout is not used.
    hidden_units : int, list of int or tuple of (int,), default: (128, 64, 32)
        Number of layers and corresponding layer size in MLP.
    recent_num : int or None, default: 10
        Number of recent items to use in user behavior sequence.
    random_num : int or None, default: None
        Number of random sampled items to use in user behavior sequence.
        If `recent_num` is not None, `random_num` is not considered.
    num_heads : int, default: 1
        Number of heads in multi-head attention.
    num_tfm_layers : int, default: 1
        Number of transformer layers.
    positional_embedding : {'trainable', 'sinusoidal'}, default: 'trainable'
        Positional embedding used in transformer layers.
    use_causal_mask : bool, default: False
        Whether to apply causal mask. Causal mask will only attend items before current item,
        which is used in transformer decoder.
    feat_agg_mode : {'concat', 'elementwise'}, default: 'concat'
        Options for aggregating item features used in sequence attention.

        - ``'concat'`` stands for concatenating all the item features.
        - ``'elementwise'`` stands for element-wise merge described in Reference[2].
          In this case, all item features must have same embed size.

    multi_sparse_combiner : {'normal', 'mean', 'sum', 'sqrtn'}, default: 'sqrtn'
        Options for combining `multi_sparse` features.
    seed : int, default: 42
        Random seed.
    lower_upper_bound : tuple or None, default: None
        Lower and upper score bound for `rating` task.
    tf_sess_config : dict or None, default: None
        Optional TensorFlow session config, see `ConfigProto options
        <https://github.com/tensorflow/tensorflow/blob/v2.10.0/tensorflow/core/protobuf/config.proto#L431>`_.

    References
    ----------
    [1] *Qiwei Chen et al.* `Behavior Sequence Transformer for E-commerce Recommendation in Alibaba
    <https://arxiv.org/pdf/1905.06874.pdf>`_.

    [2] *Gabriel de Souza Pereira et al.* `Transformers4Rec: Bridging the Gap between NLP and Sequential / Session-Based Recommendation
    <https://dl.acm.org/doi/10.1145/3460231.3474255>`_.

    [3] *Biao Zhang & Rico Sennrich.* `Root Mean Square Layer Normalization
    <https://arxiv.org/pdf/1910.07467.pdf>`_.
    """

    user_variables = ("embedding/user_embeds_var",)
    item_variables = ("embedding/item_embeds_var",)
    sparse_variables = ("embedding/sparse_embeds_var",)
    dense_variables = ("embedding/dense_embeds_var",)

    def __init__(
        self,
        task,
        data_info=None,
        loss_type="cross_entropy",
        embed_size=16,
        n_epochs=1,
        lr=0.001,
        lr_decay=False,
        epsilon=1e-5,
        reg=None,
        batch_size=256,
        sampler="random",
        num_neg=1,
        use_bn=True,
        dropout_rate=None,
        hidden_units=(128, 64, 32),
        recent_num=10,
        random_num=None,
        num_heads=1,
        num_tfm_layers=1,
        positional_embedding="trainable",
        use_causal_mask=False,
        feat_agg_mode="concat",
        multi_sparse_combiner="sqrtn",
        seed=42,
        lower_upper_bound=None,
        tf_sess_config=None,
    ):
        super().__init__(task, data_info, lower_upper_bound, tf_sess_config)

        self.all_args = locals()
        self.loss_type = loss_type
        self.embed_size = embed_size
        self.n_epochs = n_epochs
        self.lr = lr
        self.lr_decay = lr_decay
        self.epsilon = epsilon
        self.reg = reg_config(reg)
        self.batch_size = batch_size
        self.sampler = sampler
        self.num_neg = num_neg
        self.use_bn = use_bn
        self.dropout_rate = dropout_config(dropout_rate)
        self.hidden_units = hidden_units_config(hidden_units)
        self.num_heads = num_heads
        self.num_tfm_layers = num_tfm_layers
        self.positional_embedding = positional_embedding
        self.use_causal_mask = use_causal_mask
        self.feat_agg_mode = feat_agg_mode
        self.seq_mode, self.max_seq_len = check_seq_mode(recent_num, random_num)
        self.recent_seqs, self.recent_seq_lens = get_recent_seqs(
            self.n_users,
            self.user_consumed,
            self.n_items,
            self.max_seq_len,
        )
        self.seed = seed
        self.sparse = check_sparse_indices(data_info)
        self.dense = check_dense_values(data_info)
        if self.sparse:
            self.sparse_feature_size = sparse_feat_size(data_info)
            self.sparse_field_size = sparse_field_size(data_info)
            self.multi_sparse_combiner = check_multi_sparse(
                data_info, multi_sparse_combiner
            )
        if self.dense:
            self.dense_field_size = dense_field_size(data_info)
        self._check_params()

    def _check_params(self):
        if self.task == "ranking" and self.loss_type not in ("cross_entropy", "focal"):
            raise ValueError(f"unsupported `loss_type`: {self.loss_type}")
        if self.feat_agg_mode not in ("concat", "elementwise"):
            raise ValueError("`feat_agg_mode` must be `concat` or `elementwise`.")

    def build_model(self):
        tf.set_random_seed(self.seed)
        self._build_placeholders()
        user_embed = embedding_lookup(
            indices=self.user_indices,
            var_name="user_embeds_var",
            var_shape=(self.n_users + 1, self.embed_size),
            initializer=tf.glorot_uniform_initializer(),
            regularizer=self.reg,
        )
        item_embed = embedding_lookup(
            indices=self.item_indices,
            var_name="item_embeds_var",
            var_shape=(self.n_items + 1, self.embed_size),
            initializer=tf.glorot_uniform_initializer(),
            regularizer=self.reg,
        )
        concat_embeds = [user_embed, item_embed]

        if self.sparse:
            sparse_embed = compute_sparse_feats(
                self.data_info,
                self.multi_sparse_combiner,
                self.sparse_indices,
                var_name="sparse_embeds_var",
                var_shape=(self.sparse_feature_size, self.embed_size),
                initializer=tf.glorot_uniform_initializer(),
                regularizer=self.reg,
                flatten=True,
            )
            concat_embeds.append(sparse_embed)
        if self.dense:
            dense_embed = compute_dense_feats(
                self.dense_values,
                var_name="dense_embeds_var",
                var_shape=(self.dense_field_size, self.embed_size),
                initializer=tf.glorot_uniform_initializer(),
                regularizer=self.reg,
                flatten=True,
            )
            concat_embeds.append(dense_embed)

        self.seq_feats = combine_seq_features(self.data_info, self.feat_agg_mode)
        seq_embeds = self._build_seq_repr()
        dense_inputs = tf.concat([*concat_embeds, seq_embeds], axis=1)
        mlp_layer = dense_nn(
            dense_inputs,
            self.hidden_units,
            activation=swish,
            use_bn=self.use_bn,
            dropout_rate=self.dropout_rate,
            is_training=self.is_training,
            name="mlp",
        )
        self.output = tf.reshape(tf_dense(units=1)(mlp_layer), [-1])
        self.serving_topk = self.build_topk(self.output)
        count_params()

    def _build_placeholders(self):
        self.user_indices = tf.placeholder(tf.int32, shape=[None])
        self.item_indices = tf.placeholder(tf.int32, shape=[None])
        self.user_interacted_seq = tf.placeholder(
            tf.int32, shape=[None, self.max_seq_len]
        )
        self.user_interacted_len = tf.placeholder(tf.int32, shape=[None])
        self.labels = tf.placeholder(tf.float32, shape=[None])
        self.is_training = tf.placeholder_with_default(False, shape=[])

        if self.sparse:
            self.sparse_indices = tf.placeholder(
                tf.int32, shape=[None, self.sparse_field_size]
            )
        if self.dense:
            self.dense_values = tf.placeholder(
                tf.float32, shape=[None, self.dense_field_size]
            )

    def _build_seq_repr(self):
        # B * K
        item_embeds = tf.nn.embedding_lookup(self.seq_feats, self.item_indices)
        # B * seq * K
        seq_embeds = tf.nn.embedding_lookup(self.seq_feats, self.user_interacted_seq)

        # item feature dim + position dim
        output_dim = item_embeds.get_shape().as_list()[-1] + self.embed_size
        assert output_dim % self.num_heads == 0, (
            f"`item_dim`({output_dim}) should be divisible by `num_heads`({self.num_heads})"
        )  # fmt: skip

        batch_size = tf.shape(seq_embeds)[0]
        pos_embeds = self._positional_embedding(batch_size, self.embed_size)
        seq_embeds = tf.concat([seq_embeds, pos_embeds], axis=2)
        tfm_mask = self._transformer_mask(batch_size)
        head_dim = output_dim // self.num_heads
        for layer in range(self.num_tfm_layers):
            seq_embeds = self._transformer_layer(
                seq_embeds, layer, head_dim, tfm_mask, output_dim
            )
        seq_embeds = rms_norm(seq_embeds, scope_name="rms_norm_last")

        item_embeds = rms_norm(item_embeds, scope_name="rms_norm_item")
        item_pos_padding = tf.ones(shape=(batch_size, self.embed_size))
        item_embeds = tf.concat([item_embeds, item_pos_padding], axis=1)
        att_mask = tf.sequence_mask(self.user_interacted_len, self.max_seq_len)
        return tf_attention(item_embeds, seq_embeds, att_mask)

    def _transformer_layer(self, inputs, layer, head_dim, mask, output_dim):
        with tf.variable_scope(f"transformer_layer{layer+1}"):
            x = rms_norm(inputs, scope_name="rms_norm_att")
            att_out = (
                multi_head_attention(x, x, self.num_heads, head_dim, mask, output_dim)
                + inputs
            )

            x = rms_norm(att_out, scope_name="rms_norm_ffn")
            ffn_out = ffn(x, output_dim)
            return att_out + ffn_out

    def _transformer_mask(self, batch_size):
        tfm_mask = compute_seq_mask(self.user_interacted_len, self.max_seq_len)
        if self.use_causal_mask:
            causal_mask = compute_causal_mask(batch_size, self.max_seq_len)
            tfm_mask = tf.logical_and(tfm_mask, causal_mask)
        return tfm_mask

    def _positional_embedding(self, batch_size, dim):
        if self.positional_embedding in ("sinusoidal", "sin", "sinusoid"):
            pos_embeds = positional_encoding(self.max_seq_len, dim, trainable=False)
        else:
            with tf.variable_scope("transformer", reuse=tf.AUTO_REUSE):
                pos_embeds = tf.get_variable(
                    "positional_encoding",
                    shape=(self.max_seq_len, dim),
                    initializer=tf.glorot_uniform_initializer(),
                    trainable=True,
                )
        return tf.tile(pos_embeds[tf.newaxis, :, :], (batch_size, 1, 1))
