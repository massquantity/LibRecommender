"""Implementation of Transformer."""
import numpy as np

from ..bases import ModelMeta, TfBase
from ..batch.sequence import get_recent_seqs
from ..layers import (
    compute_causal_mask,
    compute_seq_mask,
    dense_nn,
    embedding_lookup,
    ffn,
    layer_normalization,
    multi_head_attention,
    tf_attention,
    tf_dense,
)
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
    """*Transformer* algorithm."""

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
        n_epochs=20,
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
        self.use_causal_mask = use_causal_mask
        self.feat_agg_mode = feat_agg_mode
        self.seq_mode, self.max_seq_len = check_seq_mode(recent_num, random_num)
        self.recent_seqs, self.recent_seq_lens = get_recent_seqs(
            self.n_users,
            self.user_consumed,
            self.n_items,
            self.max_seq_len,
            dtype=np.float32,
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

        item_seq_feats = combine_seq_features(self.data_info, self.feat_agg_mode)
        seq_embeds = self._build_seq_repr(item_seq_feats)
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
        self.user_interacted_len = tf.placeholder(tf.float32, shape=[None])
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

    def _build_seq_repr(self, item_seq_feats):
        # B * K
        item_embeds = tf.nn.embedding_lookup(item_seq_feats, self.item_indices)
        # B * seq * K
        seq_embeds = tf.nn.embedding_lookup(item_seq_feats, self.user_interacted_seq)

        tfm_mask = self._transformer_mask(tf.shape(seq_embeds)[0])
        output_dim = item_embeds.get_shape().as_list()[-1]
        assert output_dim % self.num_heads == 0, (
            f"`item_dim`({output_dim}) should be divisible by `num_heads`({self.num_heads})"
        )  # fmt: skip
        head_dim = output_dim // self.num_heads
        for i in range(self.num_tfm_layers):
            seq_embeds = self._transformer_layer(
                seq_embeds, i, head_dim, tfm_mask, output_dim
            )

        att_mask = tf.sequence_mask(self.user_interacted_len, self.max_seq_len)
        return tf_attention(item_embeds, seq_embeds, att_mask)

    def _transformer_layer(self, x, i, head_dim, mask, output_dim):
        with tf.variable_scope(f"transformer_layer{i+1}"):
            att_outputs = multi_head_attention(
                x, x, self.num_heads, head_dim, mask, output_dim
            )
            x = layer_normalization(x + att_outputs, scope_name="ln_att")

            ffn_outputs = ffn(x, output_dim)
            return layer_normalization(x + ffn_outputs, scope_name="ln_ffn")

    def _transformer_mask(self, batch_size):
        tfm_mask = compute_seq_mask(
            self.user_interacted_len, self.max_seq_len, self.num_heads
        )
        if self.use_causal_mask:
            tfm_mask = tfm_mask & compute_causal_mask(
                batch_size, self.max_seq_len, self.num_heads
            )
        return tfm_mask
