"""Implementation of DIN."""
import numpy as np

from ..bases import ModelMeta, TfBase
from ..batch.sequence import get_recent_seqs
from ..feature.multi_sparse import true_sparse_field_size
from ..layers import dense_nn, din_attention, embedding_lookup, tf_attention, tf_dense
from ..tfops import (
    compute_dense_feats,
    compute_sparse_feats,
    dropout_config,
    get_variable_from_graph,
    reg_config,
    tf,
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


class DIN(TfBase, metaclass=ModelMeta):
    """*Deep Interest Network* algorithm.

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
    n_epochs: int, default: 10
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

        .. versionadded:: 1.1.0

    num_neg : int, default: 1
        Number of negative samples for each positive sample, only used in `ranking` task.
    use_bn : bool, default: True
        Whether to use batch normalization.
    dropout_rate : float or None, default: None
        Probability of an element to be zeroed. If it is None, dropout is not used.
    hidden_units : int, list of int or tuple of (int,), default: (128, 64, 32)
        Number of layers and corresponding layer size in MLP.

        .. versionchanged:: 1.0.0
           Accept type of ``int``, ``list`` or ``tuple``, instead of ``str``.

    recent_num : int or None, default: 10
        Number of recent items to use in user behavior sequence.
    random_num : int or None, default: None
        Number of random sampled items to use in user behavior sequence.
        If `recent_num` is not None, `random_num` is not considered.
    use_tf_attention : bool, default: False
        Whether to use TensorFlow's `attention <https://www.tensorflow.org/versions/r1.15/api_docs/python/tf/keras/layers/Attention>`_ implementation.
        The TensorFlow attention version is simpler and faster, but doesn't follow the
        settings in paper, whereas our implementation does.
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
    *Guorui Zhou et al.* `Deep Interest Network for Click-Through Rate Prediction
    <https://arxiv.org/pdf/1706.06978.pdf>`_.
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
        use_tf_attention=False,
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
        self.use_tf_attention = use_tf_attention
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
            self.true_sparse_field_size = true_sparse_field_size(
                data_info, self.sparse_field_size, self.multi_sparse_combiner
            )
        if self.dense:
            self.dense_field_size = dense_field_size(data_info)
        self.item_sparse = True if data_info.item_sparse_unique is not None else False
        self.item_dense = True if data_info.item_dense_unique is not None else False
        if self.item_sparse:
            # item sparse col indices in all sparse cols
            self.item_sparse_col_indices = data_info.item_sparse_col.index
        if self.item_dense:
            # item dense col indices in all dense cols
            self.item_dense_col_indices = data_info.item_dense_col.index

    def build_model(self):
        tf.set_random_seed(self.seed)
        self.concat_embed, self.item_embed, self.seq_embed = [], [], []
        self._build_placeholders()
        self._build_user_item()
        if self.sparse:
            self._build_sparse()
        if self.dense:
            self._build_dense()
        self._build_seq_attention()

        concat_embed = tf.concat(self.concat_embed, axis=1)
        mlp_layer = dense_nn(
            concat_embed,
            self.hidden_units,
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

    def _build_user_item(self):
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
        self.concat_embed.extend([user_embed, item_embed])
        self.item_embed.append(item_embed)

    def _build_sparse(self):
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
        self.concat_embed.append(sparse_embed)

        if self.item_sparse:
            all_sparse_embed = embedding_lookup(
                self.sparse_indices, var_name="sparse_embeds_var", reuse_layer=True
            )
            item_sparse_embed = tf.keras.layers.Flatten()(
                tf.gather(all_sparse_embed, self.item_sparse_col_indices, axis=1)
            )
            self.item_embed.append(item_sparse_embed)

    def _build_dense(self):
        dense_embed = compute_dense_feats(
            self.dense_values,
            var_name="dense_embeds_var",
            var_shape=(self.dense_field_size, self.embed_size),
            initializer=tf.glorot_uniform_initializer(),
            regularizer=self.reg,
            flatten=False,
        )
        self.concat_embed.append(tf.keras.layers.Flatten()(dense_embed))

        if self.item_dense:
            item_dense_embed = tf.keras.layers.Flatten()(
                tf.gather(dense_embed, self.item_dense_col_indices, axis=1)
            )
            self.item_embed.append(item_dense_embed)

    def _build_seq_attention(self):
        # B * seq * K
        seq_item_embed = embedding_lookup(
            indices=self.user_interacted_seq,
            var_name="item_embeds_var",  # variable must exist since previous initialized in `build_item`
            reuse_layer=True,
        )
        self.seq_embed.append(seq_item_embed)

        if self.item_sparse:
            # contains unique sparse field indices for each item
            item_sparse_fields = tf.convert_to_tensor(
                self.data_info.item_sparse_unique, dtype=tf.int64
            )
            item_sparse_fields_num = tf.shape(item_sparse_fields)[1]

            # B * seq * F_sparse
            seq_sparse_fields = tf.gather(item_sparse_fields, self.user_interacted_seq)
            # B * seq * F_sparse * K
            seq_sparse_embed = embedding_lookup(
                indices=seq_sparse_fields,
                var_name="sparse_embeds_var",  # variable must exist since previous initialized in `build_sparse`
                reuse_layer=True,
            )
            # B * seq * FK
            seq_sparse_embed = tf.reshape(
                seq_sparse_embed,
                [-1, self.max_seq_len, item_sparse_fields_num * self.embed_size],
            )
            self.seq_embed.append(seq_sparse_embed)

        if self.item_dense:
            # contains unique dense values for each item
            item_dense_values = tf.convert_to_tensor(
                self.data_info.item_dense_unique, dtype=tf.float32
            )
            item_dense_fields_num = tf.shape(item_dense_values)[1]
            # B * seq * F_dense
            seq_dense_values = tf.gather(item_dense_values, self.user_interacted_seq)
            # B * seq * F_dense * 1
            seq_dense_values = tf.expand_dims(seq_dense_values, axis=-1)

            batch_size = tf.shape(seq_dense_values)[0]
            dense_embeds_var = get_variable_from_graph("dense_embeds_var", "embedding")
            # 1 * 1 * F_dense * K
            dense_embed = tf.reshape(
                tf.gather(dense_embeds_var, self.item_dense_col_indices),
                [1, 1, item_dense_fields_num, self.embed_size],
            )
            # B * seq * F_dense * K
            # Since dense_embeddings are same for all items, we can simply repeat it (batch * seq) times
            seq_dense_embed = tf.tile(dense_embed, [batch_size, self.max_seq_len, 1, 1])
            # B * seq * FK
            seq_dense_embed = tf.reshape(
                seq_dense_embed * seq_dense_values,
                [-1, self.max_seq_len, item_dense_fields_num * self.embed_size],
            )
            self.seq_embed.append(seq_dense_embed)

        # B * K
        item_total_embed = tf.concat(self.item_embed, axis=1)
        # B * seq * K
        seq_total_embed = tf.concat(self.seq_embed, axis=2)

        seq_mask = tf.sequence_mask(self.user_interacted_len, self.max_seq_len)
        if self.use_tf_attention:
            attention_layer = tf_attention(item_total_embed, seq_total_embed, seq_mask)
        else:
            attention_layer = din_attention(item_total_embed, seq_total_embed, seq_mask)
        self.concat_embed.append(attention_layer)
