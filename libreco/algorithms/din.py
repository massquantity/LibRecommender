"""Implementation of DIN."""
import numpy as np

from ..bases import ModelMeta, TfBase
from ..batch.sequence import get_user_last_interacted
from ..feature.multi_sparse import true_sparse_field_size
from ..tfops import (
    dense_nn,
    dropout_config,
    multi_sparse_combine_embedding,
    reg_config,
    tf,
    tf_dense,
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

    user_variables = ["user_feat"]
    item_variables = ["item_feat"]
    sparse_variables = ["sparse_feat"]
    dense_variables = ["dense_feat"]

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
        self.recent_seqs, self.recent_seq_lens = self._set_recent_seqs()
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
        self._build_variables()
        self._build_user_item()
        if self.sparse:
            self._build_sparse()
        if self.dense:
            self._build_dense()
        self._build_attention()

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

    def _build_variables(self):
        self.user_feat = tf.get_variable(
            name="user_feat",
            shape=[self.n_users + 1, self.embed_size],
            initializer=tf.glorot_uniform_initializer(),
            regularizer=self.reg,
        )
        self.item_feat = tf.get_variable(
            name="item_feat",
            shape=[self.n_items + 1, self.embed_size],
            initializer=tf.glorot_uniform_initializer(),
            regularizer=self.reg,
        )
        if self.sparse:
            self.sparse_feat = tf.get_variable(
                name="sparse_feat",
                shape=[self.sparse_feature_size, self.embed_size],
                initializer=tf.glorot_uniform_initializer(),
                regularizer=self.reg,
            )
        if self.dense:
            self.dense_feat = tf.get_variable(
                name="dense_feat",
                shape=[self.dense_field_size, self.embed_size],
                initializer=tf.glorot_uniform_initializer(),
                regularizer=self.reg,
            )

    def _build_user_item(self):
        user_embed = tf.nn.embedding_lookup(self.user_feat, self.user_indices)
        item_embed = tf.nn.embedding_lookup(self.item_feat, self.item_indices)
        self.concat_embed.extend([user_embed, item_embed])
        self.item_embed.append(item_embed)

    def _build_sparse(self):
        sparse_embed = tf.nn.embedding_lookup(self.sparse_feat, self.sparse_indices)

        if self.data_info.multi_sparse_combine_info and self.multi_sparse_combiner in (
            "sum",
            "mean",
            "sqrtn",
        ):
            multi_sparse_embed = multi_sparse_combine_embedding(
                self.data_info,
                self.sparse_feat,
                self.sparse_indices,
                self.multi_sparse_combiner,
                self.embed_size,
            )
            self.concat_embed.append(
                tf.reshape(
                    multi_sparse_embed,
                    [-1, self.true_sparse_field_size * self.embed_size],
                )
            )
        else:
            self.concat_embed.append(
                tf.reshape(sparse_embed, [-1, self.sparse_field_size * self.embed_size])
            )

        if self.item_sparse:
            item_sparse_embed = tf.keras.layers.Flatten()(
                tf.gather(sparse_embed, self.item_sparse_col_indices, axis=1)
            )
            self.item_embed.append(item_sparse_embed)

    def _build_dense(self):
        batch_size = tf.shape(self.dense_values)[0]
        # 1 * F_dense * K
        dense_embed = tf.expand_dims(self.dense_feat, axis=0)
        # B * F_dense * K
        dense_embed = tf.tile(dense_embed, [batch_size, 1, 1])
        dense_values_reshape = tf.reshape(
            self.dense_values, [-1, self.dense_field_size, 1]
        )
        dense_embed = tf.multiply(dense_embed, dense_values_reshape)
        self.concat_embed.append(
            tf.reshape(dense_embed, [-1, self.dense_field_size * self.embed_size])
        )

        if self.item_dense:
            item_dense_embed = tf.keras.layers.Flatten()(
                tf.gather(dense_embed, self.item_dense_col_indices, axis=1)
            )
            self.item_embed.append(item_dense_embed)

    def _build_attention(self):
        # B * seq * K
        seq_item_embed = tf.nn.embedding_lookup(
            self.item_feat, self.user_interacted_seq
        )
        self.seq_embed.append(seq_item_embed)

        if self.item_sparse:
            # contains unique field indices for each item
            item_sparse_fields = tf.convert_to_tensor(
                self.data_info.item_sparse_unique, dtype=tf.int64
            )
            item_sparse_fields_num = tf.shape(item_sparse_fields)[1]

            # B * seq * F_sparse
            seq_sparse_fields = tf.gather(item_sparse_fields, self.user_interacted_seq)
            # B * seq * F_sparse * K
            seq_sparse_embed = tf.nn.embedding_lookup(
                self.sparse_feat, seq_sparse_fields
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
            dense_embed = tf.reshape(
                tf.gather(self.dense_feat, self.item_dense_col_indices),
                [1, 1, item_dense_fields_num, self.embed_size],
            )
            # B * seq * F_dense * K
            # Since dense_embeddings are same for all items, we can simply repeat it (batch * seq) times
            seq_dense_embed = tf.tile(dense_embed, [batch_size, self.max_seq_len, 1, 1])
            seq_dense_embed = tf.multiply(seq_dense_embed, seq_dense_values)
            # B * seq * FK
            seq_dense_embed = tf.reshape(
                seq_dense_embed,
                [-1, self.max_seq_len, item_dense_fields_num * self.embed_size],
            )
            self.seq_embed.append(seq_dense_embed)

        # B * K
        item_total_embed = tf.concat(self.item_embed, axis=1)
        # B * seq * K
        seq_total_embed = tf.concat(self.seq_embed, axis=2)

        attention_layer = self._attention_unit(
            item_total_embed, seq_total_embed, self.user_interacted_len
        )
        self.concat_embed.append(tf.keras.layers.Flatten()(attention_layer))

    def _attention_unit(self, queries, keys, keys_len):
        if self.use_tf_attention:
            query_masks = tf.cast(
                tf.ones_like(tf.reshape(self.user_interacted_len, [-1, 1])),
                dtype=tf.bool,
            )
            key_masks = tf.sequence_mask(self.user_interacted_len, self.max_seq_len)
            queries = tf.expand_dims(queries, axis=1)
            attention = tf.keras.layers.Attention(use_scale=False)
            pooled_outputs = attention(
                inputs=[queries, keys], mask=[query_masks, key_masks]
            )
            return pooled_outputs
        else:
            # queries: B * K, keys: B * seq * K
            queries = tf.expand_dims(queries, axis=1)
            # B * seq * K
            queries = tf.tile(queries, [1, self.max_seq_len, 1])
            queries_keys_cross = tf.concat(
                [queries, keys, queries - keys, queries * keys], axis=2
            )
            mlp_layer = dense_nn(
                queries_keys_cross,
                (16,),
                use_bn=False,
                activation=tf.nn.sigmoid,
                name="attention",
            )
            # B * seq * 1
            mlp_layer = tf_dense(units=1, activation=None)(mlp_layer)
            # attention_weights = tf.transpose(mlp_layer, [0, 2, 1])
            attention_weights = tf.keras.layers.Flatten()(mlp_layer)

            key_masks = tf.sequence_mask(keys_len, self.max_seq_len)
            paddings = tf.ones_like(attention_weights) * (-(2**32) + 1)
            attention_scores = tf.where(key_masks, attention_weights, paddings)
            attention_scores = tf.div_no_nan(
                attention_scores,
                tf.sqrt(tf.cast(keys.get_shape().as_list()[-1], tf.float32)),
            )
            # B * 1 * seq
            attention_scores = tf.expand_dims(tf.nn.softmax(attention_scores), 1)
            # B * 1 * K
            pooled_outputs = attention_scores @ keys
            return pooled_outputs

    def _set_recent_seqs(self):
        recent_seqs, recent_seq_lens = get_user_last_interacted(
            self.n_users, self.user_consumed, self.n_items, self.max_seq_len
        )
        oov = np.full(self.max_seq_len, self.n_items, dtype=np.int32)
        recent_seqs = np.vstack([recent_seqs, oov])
        recent_seq_lens = np.append(recent_seq_lens, [1])
        return recent_seqs, recent_seq_lens
