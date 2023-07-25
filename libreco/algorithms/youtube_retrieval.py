"""Implementation of YouTubeRetrieval."""
import numpy as np

from ..bases import DynEmbedBase, ModelMeta
from ..feature.multi_sparse import true_sparse_field_size
from ..layers import dense_nn, normalize_embeds, sparse_embeds_pooling
from ..recommendation import check_dynamic_rec_feats
from ..recommendation.preprocess import process_embed_feat, process_sparse_embed_seq
from ..tfops import dropout_config, reg_config, tf
from ..tfops.features import (
    compute_dense_feats,
    compute_sparse_feats,
    get_sparse_feed_dict,
)
from ..torchops import hidden_units_config
from ..utils.misc import count_params
from ..utils.validate import (
    check_multi_sparse,
    check_seq_mode,
    dense_field_size,
    sparse_feat_size,
    sparse_field_size,
)


class YouTubeRetrieval(DynEmbedBase, metaclass=ModelMeta, backend="tensorflow"):
    """*YouTubeRetrieval* algorithm. See :ref:`YouTubeRetrieval / YouTubeRanking` for more details.

    .. NOTE::
        The algorithm implemented mainly corresponds to the candidate generation
        phase based on the original paper.

    .. WARNING::
        YouTubeRetrieval can only be used in `ranking` task.

    Parameters
    ----------
    task : {'ranking'}
        Recommendation task. See :ref:`Task`.
    data_info : :class:`~libreco.data.DataInfo` object
        Object that contains useful information for training and inference.
    loss_type : {'sampled_softmax', 'nce'}, default: 'sampled_softmax'
        Loss for model training.
    embed_size: int, default: 16
        Vector size of embeddings.
    norm_embed : bool, default: False
        Whether to l2 normalize output embeddings.
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
    use_bn : bool, default: True
        Whether to use batch normalization.
    dropout_rate : float or None, default: None
        Probability of an element to be zeroed. If it is None, dropout is not used.
    hidden_units : int, list of int or tuple of (int,), default: (128, 64)
        Number of layers and corresponding layer size in MLP.

        .. versionchanged:: 1.0.0
           Accept type of ``int``, ``list`` or ``tuple``, instead of ``str``.

    num_sampled_per_batch : int or None, default: None
        Number of negative samples in a batch. If None, it is set to `batch_size`.
    sampler : str, default: 'uniform'
        Negative Sampling strategy. 'uniform' will use uniform sampler, and setting to
        other value will use `log_uniform_candidate_sampler` in TensorFlow.
        In recommendation scenarios the uniform sampler is generally preferred.
    recent_num : int or None, default: 10
        Number of recent items to use in user behavior sequence.
    random_num : int or None, default: None
        Number of random sampled items to use in user behavior sequence.
        If `recent_num` is not None, `random_num` is not considered.
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
    *Paul Covington et al.* `Deep Neural Networks for YouTube Recommendations
    <https://static.googleusercontent.com/media/research.google.com/zh-CN//pubs/archive/45530.pdf>`_.
    """

    item_variables = (
        "embedding/seq_embeds_var",
        "embedding/item_embeds_var",
        "embedding/item_bias_var",
    )
    sparse_variables = ("embedding/sparse_embeds_var",)
    dense_variables = ("embedding/dense_embeds_var",)

    def __init__(
        self,
        task="ranking",
        data_info=None,
        loss_type="sampled_softmax",
        embed_size=16,
        norm_embed=False,
        n_epochs=20,
        lr=0.001,
        lr_decay=False,
        epsilon=1e-5,
        reg=None,
        batch_size=256,
        use_bn=True,
        dropout_rate=None,
        hidden_units=(128, 64),
        num_sampled_per_batch=None,
        sampler="uniform",
        recent_num=10,
        random_num=None,
        multi_sparse_combiner="sqrtn",
        seed=42,
        lower_upper_bound=None,
        tf_sess_config=None,
    ):
        super().__init__(task, data_info, embed_size, norm_embed)

        assert task == "ranking", "YouTube-type models is only suitable for ranking"
        if len(data_info.item_col) > 0:
            raise ValueError("The `YouTuBeRetrieval` model assumes no item features.")
        self.all_args = locals()
        self.loss_type = loss_type
        self.n_epochs = n_epochs
        self.lr = lr
        self.lr_decay = lr_decay
        self.epsilon = epsilon
        self.hidden_units = [*hidden_units_config(hidden_units), self.embed_size]
        self.reg = reg_config(reg)
        self.batch_size = batch_size
        self.use_bn = use_bn
        self.dropout_rate = dropout_config(dropout_rate)
        self.num_sampled_per_batch = num_sampled_per_batch
        self.sampler = sampler
        self.seed = seed
        self.seq_mode, self.max_seq_len = check_seq_mode(recent_num, random_num)
        self.recent_seq_indices, self.recent_seq_values = self._set_recent_seqs()
        self.user_sparse = True if data_info.user_sparse_col.name else False
        self.user_dense = True if data_info.user_dense_col.name else False
        if self.user_sparse:
            self.sparse_feature_size = sparse_feat_size(data_info)
            self.sparse_field_size = sparse_field_size(data_info)
            self.multi_sparse_combiner = check_multi_sparse(
                data_info, multi_sparse_combiner
            )
            self.true_sparse_field_size = true_sparse_field_size(
                data_info, self.sparse_field_size, self.multi_sparse_combiner
            )
        if self.user_dense:
            self.dense_field_size = dense_field_size(data_info)

    def build_model(self):
        tf.set_random_seed(self.seed)
        # item_indices actually serve as labels in `YouTubeRetrieval` model
        self.item_indices = tf.placeholder(tf.int64, shape=[None])
        self.is_training = tf.placeholder_with_default(False, shape=[])
        self.concat_embed = []

        self._build_item_interaction()
        self._build_variables()
        if self.user_sparse:
            self._build_sparse()
        if self.user_dense:
            self._build_dense()

        concat_features = tf.concat(self.concat_embed, axis=1)
        self.user_embeds = dense_nn(
            concat_features,
            self.hidden_units,
            use_bn=self.use_bn,
            dropout_rate=self.dropout_rate,
            is_training=self.is_training,
        )
        self.serving_topk = self.build_topk()
        count_params()

    def _build_item_interaction(self):
        self.item_interaction_indices = tf.placeholder(tf.int64, shape=[None, 2])
        self.item_interaction_values = tf.placeholder(tf.int64, shape=[None])
        # `batch_size` may change during training, especially first item in sequence
        self.modified_batch_size = tf.placeholder(tf.int32, shape=[])

        sparse_item_interaction = tf.SparseTensor(
            self.item_interaction_indices,
            self.item_interaction_values,
            (self.modified_batch_size, self.n_items),
        )
        pooled_embed = sparse_embeds_pooling(
            sparse_item_interaction,
            var_name="seq_embeds_var",
            var_shape=(self.n_items, self.embed_size),
            initializer=tf.glorot_uniform_initializer(),
            regularizer=self.reg,
            reuse_layer=False,
            scope_name="embedding",
        )
        self.concat_embed.append(pooled_embed)

    def _build_sparse(self):
        self.user_sparse_indices = tf.placeholder(
            tf.int32, shape=[None, self.sparse_field_size]
        )
        sparse_embed = compute_sparse_feats(
            self.data_info,
            self.multi_sparse_combiner,
            self.user_sparse_indices,
            var_name="sparse_embeds_var",
            var_shape=(self.sparse_feature_size, self.embed_size),
            initializer=tf.glorot_uniform_initializer(),
            regularizer=self.reg,
            flatten=True,
        )
        self.concat_embed.append(sparse_embed)

    def _build_dense(self):
        self.user_dense_values = tf.placeholder(
            tf.float32, shape=[None, self.dense_field_size]
        )
        dense_embed = compute_dense_feats(
            self.user_dense_values,
            var_name="dense_embeds_var",
            var_shape=(self.dense_field_size, self.embed_size),
            initializer=tf.glorot_uniform_initializer(),
            regularizer=self.reg,
            flatten=True,
        )
        self.concat_embed.append(dense_embed)

    def _build_variables(self):
        with tf.variable_scope("embedding"):
            self.item_embeds = tf.get_variable(
                name="item_embeds_var",
                shape=[self.n_items, self.embed_size],
                initializer=tf.glorot_uniform_initializer(),
                regularizer=self.reg,
            )
            self.item_biases = tf.get_variable(
                name="item_bias_var",
                shape=[self.n_items],
                initializer=tf.zeros_initializer(),
                regularizer=self.reg,
                trainable=True,
            )

    def _set_recent_seqs(self):
        interacted_indices = []
        interacted_items = []
        for u in range(self.n_users):
            u_consumed_items = self.user_consumed[u]
            u_items_len = len(u_consumed_items)
            if u_items_len < self.max_seq_len:
                interacted_indices.extend([u] * u_items_len)
                interacted_items.extend(u_consumed_items)
            else:
                interacted_indices.extend([u] * self.max_seq_len)
                interacted_items.extend(u_consumed_items[-self.max_seq_len :])

        interacted_indices = np.array(interacted_indices).reshape(-1, 1)
        indices = np.concatenate(
            [interacted_indices, np.zeros_like(interacted_indices)], axis=1
        )
        return indices, np.array(interacted_items, dtype=np.int64)

    def dyn_user_embedding(
        self,
        user,
        user_feats=None,
        seq=None,
        include_bias=False,
        inner_id=False,
    ):
        check_dynamic_rec_feats(self.model_name, user, user_feats, seq)
        user_id = self.convert_array_id(user, inner_id) if user is not None else None
        sparse_indices, dense_values = process_embed_feat(
            self.data_info, user_id, user_feats
        )
        item_interaction_indices, item_interaction_values = process_sparse_embed_seq(
            self, user_id, seq, inner_id
        )
        batch_size = self.n_users if user_id is None else 1

        feed_dict = get_sparse_feed_dict(
            model=self,
            sparse_tensor_indices=item_interaction_indices,
            sparse_tensor_values=item_interaction_values,
            user_sparse_indices=sparse_indices,
            user_dense_values=dense_values,
            batch_size=batch_size,
            is_training=False,
        )
        user_embeds = self.sess.run(self.user_embeds, feed_dict)
        if self.norm_embed:
            user_embeds = normalize_embeds(user_embeds, backend="np")
        if include_bias:
            # add pseudo bias
            user_biases = np.ones([len(user_embeds), 1], dtype=user_embeds.dtype)
            user_embeds = np.hstack([user_embeds, user_biases])
        return user_embeds if user_id is None else np.squeeze(user_embeds, axis=0)
