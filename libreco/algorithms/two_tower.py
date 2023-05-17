"""Implementation of TwoTower model."""
import numpy as np

from ..bases import EmbedBase, ModelMeta
from ..embedding import normalize_embeds
from ..tfops import dense_nn, dropout_config, reg_config, sess_config, tf
from ..torchops import hidden_units_config
from ..utils.misc import count_params
from ..utils.validate import dense_field_size, sparse_feat_size


class TwoTower(EmbedBase, metaclass=ModelMeta, backend="tensorflow"):

    user_variables = ["user_feat"]
    item_variables = ["item_feat"]
    sparse_variables = ["sparse_feat"]
    dense_variables = ["dense_feat"]

    def __init__(
        self,
        task,
        data_info=None,
        loss_type="softmax",
        embed_size=16,
        norm_embed=False,
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
        margin=1.0,
        use_correction=True,
        temperature=1.0,
        seed=42,
        lower_upper_bound=None,
        tf_sess_config=None,
    ):
        super().__init__(task, data_info, embed_size, lower_upper_bound)

        self.all_args = locals()
        self.sess = sess_config(tf_sess_config)
        self.loss_type = loss_type
        self.norm_embed = norm_embed
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
        self.margin = margin
        self.use_correction = use_correction
        self.temperature = temperature
        self.seed = seed
        self.user_sparse = True if data_info.user_sparse_col.name else False
        self.item_sparse = True if data_info.item_sparse_col.name else False
        self.user_dense = True if data_info.user_dense_col.name else False
        self.item_dense = True if data_info.item_dense_col.name else False
        self._check_params()

    def _check_params(self):
        if self.task != "ranking":
            raise ValueError("`TwoTower` is only suitable for ranking")
        if self.loss_type not in ("cross_entropy", "max_margin", "softmax"):
            raise ValueError(f"Unsupported `loss_type`: {self.loss_type}")

    def build_model(self):
        tf.set_random_seed(self.seed)
        self._build_placeholders()
        self._build_variables()
        self.user_vector = self.compute_user_embeddings()
        self.item_vector = self.compute_item_embeddings()
        if self.loss_type == "cross_entropy":
            self.output = tf.reduce_sum(self.user_vector * self.item_vector, axis=1)
        if self.loss_type == "max_margin":
            self.item_vector_neg = self.compute_item_embeddings(is_neg=True)
        count_params()

    def _build_placeholders(self):
        self.user_indices = tf.placeholder(tf.int32, shape=[None])
        self.item_indices = tf.placeholder(tf.int32, shape=[None])
        if self.loss_type == "cross_entropy":
            self.labels = tf.placeholder(tf.float32, shape=[None])
        if self.loss_type == "max_margin":
            self.item_indices_neg = tf.placeholder(tf.int32, shape=[None])
        if self.loss_type == "softmax" and self.use_correction:
            self.correction = tf.placeholder(tf.float32, shape=[None])
        self.is_training = tf.placeholder_with_default(False, shape=[])

        if self.user_sparse:
            self.user_sparse_indices = tf.placeholder(
                tf.int32, shape=[None, len(self.data_info.user_sparse_col.name)]
            )
        if self.user_dense:
            self.user_dense_values = tf.placeholder(
                tf.float32, shape=[None, len(self.data_info.user_dense_col.name)]
            )
        if self.item_sparse:
            self.item_sparse_indices = tf.placeholder(
                tf.int32, shape=[None, len(self.data_info.item_sparse_col.name)]
            )
            if self.loss_type == "max_margin":
                self.item_sparse_indices_neg = tf.placeholder(
                    tf.int32, shape=[None, len(self.data_info.item_sparse_col.name)]
                )
        if self.item_dense:
            self.item_dense_values = tf.placeholder(
                tf.float32, shape=[None, len(self.data_info.item_dense_col.name)]
            )
            if self.loss_type == "max_margin":
                self.item_dense_values_neg = tf.placeholder(
                    tf.float32, shape=[None, len(self.data_info.item_dense_col.name)]
                )

    def _build_variables(self):
        self.user_feat = tf.get_variable(
            name="user_feat",
            shape=[self.n_users, self.embed_size],
            initializer=tf.glorot_uniform_initializer(),
            regularizer=self.reg,
        )
        self.item_feat = tf.get_variable(
            name="item_feat",
            shape=[self.n_items, self.embed_size],
            initializer=tf.glorot_uniform_initializer(),
            regularizer=self.reg,
        )
        if self.user_sparse or self.item_sparse:
            self.sparse_feat = tf.get_variable(
                name="sparse_feat",
                shape=[sparse_feat_size(self.data_info), self.embed_size],
                initializer=tf.glorot_uniform_initializer(),
                regularizer=self.reg,
            )
        if self.user_dense or self.item_dense:
            self.dense_feat = tf.get_variable(
                name="dense_feat",
                shape=[dense_field_size(self.data_info), self.embed_size],
                initializer=tf.glorot_uniform_initializer(),
                regularizer=self.reg,
            )
        if self.temperature <= 0.0:
            self.temperature_var = tf.get_variable(
                name="temperature_var",
                shape=(),
                initializer=tf.ones_initializer(),
                trainable=True,
            )

    def compute_user_embeddings(self):
        user_embed = tf.nn.embedding_lookup(self.user_feat, self.user_indices)
        if not self.user_sparse and not self.user_dense:
            user_features = user_embed
        else:
            concat_embeds = [user_embed]
            if self.user_sparse:
                user_sparse_embed = self._compute_sparse_feats(is_user=True)
                concat_embeds.append(user_sparse_embed)
            if self.user_dense:
                user_dense_embed = self._compute_dense_feats(is_user=True)
                concat_embeds.append(user_dense_embed)
            user_features = tf.concat(concat_embeds, axis=1)

        user_vector = dense_nn(
            user_features,
            self.hidden_units,
            use_bn=self.use_bn,
            dropout_rate=self.dropout_rate,
            is_training=self.is_training,
            reuse_layer=True,
            name="user_tower",
        )
        return (
            normalize_embeds(user_vector, backend="tf")
            if self.norm_embed
            else user_vector
        )

    def compute_item_embeddings(self, is_neg=False):
        item_indices = self.item_indices if not is_neg else self.item_indices_neg
        item_embed = tf.nn.embedding_lookup(self.item_feat, item_indices)
        if not self.item_sparse and not self.item_dense:
            item_features = item_embed
        else:
            concat_embeds = [item_embed]
            if self.item_sparse:
                item_sparse_embed = self._compute_sparse_feats(
                    is_user=False, is_item_neg=is_neg
                )
                concat_embeds.append(item_sparse_embed)
            if self.item_dense:
                item_dense_embed = self._compute_dense_feats(
                    is_user=False, is_item_neg=is_neg
                )
                concat_embeds.append(item_dense_embed)
            item_features = tf.concat(concat_embeds, axis=1)

        item_vector = dense_nn(
            item_features,
            self.hidden_units,
            use_bn=self.use_bn,
            dropout_rate=self.dropout_rate,
            is_training=self.is_training,
            reuse_layer=True,
            name="item_tower",
        )
        return (
            normalize_embeds(item_vector, backend="tf")
            if self.norm_embed
            else item_vector
        )

    def _compute_sparse_feats(self, is_user, is_item_neg=False):
        if is_user:
            sparse_indices = self.user_sparse_indices
        else:
            sparse_indices = (
                self.item_sparse_indices
                if not is_item_neg
                else self.item_sparse_indices_neg
            )
        sparse_embed = tf.nn.embedding_lookup(self.sparse_feat, sparse_indices)
        return tf.keras.layers.Flatten()(sparse_embed)

    def _compute_dense_feats(self, is_user, is_item_neg=False):
        if is_user:
            dense_values = self.user_dense_values
            dense_col_indices = self.data_info.user_dense_col.index
        else:
            dense_values = (
                self.item_dense_values
                if not is_item_neg
                else self.item_dense_values_neg
            )
            dense_col_indices = self.data_info.item_dense_col.index
        batch_size = tf.shape(dense_values)[0]
        dense_values = tf.expand_dims(dense_values, axis=2)

        dense_embed = tf.gather(self.dense_feat, dense_col_indices, axis=0)
        dense_embed = tf.expand_dims(dense_embed, axis=0)
        dense_embed = tf.tile(dense_embed, [batch_size, 1, 1])
        # broadcast element-wise multiplication
        return tf.keras.layers.Flatten()(dense_values * dense_embed)

    def fit(
        self,
        train_data,
        neg_sampling,
        verbose=1,
        shuffle=True,
        eval_data=None,
        metrics=None,
        k=10,
        eval_batch_size=8192,
        eval_user_num=None,
        num_workers=0,
    ):
        if self.loss_type == "softmax" and self.use_correction:
            _, item_counts = np.unique(train_data.item_indices, return_counts=True)
            assert len(item_counts) == self.n_items
            self.item_corrections = item_counts / len(train_data)

        super().fit(
            train_data,
            neg_sampling,
            verbose,
            shuffle,
            eval_data,
            metrics,
            k,
            eval_batch_size,
            eval_user_num,
        )

    def set_embeddings(self):
        user_feed_dict = {
            self.user_indices: np.arange(self.n_users),
            self.is_training: False,
        }
        if self.user_sparse:
            user_sparse_indices = self.data_info.user_sparse_unique[:-1]
            user_feed_dict.update({self.user_sparse_indices: user_sparse_indices})
        if self.user_dense:
            user_dense_values = self.data_info.user_dense_unique[:-1]
            user_feed_dict.update({self.user_dense_values: user_dense_values})
        self.user_embed = self.sess.run(self.user_vector, user_feed_dict)

        item_feed_dict = {
            self.item_indices: np.arange(self.n_items),
            self.is_training: False,
        }
        if self.item_sparse:
            item_sparse_indices = self.data_info.item_sparse_unique[:-1]
            item_feed_dict.update({self.item_sparse_indices: item_sparse_indices})
        if self.item_dense:
            item_dense_values = self.data_info.item_dense_unique[:-1]
            item_feed_dict.update({self.item_dense_values: item_dense_values})
        self.item_embed = self.sess.run(self.item_vector, item_feed_dict)

    def adjust_logits(self, logits):
        temperature = (
            self.temperature_var
            if hasattr(self, "temperature_var")
            else self.temperature
        )
        logits = tf.math.divide_no_nan(logits, temperature)
        if self.use_correction:
            correction = tf.clip_by_value(self.correction, 1e-8, 1.0)
            logQ = tf.reshape(tf.math.log(correction), (1, -1))
            logits -= logQ
        return logits
