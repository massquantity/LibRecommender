"""Implementation of TwoTower model."""
import numpy as np

from ..bases import EmbedBase, ModelMeta
from ..embedding import normalize_embeds
from ..tfops import dense_nn, dropout_config, reg_config, sess_config, tf
from ..torchops import hidden_units_config
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
        sampler="in-batch",
        use_bn=True,
        dropout_rate=None,
        hidden_units=(128, 64, 32),
        seed=42,
        lower_upper_bound=None,
        tf_sess_config=None,
    ):
        super().__init__(task, data_info, embed_size, lower_upper_bound)

        assert task == "ranking", "`TwoTower` is only suitable for ranking"
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
        self.use_bn = use_bn
        self.dropout_rate = dropout_config(dropout_rate)
        self.hidden_units = hidden_units_config(hidden_units)
        self.seed = seed
        self.user_sparse = True if data_info.user_sparse_col.name else False
        self.item_sparse = True if data_info.item_sparse_col.name else False
        self.user_dense = True if data_info.user_dense_col.name else False
        self.item_dense = True if data_info.item_dense_col.name else False

    def build_model(self):
        tf.set_random_seed(self.seed)
        self.is_training = tf.placeholder_with_default(False, shape=())
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
        self.user_vector = self.compute_user_embeddings()
        self.item_vector = self.compute_item_embeddings()
        # self.output = tf.reduce_sum(self.user_vector * self.item_vector, axis=1)

    def compute_user_embeddings(self):
        self.user_indices = tf.placeholder(tf.int32, shape=[None])
        self.user_feat = tf.get_variable(
            name="user_feat",
            shape=[self.n_users, self.embed_size],
            initializer=tf.glorot_uniform_initializer(),
            regularizer=self.reg,
        )
        user_embed = tf.nn.embedding_lookup(self.user_feat, self.user_indices)
        if not self.user_sparse and not self.user_dense:
            user_features = user_embed
        else:
            concat_embeds = [user_embed]
            if self.user_sparse:
                self.user_sparse_indices = tf.placeholder(
                    tf.int32, shape=[None, len(self.data_info.user_sparse_col.name)]
                )
                user_sparse_embed = tf.keras.layers.Flatten()(
                    tf.nn.embedding_lookup(self.sparse_feat, self.user_sparse_indices)
                )
                concat_embeds.append(user_sparse_embed)
            if self.user_dense:
                self.user_dense_values = tf.placeholder(
                    tf.float32, shape=[None, len(self.data_info.user_dense_col.name)]
                )
                user_dense_embed = tf.keras.layers.Flatten()(
                    self._compute_dense_feats(is_user=True)
                )
                concat_embeds.append(user_dense_embed)
            user_features = tf.concat(concat_embeds, axis=1)

        user_vector = dense_nn(
            user_features,
            self.hidden_units,
            use_bn=self.use_bn,
            dropout_rate=self.dropout_rate,
            is_training=self.is_training,
            name="user_tower",
        )
        return (
            normalize_embeds(user_vector, backend="tf")
            if self.norm_embed
            else user_vector
        )

    def compute_item_embeddings(self):
        self.item_indices = tf.placeholder(tf.int32, shape=[None])
        self.item_feat = tf.get_variable(
            name="item_feat",
            shape=[self.n_items, self.embed_size],
            initializer=tf.glorot_uniform_initializer(),
            regularizer=self.reg,
        )
        item_embed = tf.nn.embedding_lookup(self.item_feat, self.item_indices)
        if not self.item_sparse and not self.item_dense:
            item_features = item_embed
        else:
            concat_embeds = [item_embed]
            if self.item_sparse:
                self.item_sparse_indices = tf.placeholder(
                    tf.int32, shape=[None, len(self.data_info.item_sparse_col.name)]
                )
                item_sparse_embed = tf.keras.layers.Flatten()(
                    tf.nn.embedding_lookup(self.sparse_feat, self.item_sparse_indices)
                )
                concat_embeds.append(item_sparse_embed)
            if self.item_dense:
                self.item_dense_values = tf.placeholder(
                    tf.float32, shape=[None, len(self.data_info.item_dense_col.name)]
                )
                item_dense_embed = tf.keras.layers.Flatten()(
                    self._compute_dense_feats(is_user=False)
                )
                concat_embeds.append(item_dense_embed)
            item_features = tf.concat(concat_embeds, axis=1)

        item_vector = dense_nn(
            item_features,
            self.hidden_units,
            use_bn=self.use_bn,
            dropout_rate=self.dropout_rate,
            is_training=self.is_training,
            name="item_tower",
        )
        return (
            normalize_embeds(item_vector, backend="tf")
            if self.norm_embed
            else item_vector
        )

    def _compute_dense_feats(self, is_user):
        if is_user:
            dense_values = self.user_dense_values
            dense_col_indices = self.data_info.user_dense_col.index
        else:
            dense_values = self.item_dense_values
            dense_col_indices = self.data_info.item_dense_col.index
        batch_size = tf.shape(dense_values)[0]
        dense_values = tf.expand_dims(dense_values, axis=2)
        dense_embed = tf.gather(self.dense_feat, dense_col_indices, axis=1)
        dense_embed = tf.expand_dims(dense_embed, axis=0)
        dense_embed = tf.tile(dense_embed, [batch_size, 1, 1])
        return tf.multiply(dense_values, dense_embed)

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
