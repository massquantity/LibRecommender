"""Implementation of TwoTower model."""
import numpy as np

from ..bases import EmbedBase, ModelMeta
from ..embedding import normalize_embeds
from ..feature.ssl import get_mutual_info
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
        remove_accidental_hits=False,
        ssl_pattern=None,
        alpha=0.2,
        seed=42,
        tf_sess_config=None,
    ):
        super().__init__(task, data_info, embed_size)

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
        self.remove_accidental_hits = remove_accidental_hits
        self.ssl_pattern = ssl_pattern
        self.alpha = alpha
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
        if self.ssl_pattern is not None:
            if self.ssl_pattern not in ("rfm", "rfm-complementary", "cfm"):
                raise ValueError(
                    f"`ssl` pattern supports `rfm`, `rfm-complementary` and `cfm`, "
                    f"got {self.ssl_pattern}."
                )
            if not self.item_sparse:
                raise ValueError(
                    "`ssl`(self-supervised learning) relies on item sparse features, "
                    "which are not available in training data."
                )
            if self.loss_type != "softmax":
                raise ValueError(
                    "`ssl`(self-supervised learning) can only be used in `softmax` loss."
                )

    def build_model(self):
        tf.set_random_seed(self.seed)
        self._build_placeholders()
        self._build_variables()
        self.user_vector = self.compute_user_embeddings("user")
        self.item_vector = self.compute_item_embeddings("item")
        if self.loss_type == "cross_entropy":
            self.output = tf.reduce_sum(self.user_vector * self.item_vector, axis=1)
        if self.loss_type == "max_margin":
            self.item_vector_neg = self.compute_item_embeddings("item_neg")
        if self.ssl_pattern is not None:
            self.ssl_left_vector = self.compute_ssl_embeddings("ssl_left")
            self.ssl_right_vector = self.compute_ssl_embeddings("ssl_right")
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

        if self.ssl_pattern is not None:
            # item_indices + sparse_indices
            self.ssl_left_sparse_indices = tf.placeholder(
                tf.int32, shape=[None, len(self.data_info.item_sparse_col.name) + 1]
            )
            self.ssl_right_sparse_indices = tf.placeholder(
                tf.int32, shape=[None, len(self.data_info.item_sparse_col.name) + 1]
            )
            if self.item_dense:
                self.ssl_left_dense_values = tf.placeholder(
                    tf.float32, shape=[None, len(self.data_info.item_dense_col.name)]
                )
                self.ssl_right_dense_values = tf.placeholder(
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

        if self.ssl_pattern is not None:
            default_feat = tf.get_variable(
                name="default_feat",
                shape=[1, self.embed_size],
                initializer=tf.zeros_initializer(),
                trainable=False,
            )
            self.ssl_feat = tf.concat(
                [default_feat, self.item_feat, self.sparse_feat], axis=0
            )

    def compute_user_embeddings(self, category):
        user_embed = tf.nn.embedding_lookup(self.user_feat, self.user_indices)
        concat_embeds = [user_embed]
        if self.user_sparse:
            user_sparse_embed = self._compute_sparse_feats(category)
            concat_embeds.append(user_sparse_embed)
        if self.user_dense:
            user_dense_embed = self._compute_dense_feats(category)
            concat_embeds.append(user_dense_embed)

        user_features = (
            tf.concat(concat_embeds, axis=1)
            if len(concat_embeds) > 1
            else concat_embeds[0]
        )
        return self._shared_layers(user_features, "user_tower")

    def compute_item_embeddings(self, category):
        if category == "item":
            item_embed = tf.nn.embedding_lookup(self.item_feat, self.item_indices)
        elif category == "item_neg":
            item_embed = tf.nn.embedding_lookup(self.item_feat, self.item_indices_neg)
        else:
            raise ValueError("Unknown item category")

        concat_embeds = [item_embed]
        if self.item_sparse:
            item_sparse_embed = self._compute_sparse_feats(category)
            concat_embeds.append(item_sparse_embed)
        if self.item_dense:
            item_dense_embed = self._compute_dense_feats(category)
            concat_embeds.append(item_dense_embed)

        item_features = (
            tf.concat(concat_embeds, axis=1)
            if len(concat_embeds) > 1
            else concat_embeds[0]
        )
        return self._shared_layers(item_features, "item_tower")

    def compute_ssl_embeddings(self, category):
        ssl_embed = self._compute_sparse_feats(category)
        if self.item_dense:
            ssl_dense = self._compute_dense_feats(category)
            ssl_embed = tf.concat([ssl_embed, ssl_dense], axis=1)
        return self._shared_layers(ssl_embed, "item_tower")

    def _compute_sparse_feats(self, category):
        if category == "user":
            sparse_indices = self.user_sparse_indices
        elif category == "item":
            sparse_indices = self.item_sparse_indices
        elif category == "item_neg":
            sparse_indices = self.item_sparse_indices_neg
        elif category == "ssl_left":
            sparse_indices = self.ssl_left_sparse_indices
        elif category == "ssl_right":
            sparse_indices = self.ssl_right_sparse_indices
        else:
            raise ValueError("Unknown sparse indices category.")

        if category.startswith("ssl"):
            sparse_embed = tf.nn.embedding_lookup(self.ssl_feat, sparse_indices)
        else:
            sparse_embed = tf.nn.embedding_lookup(self.sparse_feat, sparse_indices)
        return tf.keras.layers.Flatten()(sparse_embed)

    def _compute_dense_feats(self, category):
        if category == "user":
            dense_col_indices = self.data_info.user_dense_col.index
            dense_values = self.user_dense_values
        else:
            dense_col_indices = self.data_info.item_dense_col.index
            if category == "item":
                dense_values = self.item_dense_values
            elif category == "item_neg":
                dense_values = self.item_dense_values_neg
            elif category == "ssl_left":
                dense_values = self.ssl_left_dense_values
            elif category == "ssl_right":
                dense_values = self.ssl_right_dense_values
            else:
                raise ValueError("Unknown dense values category.")
        batch_size = tf.shape(dense_values)[0]
        dense_values = tf.expand_dims(dense_values, axis=2)

        dense_embed = tf.gather(self.dense_feat, dense_col_indices, axis=0)
        dense_embed = tf.expand_dims(dense_embed, axis=0)
        dense_embed = tf.tile(dense_embed, [batch_size, 1, 1])
        # broadcast element-wise multiplication
        return tf.keras.layers.Flatten()(dense_values * dense_embed)

    def _shared_layers(self, inputs, name):
        embeds = dense_nn(
            inputs,
            self.hidden_units,
            use_bn=self.use_bn,
            dropout_rate=self.dropout_rate,
            is_training=self.is_training,
            reuse_layer=True,
            name=name,
        )
        return normalize_embeds(embeds, backend="tf") if self.norm_embed else embeds

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
        if self.ssl_pattern is not None and self.ssl_pattern == "cfm":
            self.sparse_feat_mutual_info = get_mutual_info(train_data, self.data_info)

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

        if hasattr(self, "temperature_var"):
            self.temperature = self.sess.run(self.temperature_var)
            print(f"Learned temperature variable: {self.temperature}")

    def adjust_logits(self, logits, all_adjust=True):
        temperature = (
            self.temperature_var
            if hasattr(self, "temperature_var")
            else self.temperature
        )
        logits = tf.math.divide_no_nan(logits, temperature)
        if self.use_correction and all_adjust:
            correction = tf.clip_by_value(self.correction, 1e-8, 1.0)
            logQ = tf.reshape(tf.math.log(correction), (1, -1))
            logits -= logQ

        if self.remove_accidental_hits and all_adjust:
            row_items = tf.reshape(self.item_indices, (1, -1))
            col_items = tf.reshape(self.item_indices, (-1, 1))
            equal_items = tf.cast(tf.equal(row_items, col_items), tf.float32)
            label_diag = tf.eye(tf.shape(logits)[0])
            mask = tf.cast(equal_items - label_diag, tf.bool)
            paddings = tf.fill(tf.shape(logits), tf.float32.min)
            return tf.where(mask, paddings, logits)
        else:
            return logits
