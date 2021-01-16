"""

Reference: Jiaxi Tang & Ke Wang. "Personalized Top-N Sequential Recommendation via Convolutional Sequence Embedding"
           (https://arxiv.org/pdf/1809.07426.pdf)

author: massquantity

"""
from itertools import islice
import numpy as np
import tensorflow as tf2
from tensorflow.keras.initializers import (
    truncated_normal as tf_truncated_normal,
    glorot_normal as tf_glorot_normal
)
from .base import Base, TfMixin
from ..evaluate.evaluate import EvalMixin
from ..utils.tf_ops import (
    reg_config,
    dropout_config,
    lr_decay_config
)
from ..data.data_generator import DataGenSequence
from ..data.sequence import user_last_interacted
from ..utils.misc import time_block, colorize
from ..utils.misc import count_params
from ..utils.tf_ops import conv_nn, max_pool
tf = tf2.compat.v1
tf.disable_v2_behavior()


class Caser(Base, TfMixin, EvalMixin):
    def __init__(
            self,
            task,
            data_info=None,
            embed_size=16,
            n_epochs=20,
            lr=0.001,
            lr_decay=False,
            reg=None,
            batch_size=256,
            num_neg=1,
            dropout_rate=None,
            use_bn=False,
            nh_filters=2,
            nv_filters=4,
            recent_num=10,
            random_num=None,
            seed=42,
            lower_upper_bound=None,
            tf_sess_config=None
    ):
        Base.__init__(self, task, data_info, lower_upper_bound)
        TfMixin.__init__(self, tf_sess_config)
        EvalMixin.__init__(self, task)

        self.task = task
        self.data_info = data_info
        self.embed_size = embed_size
        self.n_epochs = n_epochs
        self.lr = lr
        self.lr_decay = lr_decay
        self.reg = reg_config(reg)
        self.batch_size = batch_size
        self.num_neg = num_neg
        self.dropout_rate = dropout_config(dropout_rate)
        self.use_bn = use_bn
        self.nh_filters = nh_filters
        self.nv_filters = nv_filters
        self.seed = seed
        self.n_users = data_info.n_users
        self.n_items = data_info.n_items
        self.user_consumed = data_info.user_consumed
        (
            self.interaction_mode,
            self.max_seq_len
        ) = self._check_interaction_mode(recent_num, random_num)
        self.user_last_interacted = None
        self.last_interacted_len = None
        self.user_vector = None
        self.item_vector = None
        self.sparse = False
        self.dense = False

    def _build_model(self):
        tf.set_random_seed(self.seed)
        self._build_placeholders()
        self._build_variables()
        self._build_user_embeddings()

        item_embed = tf.nn.embedding_lookup(
            self.item_weights, self.item_indices
        )
        item_bias = tf.nn.embedding_lookup(
            self.item_biases, self.item_indices
        )
        self.output = tf.reduce_sum(
            tf.multiply(self.user_embed, item_embed),
            axis=1
        ) + item_bias
        count_params()

    def _build_placeholders(self):
        self.user_indices = tf.placeholder(tf.int32, shape=[None])
        self.item_indices = tf.placeholder(tf.int32, shape=[None])
        self.user_interacted_seq = tf.placeholder(
            tf.int32, shape=[None, self.max_seq_len])  # B * seq
        self.user_interacted_len = tf.placeholder(tf.int64, shape=[None])
        self.labels = tf.placeholder(tf.float32, shape=[None])
        self.is_training = tf.placeholder_with_default(False, shape=[])
        
    def _build_variables(self):
        self.user_feat = tf.get_variable(
            name="user_feat",
            shape=[self.n_users, self.embed_size],
            initializer=tf_truncated_normal(0.0, 0.01),
            regularizer=self.reg)

        # weight and bias parameters for last fc_layer
        self.item_biases = tf.get_variable(
            name="item_biases",
            shape=[self.n_items],
            initializer=tf.zeros,
        )
        self.item_weights = tf.get_variable(
            name="item_weights",
            shape=[self.n_items, self.embed_size * 2],
            initializer=tf_truncated_normal(0.0, 0.02),
            regularizer=self.reg
        )

        # input_embed for cnn_layer, include padding value
        self.input_embed = tf.get_variable(
            name="input_embed",
            shape=[self.n_items + 1, self.embed_size],
            initializer=tf_glorot_normal,
            regularizer=self.reg
        )

    def _build_user_embeddings(self):
        user_repr = tf.nn.embedding_lookup(self.user_feat, self.user_indices)
        # B * seq * K
        seq_item_embed = tf.nn.embedding_lookup(
            self.input_embed, self.user_interacted_seq)

        convs_out = []
        for i in range(1, self.max_seq_len + 1):
            # h_conv = tf.layers.conv1d(
            #    inputs=seq_item_embed,
            #    filters=self.nh_filters,
            #    kernel_size=i,
            #    strides=1,
            #    padding="valid",
            #    activation=tf.nn.relu
            # )

            h_conv = conv_nn(
                tf_version=tf.__version__,
                filters=self.nh_filters,
                kernel_size=i,
                strides=1,
                padding="valid",
                activation="relu"
            )(inputs=seq_item_embed)

            # h_conv = tf.reduce_max(h_conv, axis=1)
            h_size = h_conv.get_shape().as_list()[1]
            h_conv = max_pool(
                tf_version=tf.__version__,
                pool_size=h_size,
                strides=1,
                padding="valid"
            )(inputs=h_conv)
            h_conv = tf.squeeze(h_conv, axis=1)
            convs_out.append(h_conv)

        v_conv = conv_nn(
            tf_version=tf.__version__,
            filters=self.nv_filters,
            kernel_size=1,
            strides=1,
            padding="valid",
            activation="relu"
        )(inputs=tf.transpose(seq_item_embed, [0, 2, 1]))
        convs_out.append(tf.layers.flatten(v_conv))

        convs_out = tf.concat(convs_out, axis=1)
        convs_out = tf.layers.dense(
            inputs=convs_out,
            units=self.embed_size,
            activation=tf.nn.relu
        )
        self.user_embed = tf.concat([user_repr, convs_out], axis=1)

    def _build_train_ops(self, global_steps=None):
        if self.task == "rating":
            self.loss = tf.losses.mean_squared_error(labels=self.labels,
                                                     predictions=self.output)
        elif self.task == "ranking":
            self.loss = tf.reduce_mean(
                tf.nn.sigmoid_cross_entropy_with_logits(labels=self.labels,
                                                        logits=self.output)
            )

        if self.reg is not None:
            reg_keys = tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES)
            total_loss = self.loss + tf.add_n(reg_keys)
        else:
            total_loss = self.loss

        optimizer = tf.train.AdamOptimizer(self.lr)
        optimizer_op = optimizer.minimize(total_loss, global_step=global_steps)
        self.training_op = optimizer_op
        self.sess.run(tf.global_variables_initializer())

    def fit(self, train_data, verbose=1, shuffle=True,
            eval_data=None, metrics=None, **kwargs):
        self.show_start_time()
        if self.lr_decay:
            n_batches = int(len(train_data) / self.batch_size)
            self.lr, global_steps = lr_decay_config(
                self.lr, n_batches, **kwargs
            )
        else:
            global_steps = None

        self._build_model()
        self._build_train_ops(global_steps)

        data_generator = DataGenSequence(
            data=train_data,
            data_info=self.data_info,
            sparse=None,
            dense=None,
            mode=self.interaction_mode,
            num=self.max_seq_len,
            padding_idx=self.n_items
        )

        for epoch in range(1, self.n_epochs + 1):
            if self.lr_decay:
                print(f"With lr_decay, epoch {epoch} learning rate: "
                      f"{self.sess.run(self.lr)}")

            with time_block(f"Epoch {epoch}", verbose):
                train_total_loss = []
                for (u_seq, u_len, user, item, label, sparse_idx, dense_val
                     ) in data_generator(shuffle, self.batch_size):
                    u_len = np.asarray(u_len).astype(np.int64)
                    feed_dict = self._get_seq_feed_dict(
                        u_seq, u_len, user, item, label,
                        sparse_idx, dense_val, True
                    )
                    train_loss, _ = self.sess.run(
                        [self.loss, self.training_op], feed_dict
                    )
                    train_total_loss.append(train_loss)

            if verbose > 1:
                train_loss_str = "train_loss: " + str(
                    round(float(np.mean(train_total_loss)), 4)
                )
                print(f"\t {colorize(train_loss_str, 'green')}")
                # for evaluation
                self._set_latent_factors()
                self.print_metrics(eval_data=eval_data, metrics=metrics)
                print("=" * 30)

        # for prediction and recommendation
        self._set_latent_factors()

    def predict(self, user, item):
        user = (
            np.asarray([user])
            if isinstance(user, int)
            else np.asarray(user)
        )
        item = (
            np.asarray([item])
            if isinstance(item, int)
            else np.asarray(item)
        )
        unknown_num, unknown_index, user, item = self._check_unknown(user, item)

        preds = np.sum(
            np.multiply(self.user_vector[user],
                        self.item_vector[item]),
            axis=1
        )

        if self.task == "rating":
            preds = np.clip(preds, self.lower_bound, self.upper_bound)
        elif self.task == "ranking":
            preds = 1 / (1 + np.exp(-preds))

        if unknown_num > 0:
            preds[unknown_index] = self.default_prediction

        return preds[0] if len(user) == 1 else preds

    def recommend_user(self, user, n_rec, **kwargs):
        user = self._check_unknown_user(user)
        if not user:
            return

        recos = self.user_vector[user] @ self.item_vector.T
        if self.task == "ranking":
            recos = 1 / (1 + np.exp(-recos))

        consumed = self.user_consumed[user]
        count = n_rec + len(consumed)
        ids = np.argpartition(recos, -count)[-count:]
        rank = sorted(zip(ids, recos[ids]), key=lambda x: -x[1])
        return list(
            islice(
                (rec for rec in rank if rec[0] not in consumed), n_rec
            )
        )

    def _set_last_interacted(self):
        if (self.user_last_interacted is None
                and self.last_interacted_len is None):

            user_indices = np.arange(self.n_users)
            (
                self.user_last_interacted,
                self.last_interacted_len
            ) = user_last_interacted(
                user_indices, self.user_consumed, self.n_items,
                self.max_seq_len
            )

            self.last_interacted_len = np.asarray(
                self.last_interacted_len
            ).astype(np.int64)

    def _set_latent_factors(self):
        self._set_last_interacted()
        feed_dict = {self.user_indices: np.arange(self.n_users),
                     self.user_interacted_seq: self.user_last_interacted,
                     self.user_interacted_len: self.last_interacted_len}
        user_embed = self.sess.run(self.user_embed, feed_dict)
        item_weights = self.sess.run(self.item_weights)
        item_biases = self.sess.run(self.item_biases)

        user_bias = np.ones([len(user_embed), 1], dtype=user_embed.dtype)
        item_bias = item_biases[:, None]
        self.user_vector = np.hstack([user_embed, user_bias])
        self.item_vector = np.hstack([item_weights, item_bias])
