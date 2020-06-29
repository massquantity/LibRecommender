"""

Reference: Paul Covington et al.  "Deep Neural Networks for YouTube Recommendations"
           (https://static.googleusercontent.com/media/research.google.com/zh-CN//pubs/archive/45530.pdf)

author: massquantity

"""
import time
from itertools import islice
import numpy as np
import tensorflow as tf
from tensorflow.python.keras.initializers import (
    zeros as tf_zeros,
    truncated_normal as tf_truncated_normal
)
from .base import Base, TfMixin
from ..evaluate.evaluate import EvalMixin
from ..utils.tf_ops import (
    reg_config,
    dropout_config,
    dense_nn,
    lr_decay_config
)
from ..data.data_generator import DataGenYoutube
from ..utils.tf_ops import sparse_tensor_interaction
from ..utils.colorize import colorize
from ..utils.timing import time_block


class YouTubeMatch(Base, TfMixin, EvalMixin):
    """
    The model implemented mainly correspond to the candidate generation
    part based on the original paper.
    """
    def __init__(self, task="ranking", data_info=None, embed_size=16,
                 n_epochs=20, lr=0.01, lr_decay=False, reg=None,
                 batch_size=256, num_neg=1, use_bn=True, dropout_rate=None,
                 hidden_units="128,64,32", loss_type="nce", seed=42,
                 lower_upper_bound=None, tf_sess_config=None):

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
        self.use_bn = use_bn
        self.dropout_rate = dropout_config(dropout_rate)
        self.hidden_units = list(map(int, hidden_units.split(",")))
        # the output of last DNN layer is user vector
        self.user_vector_size = self.hidden_units[-1]
        self.loss_type = loss_type
        self.n_users = data_info.n_users
        self.n_items = data_info.n_items
        self.global_mean = data_info.global_mean
        self.default_prediction = data_info.global_mean if (
                task == "rating") else 0.0
        self.seed = seed
    #    self.sess = tf.Session()
        self.user_vector = None
        self.item_weights = None
        self.item_biases = None
        self.user_consumed = None
        self.dense = self._decide_dense_values(data_info)
        self.sparse_feature_size = self._sparse_feat_size(data_info)
        self.sparse_field_size = self._sparse_field_size(data_info)
        if self.dense:
            self.dense_field_size = self._dense_field_size(data_info)

    def _build_model(self, sparse_item_interaction):
        tf.set_random_seed(self.seed)
        self.user_indices = tf.placeholder(tf.int32, shape=[None])
        # item_indices actually serve as label
        self.item_indices = tf.placeholder(tf.int32, shape=[None])
        self.sparse_indices = tf.placeholder(
            tf.int32, shape=[None, self.sparse_field_size])
        self.is_training = tf.placeholder_with_default(True, shape=[])

        item_interaction_features = tf.get_variable(
            name="item_interaction_features",
            shape=[self.n_items, self.embed_size],
            initializer=tf_truncated_normal(0.0, 0.01),
            regularizer=self.reg)
        sparse_features = tf.get_variable(
            name="sparse_features",
            shape=[self.sparse_feature_size, self.embed_size],
            initializer=tf_truncated_normal(0.0, 0.01),
            regularizer=self.reg)

        pooled_items = tf.nn.safe_embedding_lookup_sparse(
            item_interaction_features, sparse_item_interaction,
            sparse_weights=None, combiner="sqrtn", default_id=None
        )   # unknown user will return 0-vector
        pooled_embed = tf.nn.embedding_lookup(
            pooled_items, self.user_indices)

        sparse_embed = tf.nn.embedding_lookup(
            sparse_features, self.sparse_indices)
        sparse_embed = tf.reshape(
            sparse_embed, [-1, self.sparse_field_size * self.embed_size])

        concat_features = tf.concat(
            [pooled_embed, sparse_embed], axis=1)

        if self.dense:  # build dense part if present
            self.dense_indices = tf.placeholder(
                tf.int32, shape=[None, self.dense_field_size])
            self.dense_values = tf.placeholder(
                tf.float32, shape=[None, self.dense_field_size])

            dense_features = tf.get_variable(
                name="dense_features",
                shape=[self.dense_field_size, self.embed_size],
                initializer=tf_truncated_normal(0.0, 0.01),
                regularizer=self.reg)
            dense_embed = tf.nn.embedding_lookup(
                dense_features, self.dense_indices)
            dense_values = tf.reshape(
                self.dense_values, [-1, self.dense_field_size, 1])
            dense_embed = tf.multiply(dense_embed, dense_values)
            dense_embed = tf.reshape(
                dense_embed, [-1, self.dense_field_size * self.embed_size])
            concat_features = tf.concat(
                [concat_features, dense_embed], axis=1)

        self.user_vector_repr = dense_nn(concat_features,
                                         self.hidden_units,
                                         use_bn=self.use_bn,
                                         dropout_rate=self.dropout_rate,
                                         is_training=self.is_training)

    def _build_train_ops(self, global_steps=None):
        self.nce_weights = tf.get_variable(
            name="nce_weights",
            # n_classes, embed_size
            shape=[self.n_items, self.user_vector_size],
            initializer=tf_truncated_normal(0.0, 0.01),
            regularizer=self.reg)
        self.nce_biases = tf.get_variable(
            name="nce_biases",
            shape=[self.n_items],
            initializer=tf_zeros,
            regularizer=self.reg)

        if self.loss_type == "nce":
            self.loss = tf.reduce_mean(tf.nn.nce_loss(
                weights=self.nce_weights,
                biases=self.nce_biases,
                labels=tf.reshape(self.item_indices, [-1, 1]),
                inputs=self.user_vector_repr,
                num_sampled=self.num_neg,
                num_classes=self.n_items,
                num_true=1,
                remove_accidental_hits=True,
                partition_strategy="div"))
        elif self.loss_type == "sampled_softmax":
            self.loss = tf.reduce_mean(tf.nn.sampled_softmax_loss(
                weights=self.nce_weights,
                biases=self.nce_biases,
                labels=tf.reshape(self.item_indices, [-1, 1]),
                inputs=self.user_vector_repr,
                num_sampled=self.num_neg,
                num_classes=self.n_items,
                num_true=1,
                remove_accidental_hits=True,
                seed=self.seed,
                partition_strategy="div"))
        else:
            raise ValueError("Loss type must either be 'nce' "
                             "or 'sampled_softmax")

        if self.reg is not None:
            reg_keys = tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES)
            total_loss = self.loss + tf.add_n(reg_keys)
        else:
            total_loss = self.loss

        optimizer = tf.train.AdamOptimizer(self.lr)
        optimizer_op = optimizer.minimize(total_loss, global_step=global_steps)
        update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
        self.training_op = tf.group([optimizer_op, update_ops])
        self.sess.run(tf.global_variables_initializer())

    def fit(self, train_data, verbose=1, shuffle=True, sample_rate=None,
            recent_num=None, eval_data=None, metrics=None, **kwargs):
        self._check_item_col()
        start_time = time.strftime('%Y-%m-%d %H:%M:%S', time.localtime())
        print(f"training start time: {colorize(start_time, 'magenta')}")
        self.user_consumed = train_data.user_consumed
        if self.lr_decay:
            n_batches = int(len(train_data) / self.batch_size)
            self.lr, global_steps = lr_decay_config(self.lr, n_batches,
                                                    **kwargs)
        else:
            global_steps = None

        sparse_item_interaction = sparse_tensor_interaction(
            train_data, random_sample_rate=sample_rate, recent_num=recent_num)
        self._build_model(sparse_item_interaction)
        self._build_train_ops(global_steps)

        data_generator = DataGenYoutube(train_data, self.batch_size,
                                        self.dense, mode="match")
        for epoch in range(1, self.n_epochs + 1):
            with time_block(f"Epoch {epoch}", verbose):
                train_total_loss = []
                for user, item, si, di, dv, _ in data_generator(shuffle):
                    feed_dict = {self.user_indices: user,
                                 self.item_indices: item,
                                 self.sparse_indices: si,
                                 self.is_training: True}
                    if self.dense:
                        feed_dict.update({self.dense_indices: di,
                                          self.dense_values: dv})
                    train_loss, _ = self.sess.run(
                        [self.loss, self.training_op], feed_dict)
                    train_total_loss.append(train_loss)

            if verbose > 1:
                train_loss_str = "train_loss: " + str(
                    round(np.mean(train_total_loss), 4)
                )
                print(f"\t {colorize(train_loss_str, 'green')}")
                # for evaluation
                self._set_latent_vectors()
                self.print_metrics(eval_data=eval_data, metrics=metrics)
                print("="*30)

        # for prediction and recommendation
        self._set_latent_vectors()

    def predict(self, user, item):
        user = np.asarray(
            [user]) if isinstance(user, int) else np.asarray(user)
        item = np.asarray(
            [item]) if isinstance(item, int) else np.asarray(item)

        unknown_num, unknown_index, user, item = self._check_unknown(
            user, item)

        preds = np.sum(
            np.multiply(self.user_vector[user],
                        self.item_weights[item]),
            axis=1) + self.item_biases[item]
        preds = 1 / (1 + np.exp(-preds))

        if unknown_num > 0:
            preds[unknown_index] = self.default_prediction

        return preds[0] if len(user) == 1 else preds

    def recommend_user(self, user, n_rec, **kwargs):
        user = self._check_unknown_user(user)
        if not user:
            return   # popular ?

        consumed = self.user_consumed[user]
        count = n_rec + len(consumed)
        recos = self.user_vector[user] @ self.item_weights.T + self.item_biases
        recos = 1 / (1 + np.exp(-recos))

        ids = np.argpartition(recos, -count)[-count:]
        rank = sorted(zip(ids, recos[ids]), key=lambda x: -x[1])
        return list(
            islice(
                (rec for rec in rank if rec[0] not in consumed), n_rec
            )
        )

    def _set_latent_vectors(self):
        user_indices = np.arange(self.n_users)
        user_sparse_indices = self.data_info.user_sparse_unique

        feed_dict = {self.user_indices: user_indices,
                     self.sparse_indices: user_sparse_indices}
        if self.dense:
            user_dense_indices = np.tile(np.arange(self.dense_field_size),
                                         (self.n_users, 1))
            user_dense_values = self.data_info.user_dense_unique
            feed_dict.update({self.dense_indices: user_dense_indices,
                              self.dense_values: user_dense_values})

        self.user_vector = self.sess.run(self.user_vector_repr, feed_dict)
        self.item_weights = self.sess.run(self.nce_weights)
        self.item_biases = self.sess.run(self.nce_biases)

    def _check_item_col(self):
        if len(self.data_info.item_col) > 0:
            raise ValueError("The YouTubeMatch model assumes no item features.")


