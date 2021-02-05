"""

Reference: Paul Covington et al.  "Deep Neural Networks for YouTube Recommendations"
           (https://static.googleusercontent.com/media/research.google.com/zh-CN//pubs/archive/45530.pdf)

author: massquantity

"""
import os
from itertools import islice
import numpy as np
import tensorflow.compat.v1 as tf
from tensorflow.keras.initializers import (
    zeros as tf_zeros,
    truncated_normal as tf_truncated_normal
)
from .base import Base, TfMixin
from ..evaluation.evaluate import EvalMixin
from ..utils.tf_ops import (
    reg_config,
    dropout_config,
    dense_nn,
    lr_decay_config
)
from ..data.data_generator import DataGenSequence
from ..data.sequence import sparse_user_last_interacted
from ..utils.misc import time_block, colorize, assign_oov_vector
tf.disable_v2_behavior()


class YouTubeMatch(Base, TfMixin, EvalMixin):
    """
    The model implemented mainly corresponds to the candidate generation
    phase based on the original paper.
    """
    # user_variables = []
    item_variables = ["item_interaction_features", "nce_weights", "nce_biases"]
    sparse_variables = ["sparse_features"]
    dense_variables = ["dense_features"]
    user_variables_np = ["user_vector"]
    item_variables_np = ["item_weights"]

    def __init__(
            self,
            task="ranking",
            data_info=None,
            embed_size=16,
            n_epochs=20,
            lr=0.01,
            lr_decay=False,
            reg=None,
            batch_size=256,
            num_neg=1,
            use_bn=True,
            dropout_rate=None,
            hidden_units="128,64,32",
            loss_type="nce",
            recent_num=10,
            random_num=None,
            seed=42,
            lower_upper_bound=None,
            tf_sess_config=None
    ):
        Base.__init__(self, task, data_info, lower_upper_bound)
        TfMixin.__init__(self, tf_sess_config)
        EvalMixin.__init__(self, task, data_info)

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
        (
            self.interaction_mode,
            self.interaction_num
        ) = self._check_interaction_mode(recent_num, random_num)
        self.seed = seed
        self.user_vector = None
        self.item_weights = None
        # self.item_biases = None
        self.user_consumed = data_info.user_consumed
        self.sparse = self._decide_sparse_indices(data_info)
        self.dense = self._decide_dense_values(data_info)
        if self.sparse:
            self.sparse_feature_size = self._sparse_feat_size(data_info)
            self.sparse_field_size = self._sparse_field_size(data_info)
        if self.dense:
            self.dense_field_size = self._dense_field_size(data_info)
        self.vector_infer = True
        self.all_args = locals()

    def _build_model(self):
        self.graph_built = True
        tf.set_random_seed(self.seed)
        # item_indices actually serve as labels in YouTubeMatch model
        self.item_indices = tf.placeholder(tf.int32, shape=[None])
        self.is_training = tf.placeholder_with_default(False, shape=[])
        self.concat_embed = []

        self._build_item_interaction()
        if self.sparse:
            self._build_sparse()
        if self.dense:
            self._build_dense()

        concat_features = tf.concat(self.concat_embed, axis=1)
        self.user_vector_repr = dense_nn(concat_features,
                                         self.hidden_units,
                                         use_bn=self.use_bn,
                                         dropout_rate=self.dropout_rate,
                                         is_training=self.is_training)

    def _build_item_interaction(self):
        self.item_interaction_indices = tf.placeholder(
            tf.int64, shape=[None, 2])
        self.item_interaction_values = tf.placeholder(tf.int32, shape=[None])
        self.modified_batch_size = tf.placeholder(tf.int32, shape=[])

        item_interaction_features = tf.get_variable(
            name="item_interaction_features",
            shape=[self.n_items, self.embed_size],
            initializer=tf_truncated_normal(0.0, 0.01),
            regularizer=self.reg)

        sparse_item_interaction = tf.SparseTensor(
            self.item_interaction_indices,
            self.item_interaction_values,
            [self.modified_batch_size, self.n_items]
        )
        pooled_embed = tf.nn.safe_embedding_lookup_sparse(
            item_interaction_features, sparse_item_interaction,
            sparse_weights=None, combiner="sqrtn", default_id=None
        )  # unknown user will return 0-vector
        self.concat_embed.append(pooled_embed)

    def _build_sparse(self):
        self.sparse_indices = tf.placeholder(
            tf.int32, shape=[None, self.sparse_field_size])
        sparse_features = tf.get_variable(
            name="sparse_features",
            shape=[self.sparse_feature_size, self.embed_size],
            initializer=tf_truncated_normal(0.0, 0.01),
            regularizer=self.reg)

        sparse_embed = tf.nn.embedding_lookup(
            sparse_features, self.sparse_indices)
        sparse_embed = tf.reshape(
            sparse_embed, [-1, self.sparse_field_size * self.embed_size])
        self.concat_embed.append(sparse_embed)

    def _build_dense(self):
        self.dense_values = tf.placeholder(
            tf.float32, shape=[None, self.dense_field_size])
        dense_values_reshape = tf.reshape(
            self.dense_values, [-1, self.dense_field_size, 1])
        batch_size = tf.shape(self.dense_values)[0]

        dense_features = tf.get_variable(
            name="dense_features",
            shape=[self.dense_field_size, self.embed_size],
            initializer=tf_truncated_normal(0.0, 0.01),
            regularizer=self.reg)

        dense_embed = tf.expand_dims(dense_features, axis=0)
        # B * F2 * K
        dense_embed = tf.tile(dense_embed, [batch_size, 1, 1])
        dense_embed = tf.multiply(dense_embed, dense_values_reshape)
        dense_embed = tf.reshape(
            dense_embed, [-1, self.dense_field_size * self.embed_size])
        self.concat_embed.append(dense_embed)

    def _build_train_ops(self, **kwargs):
        self.nce_weights = tf.get_variable(
            name="nce_weights",
            # n_classes, embed_size
            shape=[self.n_items, self.user_vector_size],
            initializer=tf_truncated_normal(0.0, 0.01),
            regularizer=self.reg
        )
        self.nce_biases = tf.get_variable(
            name="nce_biases",
            shape=[self.n_items],
            initializer=tf_zeros,
            regularizer=self.reg,
            trainable=True
        )

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
                partition_strategy="div")
            )
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
                partition_strategy="div")
            )
        else:
            raise ValueError("Loss type must either be 'nce' "
                             "or 'sampled_softmax")

        if self.reg is not None:
            reg_keys = tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES)
            total_loss = self.loss + tf.add_n(reg_keys)
        else:
            total_loss = self.loss

        if self.lr_decay:
            n_batches = int(self.data_info.data_size / self.batch_size)
            self.lr, global_steps = lr_decay_config(self.lr, n_batches,
                                                    **kwargs)
        else:
            global_steps = None

        optimizer = tf.train.AdamOptimizer(self.lr)
        optimizer_op = optimizer.minimize(total_loss, global_step=global_steps)
        update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
        self.training_op = tf.group([optimizer_op, update_ops])
        self.sess.run(tf.global_variables_initializer())

    def fit(self, train_data, verbose=1, shuffle=True, eval_data=None,
            metrics=None, **kwargs):
        assert self.task == "ranking", (
            "YouTube models is only suitable for ranking"
        )
        self._check_item_col()
        self.show_start_time()
        if not self.graph_built:
            self._build_model()
            self._build_train_ops(**kwargs)

        data_generator = DataGenSequence(
            train_data, self.data_info, self.sparse, self.dense,
            mode=self.interaction_mode, num=self.interaction_num,
            class_name="YoutubeMatch", padding_idx=self.n_items
        )
        for epoch in range(1, self.n_epochs + 1):
            with time_block(f"Epoch {epoch}", verbose):
                train_total_loss = []
                for b, ii, iv, user, item, _, si, dv in data_generator(
                        shuffle, self.batch_size):
                    feed_dict = {self.modified_batch_size: b,
                                 self.item_interaction_indices: ii,
                                 self.item_interaction_values: iv,
                                 self.item_indices: item,
                                 self.is_training: True}
                    if self.sparse:
                        feed_dict.update({self.sparse_indices: si})
                    if self.dense:
                        feed_dict.update({self.dense_values: dv})
                    train_loss, _ = self.sess.run(
                        [self.loss, self.training_op], feed_dict)
                    train_total_loss.append(train_loss)

            if verbose > 1:
                train_loss_str = "train_loss: " + str(
                    round(float(np.mean(train_total_loss)), 4)
                )
                print(f"\t {colorize(train_loss_str, 'green')}")
                # for evaluation
                self._set_latent_vectors()
                self.print_metrics(eval_data=eval_data, metrics=metrics,
                                   **kwargs)
                print("="*30)

        # for prediction and recommendation
        self._set_latent_vectors()
        assign_oov_vector(self)

    def predict(self, user, item, cold_start="average", inner_id=False):
        user, item = self.convert_id(user, item, inner_id)
        unknown_num, unknown_index, user, item = self._check_unknown(user, item)

        preds = np.sum(
            np.multiply(self.user_vector[user],
                        self.item_weights[item]),
            axis=1)
        preds = 1 / (1 + np.exp(-preds))

        if unknown_num > 0 and cold_start == "popular":
            preds[unknown_index] = self.default_prediction
        return preds

    def recommend_user(self, user, n_rec, cold_start="average", inner_id=False):
        user_id = self._check_unknown_user(user, inner_id)
        if user_id is None:
            if cold_start == "average":
                user_id = self.n_users
            elif cold_start == "popular":
                return self.data_info.popular_items[:n_rec]
            else:
                raise ValueError(user)

        consumed = set(self.user_consumed[user_id])
        count = n_rec + len(consumed)
        recos = self.user_vector[user_id] @ self.item_weights.T
        recos = 1 / (1 + np.exp(-recos))

        ids = np.argpartition(recos, -count)[-count:]
        rank = sorted(zip(ids, recos[ids]), key=lambda x: -x[1])
        recs_and_scores = islice(
            (rec if inner_id else (self.data_info.id2item[rec[0]], rec[1])
             for rec in rank if rec[0] not in consumed),
            n_rec
        )
        return list(recs_and_scores)

    def _set_latent_vectors(self):
        user_indices = np.arange(self.n_users)
        (
            interacted_indices,
            interacted_values
        ) = sparse_user_last_interacted(
            user_indices, self.user_consumed, self.interaction_num
        )

        feed_dict = {self.item_interaction_indices: interacted_indices,
                     self.item_interaction_values: interacted_values,
                     self.modified_batch_size: self.n_users,
                     self.is_training: False}

        if self.sparse:
            # remove oov
            user_sparse_indices = self.data_info.user_sparse_unique[:-1]
            feed_dict.update({self.sparse_indices: user_sparse_indices})
        if self.dense:
            user_dense_values = self.data_info.user_dense_unique[:-1]
            feed_dict.update({self.dense_values: user_dense_values})

        user_vector = self.sess.run(self.user_vector_repr, feed_dict)
        item_weights = self.sess.run(self.nce_weights)
        item_biases = self.sess.run(self.nce_biases)

        user_bias = np.ones([len(user_vector), 1], dtype=user_vector.dtype)
        item_bias = item_biases[:, None]
        u_vector = np.hstack([user_vector, user_bias])
        i_weights = np.hstack([item_weights, item_bias])
        oov_zeros = np.zeros(self.user_vector_size + 1, dtype=np.float32)
        self.user_vector = np.vstack([u_vector, oov_zeros])
        self.item_weights = np.vstack([i_weights, oov_zeros])

    def _check_item_col(self):
        if len(self.data_info.item_col) > 0:
            raise ValueError(
                "The YouTubeMatch model assumes no item features."
            )

    def save(self, path, model_name, manual=True, inference_only=False):
        if not os.path.isdir(path):
            print(f"file folder {path} doesn't exists, creating a new one...")
            os.makedirs(path)
        self.save_params(path)
        if inference_only:
            variable_path = os.path.join(path, model_name)
            np.savez_compressed(variable_path,
                                user_vector=self.user_vector,
                                item_weights=self.item_weights)
        else:
            self.save_variables(path, model_name, inference_only=False)

    @classmethod
    def load(cls, path, model_name, data_info, manual=True):
        variable_path = os.path.join(path, f"{model_name}.npz")
        variables = np.load(variable_path)
        hparams = cls.load_params(path, data_info)
        model = cls(**hparams)
        model.user_vector = variables["user_vector"]
        model.item_weights = variables["item_weights"]
        return model
