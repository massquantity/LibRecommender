"""

Reference: Balazs Hidasi et al.  "Session-based Recommendations with Recurrent Neural Networks"
           (https://arxiv.org/pdf/1511.06939.pdf)

author: massquantity

"""
from itertools import islice
import os
import numpy as np
import tensorflow.compat.v1 as tf
from tensorflow.keras.initializers import (
    truncated_normal as tf_truncated_normal,
    orthogonal as tf_orthogonal,
    glorot_normal as tf_glorot_normal
)
from .base import Base, TfMixin
from ..evaluation.evaluate import EvalMixin
from ..utils.tf_ops import (
    reg_config,
    dropout_config,
    lr_decay_config
)
from ..data.data_generator import DataGenSequence
from ..data.sequence import user_last_interacted
from ..utils.sampling import PairwiseSamplingSeq
from ..utils.misc import time_block, colorize
from ..utils.misc import count_params, assign_oov_vector
tf.disable_v2_behavior()


class RNN4Rec(Base, TfMixin, EvalMixin):
    # user_variables = []
    item_variables = ["item_weights", "item_biases", "input_embed"]
    user_variables_np = ["user_vector"]
    item_variables_np = ["item_vector"]

    def __init__(
            self,
            task,
            data_info=None,
            rnn_type="lstm",
            loss_type="cross_entropy",
            embed_size=16,
            n_epochs=20,
            lr=0.001,
            lr_decay=False,
            hidden_units="16",
            reg=None,
            batch_size=256,
            num_neg=1,
            dropout_rate=None,
            use_layer_norm=False,
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
        self.rnn_type = rnn_type.lower()
        self.loss_type = loss_type.lower()
        self.embed_size = embed_size
        self.n_epochs = n_epochs
        self.lr = lr
        self.lr_decay = lr_decay
        self.hidden_units = list(map(int, hidden_units.split(",")))
        self.reg = reg_config(reg)
        self.batch_size = batch_size
        self.num_neg = num_neg
        self.dropout_rate = dropout_config(dropout_rate)
        self.use_ln = use_layer_norm
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
        self._check_params()
        self.vector_infer = True
        self.all_args = locals()

    def _build_model(self):
        self.graph_built = True
        tf.set_random_seed(self.seed)
        self.labels = tf.placeholder(tf.float32, shape=[None])
        self.is_training = tf.placeholder_with_default(False, shape=[])
        self._build_variables()
        self._build_user_embeddings()
        if self.task == "rating" or self.loss_type == "cross_entropy":
            self.user_indices = tf.placeholder(tf.int32, shape=[None])
            self.item_indices = tf.placeholder(tf.int32, shape=[None])

            item_embed = tf.nn.embedding_lookup(
                self.item_weights, self.item_indices
            )
            item_bias = tf.nn.embedding_lookup(
                self.item_biases, self.item_indices
            )
            self.output = tf.reduce_sum(
                tf.multiply(self.user_embed, item_embed), axis=1
            ) + item_bias

        elif self.loss_type == "bpr":
            self.item_indices_pos = tf.placeholder(tf.int32, shape=[None])
            self.item_indices_neg = tf.placeholder(tf.int32, shape=[None])
            item_embed_pos = tf.nn.embedding_lookup(
                self.item_weights, self.item_indices_pos
            )
            item_embed_neg = tf.nn.embedding_lookup(
                self.item_weights, self.item_indices_neg
            )
            item_bias_pos = tf.nn.embedding_lookup(
                self.item_biases, self.item_indices_pos
            )
            item_bias_neg = tf.nn.embedding_lookup(
                self.item_biases, self.item_indices_neg
            )

            item_diff = tf.subtract(item_bias_pos,
                                    item_bias_neg) + tf.reduce_sum(
                tf.multiply(
                    self.user_embed,
                    tf.subtract(item_embed_pos, item_embed_neg)
                ), axis=1
            )
            self.log_sigmoid = tf.log_sigmoid(item_diff)

        count_params()

    def _build_variables(self):
        # weight and bias parameters for last fc_layer
        self.item_biases = tf.get_variable(
            name="item_biases",
            shape=[self.n_items],
            initializer=tf.zeros,
            regularizer=self.reg
        )
        self.item_weights = tf.get_variable(
            name="item_weights",
            shape=[self.n_items, self.embed_size],
            initializer=tf_truncated_normal(0.0, 0.02),
            regularizer=self.reg
        )

        # input_embed for rnn_layer, include padding value
        self.input_embed = tf.get_variable(
            name="input_embed",
            shape=[self.n_items + 1, self.hidden_units[0]],
            initializer=tf_glorot_normal,
            regularizer=self.reg
        )

    def _build_user_embeddings(self):
        self.user_interacted_seq = tf.placeholder(
            tf.int32, shape=[None, self.max_seq_len]
        )
        self.user_interacted_len = tf.placeholder(tf.int64, shape=[None])
        seq_item_embed = tf.nn.embedding_lookup(
            self.input_embed, self.user_interacted_seq
        )

        if tf.__version__ >= "2.0.0":
            # cell_type = (
            #    tf.keras.layers.LSTMCell
            #    if self.rnn_type.endswith("lstm")
            #    else tf.keras.layers.GRUCell
            # )
            # cells = [cell_type(size) for size in self.hidden_units]
            # masks = tf.sequence_mask(self.user_interacted_len, self.max_seq_len)
            # tf2_rnn = tf.keras.layers.RNN(cells, return_state=True)
            # output, *state = tf2_rnn(seq_item_embed, mask=masks)

            rnn = (
                tf.keras.layers.LSTM
                if self.rnn_type.endswith("lstm")
                else tf.keras.layers.GRU
            )
            out = seq_item_embed
            masks = tf.sequence_mask(self.user_interacted_len, self.max_seq_len)
            for units in self.hidden_units:
                out = rnn(
                    units, return_sequences=True,
                    dropout=self.dropout_rate,
                    recurrent_dropout=self.dropout_rate,
                    activation=None if self.use_ln else "tanh"
                )(out, mask=masks, training=self.is_training)

                if self.use_ln:
                    out = tf.keras.layers.LayerNormalization()(out)
                    out = tf.keras.activations.get("tanh")(out)

            out = out[:, -1, :]
            self.user_embed = tf.keras.layers.Dense(
                units=self.embed_size, activation=None
            )(out)

        else:
            cell_type = (
                tf.nn.rnn_cell.LSTMCell
                if self.rnn_type.endswith("lstm")
                else tf.nn.rnn_cell.GRUCell
            )
            cells = [cell_type(size) for size in self.hidden_units]
            stacked_cells = tf.nn.rnn_cell.MultiRNNCell(cells)
            zero_state = stacked_cells.zero_state(
                tf.shape(seq_item_embed)[0], dtype=tf.float32
            )

            output, state = tf.nn.dynamic_rnn(
                cell=stacked_cells,
                inputs=seq_item_embed,
                sequence_length=self.user_interacted_len,
                initial_state=zero_state,
                time_major=False
            )
            out = state[-1][1] if self.rnn_type == "lstm" else state[-1]
            self.user_embed = tf.layers.dense(
                inputs=out, units=self.embed_size, activation=None
            )

    def _build_train_ops(self, **kwargs):
        if self.task == "rating":
            self.loss = tf.losses.mean_squared_error(labels=self.labels,
                                                     predictions=self.output)
        elif self.task == "ranking" and self.loss_type == "cross_entropy":
            self.loss = tf.reduce_mean(
                tf.nn.sigmoid_cross_entropy_with_logits(labels=self.labels,
                                                        logits=self.output)
            )
        elif self.task == "ranking" and self.loss_type == "bpr":
            self.loss = -tf.reduce_mean(self.log_sigmoid)

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

    def fit(self, train_data, verbose=1, shuffle=True,
            eval_data=None, metrics=None, **kwargs):
        self.show_start_time()
        if not self.graph_built:
            self._build_model()
            self._build_train_ops(**kwargs)

        if self.task == "rating" or self.loss_type == "cross_entropy":
            self._fit(train_data, verbose, shuffle, eval_data, metrics)
        elif self.loss_type == "bpr":
            self._fit_bpr(train_data, verbose, shuffle, eval_data, metrics)

    def _fit(self, train_data, verbose, shuffle, eval_data, metrics):
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
        assign_oov_vector(self)

    def _fit_bpr(self, train_data, verbose, shuffle, eval_data, metrics):
        data_generator = PairwiseSamplingSeq(
            dataset=train_data,
            data_info=self.data_info,
            num_neg=self.num_neg,
            mode=self.interaction_mode,
            num=self.max_seq_len
        )

        for epoch in range(1, self.n_epochs + 1):
            if self.lr_decay:
                print(f"With lr_decay, epoch {epoch} learning rate: "
                      f"{self.sess.run(self.lr)}")

            with time_block(f"Epoch {epoch}", verbose):
                train_total_loss = []
                for user, item_pos, item_neg, u_seq, u_len in data_generator(
                        shuffle, self.batch_size
                ):
                    u_len = np.asarray(u_len).astype(np.int64)
                    feed_dict = {self.user_interacted_seq: u_seq,
                                 self.user_interacted_len: u_len,
                                 self.item_indices_pos: item_pos,
                                 self.item_indices_neg: item_neg}
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
        assign_oov_vector(self)

    def predict(self, user, item, cold_start="average", inner_id=False):
        user, item = self.convert_id(user, item, inner_id)
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
        recos = self.user_vector[user_id] @ self.item_vector.T

        if self.task == "ranking":
            recos = 1 / (1 + np.exp(-recos))
        ids = np.argpartition(recos, -count)[-count:]
        rank = sorted(zip(ids, recos[ids]), key=lambda x: -x[1])
        recs_and_scores = islice(
            (rec if inner_id else (self.data_info.id2item[rec[0]], rec[1])
             for rec in rank if rec[0] not in consumed),
            n_rec
        )
        return list(recs_and_scores)

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
        feed_dict = {self.user_interacted_seq: self.user_last_interacted,
                     self.user_interacted_len: self.last_interacted_len}
        user_embed = self.sess.run(self.user_embed, feed_dict)
        item_weights = self.sess.run(self.item_weights)
        item_biases = self.sess.run(self.item_biases)

        user_bias = np.ones([len(user_embed), 1], dtype=user_embed.dtype)
        item_bias = item_biases[:, None]
        self.user_vector = np.hstack([user_embed, user_bias])
        self.item_vector = np.hstack([item_weights, item_bias])

    def _check_params(self):
        # assert self.hidden_units[-1] == self.embed_size, (
        #    "dimension of last rnn hidden unit should equal to embed_size"
        # )
        assert self.loss_type in ("cross_entropy", "bpr"), (
            "loss_type must be either cross_entropy or bpr"
        )
        assert self.rnn_type in ("lstm", "gru"), (
            "rnn_type must be either lstm or gru"
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
                                item_vector=self.item_vector)
        else:
            self.save_variables(path, model_name, inference_only=False)

    @classmethod
    def load(cls, path, model_name, data_info, manual=True):
        variable_path = os.path.join(path, f"{model_name}.npz")
        variables = np.load(variable_path)
        hparams = cls.load_params(path, data_info)
        model = cls(**hparams)
        model.user_vector = variables["user_vector"]
        model.item_vector = variables["item_vector"]
        return model
