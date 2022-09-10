import numpy as np

from ..data.data_generator import SparseTensorSequence
from ..evaluation import print_metrics
from ..tfops import choose_tf_loss, get_feed_dict, lr_decay_config, tf, var_list_by_name
from ..training.trainer import BaseTrainer
from ..utils.constants import EMBEDDING_MODELS
from ..utils.misc import colorize, time_block
from ..utils.sampling import PairwiseSampling, PairwiseSamplingSeq


class TensorFlowTrainer(BaseTrainer):
    def __init__(
        self,
        model,
        task,
        loss_type,
        n_epochs,
        lr,
        lr_decay,
        batch_size,
        num_neg,
        k,
        eval_batch_size,
        eval_user_num,
        *args,
        **kwargs,
    ):
        super().__init__(
            model,
            task,
            loss_type,
            n_epochs,
            lr,
            lr_decay,
            batch_size,
            num_neg,
            k,
            eval_batch_size,
            eval_user_num,
        )
        self.sess = model.sess
        self.use_reg = self._check_reg()
        self._build_train_ops(*args, **kwargs)

    def run(self, train_data, verbose, shuffle, eval_data, metrics):
        data_generator = self.get_data_generator(train_data)
        for epoch in range(1, self.n_epochs + 1):
            if self.lr_decay:
                print(
                    f"With lr_decay, epoch {epoch} learning rate: "
                    f"{self.sess.run(self.lr)}"
                )
            with time_block(f"Epoch {epoch}", verbose):
                train_total_loss = []
                for data in data_generator(shuffle, self.batch_size):
                    user_indices = data[0]
                    item_indices = data[1]
                    labels = data[2]
                    sparse_indices = data[3] if len(data) >= 4 else None
                    dense_values = data[4] if len(data) >= 5 else None
                    user_interacted_seq = data[5] if len(data) >= 6 else None
                    user_interacted_len = data[6] if len(data) >= 7 else None
                    feed_dict = get_feed_dict(
                        self.model,
                        user_indices,
                        item_indices,
                        labels,
                        sparse_indices,
                        dense_values,
                        user_interacted_seq,
                        user_interacted_len,
                        is_training=True,
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
                # get embedding for evaluation
                if self.model.model_name in EMBEDDING_MODELS:
                    self.model.set_embeddings()
                print_metrics(
                    model=self.model,
                    eval_data=eval_data,
                    metrics=metrics,
                    eval_batch_size=self.eval_batch_size,
                    k=self.k,
                    sample_user_num=self.eval_user_num,
                    seed=self.model.seed,
                )
                print("=" * 30)

    def _build_train_ops(self, **kwargs):
        self.loss = choose_tf_loss(self.model, self.task, self.loss_type)
        if self.use_reg:
            reg_keys = tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES)
            total_loss = self.loss + tf.add_n(reg_keys)
        else:
            total_loss = self.loss

        if self.lr_decay:
            n_batches = int(self.model.data_info.data_size / self.batch_size)
            self.lr, global_steps = lr_decay_config(self.lr, n_batches, **kwargs)
        else:
            self.lr, global_steps = self.lr, None

        optimizer = tf.train.AdamOptimizer(self.lr)
        optimizer_op = optimizer.minimize(total_loss, global_step=global_steps)
        update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
        self.training_op = tf.group([optimizer_op, update_ops])
        self.sess.run(tf.global_variables_initializer())

    def _check_reg(self):
        if hasattr(self.model, "reg") and self.model.reg is not None:
            return True
        else:
            return False


class YoutubeRetrievalTrainer(TensorFlowTrainer):
    def __init__(
        self,
        model,
        task,
        loss_type,
        n_epochs,
        lr,
        lr_decay,
        batch_size,
        num_sampled_per_batch,
        k,
        eval_batch_size,
        eval_user_num,
        sampler,
    ):
        super().__init__(
            model,
            task,
            loss_type,
            n_epochs,
            lr,
            lr_decay,
            batch_size,
            1,
            k,
            eval_batch_size,
            eval_user_num,
            num_sampled_per_batch,
            sampler,
        )

    def run(self, train_data, verbose, shuffle, eval_data, metrics):
        data_generator = SparseTensorSequence(
            train_data,
            self.model.data_info,
            self.model.sparse,
            self.model.dense,
            self.model.interaction_mode,
            self.model.max_seq_len,
            self.model.n_items,
        )
        for epoch in range(1, self.n_epochs + 1):
            if self.lr_decay:
                print(
                    f"With lr_decay, epoch {epoch} learning rate: "
                    f"{self.sess.run(self.lr)}"
                )
            with time_block(f"Epoch {epoch}", verbose):
                train_total_loss = []
                for data in data_generator(shuffle, self.batch_size):
                    feed_dict = {
                        self.model.modified_batch_size: data[0],
                        self.model.item_interaction_indices: data[1],
                        self.model.item_interaction_values: data[2],
                        self.model.item_indices: data[3],
                        self.model.is_training: True,
                    }
                    if self.model.sparse:
                        feed_dict.update({self.model.sparse_indices: data[4]})
                    if self.model.dense:
                        feed_dict.update({self.model.dense_values: data[5]})
                    train_loss, _ = self.sess.run(
                        [self.loss, self.training_op], feed_dict
                    )
                    train_total_loss.append(train_loss)

            if verbose > 1:
                train_loss_str = "train_loss: " + str(
                    round(float(np.mean(train_total_loss)), 4)
                )
                print(f"\t {colorize(train_loss_str, 'green')}")
                self.model.set_embeddings()
                print_metrics(
                    model=self.model,
                    eval_data=eval_data,
                    metrics=metrics,
                    eval_batch_size=self.eval_batch_size,
                    k=self.k,
                    sample_user_num=self.eval_user_num,
                    seed=self.model.seed,
                )
                print("=" * 30)

    def _build_train_ops(self, num_sampled_per_batch, sampler, **kwargs):
        num_sampled_per_batch = (
            num_sampled_per_batch
            if num_sampled_per_batch and num_sampled_per_batch > 0
            else self.batch_size
        )
        # By default, `sampled_softmax_loss` and `nce_loss` in tensorflow
        # uses `log_uniform_candidate_sampler` to sample negative items,
        # which may not be suitable in recommendation scenarios.
        labels = tf.reshape(self.model.item_indices, [-1, 1])
        sampled_values = (
            tf.random.uniform_candidate_sampler(
                true_classes=labels,
                num_true=1,
                num_sampled=num_sampled_per_batch,
                unique=True,
                range_max=self.model.n_items,
            )
            if sampler == "uniform"
            else None
        )

        if self.loss_type == "nce":
            self.loss = tf.reduce_mean(
                tf.nn.nce_loss(
                    weights=self.model.nce_weights,
                    biases=self.model.nce_biases,
                    labels=labels,
                    inputs=self.model.user_vector_repr,
                    num_sampled=num_sampled_per_batch,
                    num_classes=self.model.n_items,
                    num_true=1,
                    sampled_values=sampled_values,
                    remove_accidental_hits=True,
                    partition_strategy="div",
                )
            )
        elif self.loss_type == "sampled_softmax":
            self.loss = tf.reduce_mean(
                tf.nn.sampled_softmax_loss(
                    weights=self.model.nce_weights,
                    biases=self.model.nce_biases,
                    labels=labels,
                    inputs=self.model.user_vector_repr,
                    num_sampled=num_sampled_per_batch,
                    num_classes=self.model.n_items,
                    num_true=1,
                    sampled_values=sampled_values,
                    remove_accidental_hits=True,
                    seed=self.model.seed,
                    partition_strategy="div",
                )
            )
        else:
            raise ValueError("Loss type must either be 'nce' or 'sampled_softmax")

        if self.use_reg:
            reg_keys = tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES)
            total_loss = self.loss + tf.add_n(reg_keys)
        else:
            total_loss = self.loss

        if self.lr_decay:
            n_batches = int(self.model.data_info.data_size / self.batch_size)
            self.lr, global_steps = lr_decay_config(self.lr, n_batches, **kwargs)
        else:
            global_steps = None

        optimizer = tf.train.AdamOptimizer(self.lr)
        optimizer_op = optimizer.minimize(total_loss, global_step=global_steps)
        update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
        self.training_op = tf.group([optimizer_op, update_ops])
        self.sess.run(tf.global_variables_initializer())


class BPRTrainer(TensorFlowTrainer):
    def __init__(
        self,
        model,
        task,
        loss_type,
        n_epochs,
        lr,
        lr_decay,
        batch_size,
        num_neg,
        k,
        eval_batch_size,
        eval_user_num,
    ):
        super().__init__(
            model,
            task,
            loss_type,
            n_epochs,
            lr,
            lr_decay,
            batch_size,
            num_neg,
            k,
            eval_batch_size,
            eval_user_num,
        )

    def run(self, train_data, verbose, shuffle, eval_data, metrics):
        data_generator = PairwiseSampling(
            train_data, self.model.data_info, self.num_neg
        )
        for epoch in range(1, self.n_epochs + 1):
            if self.lr_decay:
                print(
                    f"With lr_decay, epoch {epoch} learning rate: "
                    f"{self.sess.run(self.lr)}"
                )
            with time_block(f"Epoch {epoch}", verbose):
                for data in data_generator(shuffle, self.batch_size):
                    self.sess.run(
                        self.training_op,
                        feed_dict={
                            self.model.user_indices: data[0],
                            self.model.item_indices_pos: data[1],
                            self.model.item_indices_neg: data[2],
                        },
                    )

            if verbose > 1:
                # get embedding for evaluation
                self.model.set_embeddings()
                print_metrics(
                    model=self.model,
                    eval_data=eval_data,
                    metrics=metrics,
                    eval_batch_size=self.eval_batch_size,
                    k=self.k,
                    sample_user_num=self.eval_user_num,
                    seed=self.model.seed,
                )
                print("=" * 30)


class RNN4RecTrainer(TensorFlowTrainer):
    def __init__(
        self,
        model,
        task,
        loss_type,
        n_epochs,
        lr,
        lr_decay,
        batch_size,
        num_neg,
        k,
        eval_batch_size,
        eval_user_num,
    ):
        super().__init__(
            model,
            task,
            loss_type,
            n_epochs,
            lr,
            lr_decay,
            batch_size,
            num_neg,
            k,
            eval_batch_size,
            eval_user_num,
        )

    def run(self, train_data, verbose, shuffle, eval_data, metrics):
        if self.task == "rating" or self.loss_type in ("cross_entropy", "focal"):
            super().run(train_data, verbose, shuffle, eval_data, metrics)
        elif self.loss_type == "bpr":
            self._run_bpr(train_data, verbose, shuffle, eval_data, metrics)
        else:
            raise ValueError(f"unknown task or loss: {self.task}, {self.loss_type}")

    def _run_bpr(self, train_data, verbose, shuffle, eval_data, metrics):
        data_generator = PairwiseSamplingSeq(
            dataset=train_data,
            data_info=self.model.data_info,
            num_neg=self.num_neg,
            mode=self.model.interaction_mode,
            num=self.model.max_seq_len,
        )
        for epoch in range(1, self.n_epochs + 1):
            if self.lr_decay:
                print(
                    f"With lr_decay, epoch {epoch} learning rate: "
                    f"{self.sess.run(self.lr)}"
                )
            with time_block(f"Epoch {epoch}", verbose):
                train_total_loss = []
                for _, item_pos, item_neg, u_seq, u_len in data_generator(
                    shuffle, self.batch_size
                ):
                    u_len = np.asarray(u_len).astype(np.int64)
                    feed_dict = {
                        self.model.user_interacted_seq: u_seq,
                        self.model.user_interacted_len: u_len,
                        self.model.item_indices_pos: item_pos,
                        self.model.item_indices_neg: item_neg,
                    }
                    train_loss, _ = self.sess.run(
                        [self.loss, self.training_op], feed_dict
                    )
                    train_total_loss.append(train_loss)

            if verbose > 1:
                train_loss_str = "train_loss: " + str(
                    round(float(np.mean(train_total_loss)), 4)
                )
                print(f"\t {colorize(train_loss_str, 'green')}")
                # get embedding for evaluation
                self.model.set_embeddings()
                print_metrics(
                    model=self.model,
                    eval_data=eval_data,
                    metrics=metrics,
                    eval_batch_size=self.eval_batch_size,
                    k=self.k,
                    sample_user_num=self.eval_user_num,
                    seed=self.model.seed,
                )
                print("=" * 30)


class WideDeepTrainer(TensorFlowTrainer):
    def __init__(
        self,
        model,
        task,
        loss_type,
        n_epochs,
        lr,
        lr_decay,
        batch_size,
        num_neg,
        k,
        eval_batch_size,
        eval_user_num,
    ):
        super().__init__(
            model,
            task,
            loss_type,
            n_epochs,
            lr,
            lr_decay,
            batch_size,
            num_neg,
            k,
            eval_batch_size,
            eval_user_num,
        )

    def _build_train_ops(self, **kwargs):
        self.loss = choose_tf_loss(self.model, self.task, self.loss_type)
        if self.use_reg:
            reg_keys = tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES)
            total_loss = self.loss + tf.add_n(reg_keys)
        else:
            total_loss = self.loss

        if self.lr_decay:
            n_batches = int(self.model.data_info.data_size / self.batch_size)
            self.lr["wide"], wide_global_steps = lr_decay_config(
                self.lr["wide"], n_batches, **kwargs
            )
            self.lr["deep"], deep_global_steps = lr_decay_config(
                self.lr["deep"], n_batches, **kwargs
            )
        else:
            wide_global_steps = deep_global_steps = None

        var_dict = var_list_by_name(names=["wide", "deep"])
        # print(f"{colorize('Wide_variables', 'blue')}: {var_dict['wide']}\n"
        #       f"{colorize('Deep_variables', 'blue')}: {var_dict['deep']}")
        wide_optimizer = tf.train.FtrlOptimizer(
            self.lr["wide"], l1_regularization_strength=1e-3
        )
        wide_optimizer_op = wide_optimizer.minimize(
            loss=total_loss, global_step=wide_global_steps, var_list=var_dict["wide"]
        )
        deep_optimizer = tf.train.AdamOptimizer(self.lr["deep"])
        deep_optimizer_op = deep_optimizer.minimize(
            loss=total_loss, global_step=deep_global_steps, var_list=var_dict["deep"]
        )

        update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
        self.training_op = tf.group([wide_optimizer_op, deep_optimizer_op, update_ops])
        self.sess.run(tf.global_variables_initializer())
