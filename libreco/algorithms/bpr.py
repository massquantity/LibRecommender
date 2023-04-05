"""Implementation of Bayesian Personalized Ranking."""
import logging
from functools import partial

import numpy as np

from ..bases import EmbedBase, ModelMeta
from ..evaluation import print_metrics
from ..recommendation import recommend_from_embedding
from ..tfops import reg_config, sess_config, tf
from ..training.dispatch import get_trainer
from ..utils.initializers import truncated_normal
from ..utils.misc import time_block
from ..utils.validate import check_fitting

LOG_FORMAT = "%(asctime)s - %(levelname)s - %(message)s"
logging.basicConfig(format=LOG_FORMAT)


class BPR(EmbedBase, metaclass=ModelMeta, backend="tensorflow"):
    """*Bayesian Personalized Ranking* algorithm.

    *BPR* is implemented in both TensorFlow and Cython.

    .. CAUTION::
        + BPR can only be used in ``ranking`` task.
        + BPR can only use ``bpr`` loss in ``loss_type``.

    Parameters
    ----------
    task : {'ranking'}
        Recommendation task. See :ref:`Task`.
    data_info : :class:`~libreco.data.DataInfo` object
        Object that contains useful information for training and inference.
    loss_type : {'bpr'}
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
    use_tf : bool, default: True
        Whether to use TensorFlow or Cython version. The TensorFlow version is more
        accurate, whereas the Cython version is faster.
    seed : int, default: 42
        Random seed.
    lower_upper_bound : tuple or None, default: None
        Lower and upper score bound for `rating` task.
    tf_sess_config : dict or None, default: None
        Optional TensorFlow session config, see `ConfigProto options
        <https://github.com/tensorflow/tensorflow/blob/v2.10.0/tensorflow/core/protobuf/config.proto#L431>`_.
    optimizer : {'sgd', 'momentum', 'adam'}, default: 'adam'
        Optimizer used in Cython version.
    num_threads : int, default: 1
        Number of threads used in Cython version.

    References
    ----------
    *Steffen Rendle et al.* `BPR: Bayesian Personalized Ranking from Implicit Feedback
    <https://arxiv.org/ftp/arxiv/papers/1205/1205.2618.pdf>`_.
    """

    user_variables = ["user_embed_var"]
    item_variables = ["item_embed_var", "item_bias_var"]

    def __init__(
        self,
        task="ranking",
        data_info=None,
        loss_type="bpr",
        embed_size=16,
        n_epochs=20,
        lr=0.001,
        lr_decay=False,
        epsilon=1e-5,
        reg=None,
        batch_size=256,
        sampler="random",
        num_neg=1,
        use_tf=True,
        seed=42,
        lower_upper_bound=None,
        tf_sess_config=None,
        optimizer="adam",
        num_threads=1,
    ):
        super().__init__(task, data_info, embed_size)

        assert task == "ranking", "BPR is only suitable for ranking"
        assert loss_type == "bpr", "BPR should use bpr loss"
        self.all_args = locals()
        self.loss_type = loss_type
        self.n_epochs = n_epochs
        self.lr = lr
        self.lr_decay = lr_decay
        self.epsilon = epsilon
        self.reg = reg_config(reg) if use_tf else reg
        self.batch_size = batch_size
        self.sampler = sampler
        self.num_neg = num_neg
        self.use_tf = use_tf
        self.seed = seed
        self.optimizer = optimizer
        self.num_threads = num_threads
        if use_tf:
            self.sess = sess_config(tf_sess_config)

    def build_model(self):
        if self.use_tf:
            self._build_model_tf()
        else:
            self._build_model_cython()

    def _build_model_cython(self):
        np.random.seed(self.seed)
        # last dimension is item bias, so for user all set to 1.0
        self.user_embed = truncated_normal(
            shape=(self.n_users, self.embed_size + 1), mean=0.0, scale=0.03
        )
        self.user_embed[:, self.embed_size] = 1.0
        self.item_embed = truncated_normal(
            shape=(self.n_items, self.embed_size + 1), mean=0.0, scale=0.03
        )
        self.item_embed[:, self.embed_size] = 0.0

    def _build_model_tf(self):
        self.user_indices = tf.placeholder(tf.int32, shape=[None])
        self.item_indices_pos = tf.placeholder(tf.int32, shape=[None])
        self.item_indices_neg = tf.placeholder(tf.int32, shape=[None])

        self.user_embed_var = tf.get_variable(
            name="user_embed_var",
            shape=[self.n_users, self.embed_size],
            initializer=tf.glorot_uniform_initializer(),
            regularizer=self.reg,
        )
        self.item_embed_var = tf.get_variable(
            name="item_embed_var",
            shape=[self.n_items, self.embed_size],
            initializer=tf.glorot_uniform_initializer(),
            regularizer=self.reg,
        )
        self.item_bias_var = tf.get_variable(
            name="item_bias_var",
            shape=[self.n_items],
            initializer=tf.zeros_initializer(),
            regularizer=self.reg,
        )

        embed_user = tf.nn.embedding_lookup(self.user_embed_var, self.user_indices)
        embed_item_pos = tf.nn.embedding_lookup(
            self.item_embed_var, self.item_indices_pos
        )
        embed_item_neg = tf.nn.embedding_lookup(
            self.item_embed_var, self.item_indices_neg
        )
        bias_item_pos = tf.nn.embedding_lookup(
            self.item_bias_var, self.item_indices_pos
        )
        bias_item_neg = tf.nn.embedding_lookup(
            self.item_bias_var, self.item_indices_neg
        )

        item_diff = tf.subtract(bias_item_pos, bias_item_neg) + tf.reduce_sum(
            tf.multiply(embed_user, tf.subtract(embed_item_pos, embed_item_neg)), axis=1
        )
        self.bpr_loss = tf.log_sigmoid(item_diff)

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
        """Fit BPR model on the training data.

        Parameters
        ----------
        train_data : :class:`~libreco.data.TransformedSet` object
            Data object used for training.
        neg_sampling : bool
            Whether to perform negative sampling for training or evaluating data.

            .. versionadded:: 1.1.0

        verbose : int, default: 1
            Print verbosity. If `eval_data` is provided, setting it to higher than 1
            will print evaluation metrics during training.
        shuffle : bool, default: True
            Whether to shuffle the training data.
        eval_data : :class:`~libreco.data.TransformedSet` object, default: None
            Data object used for evaluating.
        metrics : list or None, default: None
            List of metrics for evaluating.
        k : int, default: 10
            Parameter of metrics, e.g. recall at k, ndcg at k
        eval_batch_size : int, default: 8192
            Batch size for evaluating.
        eval_user_num : int or None, default: None
            Number of users for evaluating. Setting it to a positive number will sample
            users randomly from eval data.
        num_workers : int, default: 0
            How many subprocesses to use for data loading.
            0 means that the data will be loaded in the main process.
        """
        check_fitting(self, train_data, eval_data, neg_sampling, k)
        self.show_start_time()
        if not self.model_built:
            self.build_model()
            self.model_built = True
        if self.use_tf:
            if self.trainer is None:
                self.trainer = get_trainer(self)
            self.trainer.run(
                train_data,
                neg_sampling,
                verbose,
                shuffle,
                eval_data,
                metrics,
                k,
                eval_batch_size,
                eval_user_num,
                num_workers,
            )
            self.set_embeddings()
        else:
            self._fit_cython(
                train_data=train_data,
                neg_sampling=neg_sampling,
                verbose=verbose,
                shuffle=shuffle,
                eval_data=eval_data,
                metrics=metrics,
                k=k,
                eval_batch_size=eval_batch_size,
                eval_user_num=eval_user_num,
            )
        self.assign_embedding_oov()
        self.default_recs = recommend_from_embedding(
            model=self,
            user_ids=[self.n_users],
            n_rec=min(2000, self.n_items),
            user_embeddings=self.user_embed,
            item_embeddings=self.item_embed,
            seq=None,
            filter_consumed=False,
            random_rec=False,
        ).flatten()

    def _fit_cython(
        self,
        train_data,
        neg_sampling,
        verbose=1,
        shuffle=True,
        eval_data=None,
        metrics=None,
        k=None,
        eval_batch_size=None,
        eval_user_num=None,
    ):
        try:
            from ._bpr import bpr_update
        except (ImportError, ModuleNotFoundError):
            logging.warning("BPR cython version is not available")
            raise

        if self.optimizer == "sgd":
            trainer = partial(bpr_update)

        elif self.optimizer == "momentum":
            user_velocity = np.zeros_like(self.user_embed, dtype=np.float32)
            item_velocity = np.zeros_like(self.item_embed, dtype=np.float32)
            momentum = 0.9
            trainer = partial(
                bpr_update,
                u_velocity=user_velocity,
                i_velocity=item_velocity,
                momentum=momentum,
            )

        elif self.optimizer == "adam":
            # refer to the `Deep Learning` book,
            # which is called first and second moment
            user_1st_moment = np.zeros_like(self.user_embed, dtype=np.float32)
            item_1st_moment = np.zeros_like(self.item_embed, dtype=np.float32)
            user_2nd_moment = np.zeros_like(self.user_embed, dtype=np.float32)
            item_2nd_moment = np.zeros_like(self.item_embed, dtype=np.float32)
            rho1, rho2 = 0.9, 0.999
            trainer = partial(
                bpr_update,
                u_1st_mom=user_1st_moment,
                i_1st_mom=item_1st_moment,
                u_2nd_mom=user_2nd_moment,
                i_2nd_mom=item_2nd_moment,
                rho1=rho1,
                rho2=rho2,
            )

        else:
            raise ValueError(
                "optimizer must be one of these: ('sgd', 'momentum', 'adam')"
            )

        for epoch in range(1, self.n_epochs + 1):
            with time_block(f"Epoch {epoch}", verbose):
                trainer(
                    optimizer=self.optimizer,
                    train_data=train_data,
                    user_embed=self.user_embed,
                    item_embed=self.item_embed,
                    lr=self.lr,
                    reg=self.reg,
                    n_users=self.n_users,
                    n_items=self.n_items,
                    shuffle=shuffle,
                    num_threads=self.num_threads,
                    seed=self.seed,
                    epoch=epoch,
                )

            if verbose > 1:
                print_metrics(
                    model=self,
                    neg_sampling=neg_sampling,
                    eval_data=eval_data,
                    metrics=metrics,
                    eval_batch_size=eval_batch_size,
                    k=k,
                    sample_user_num=eval_user_num,
                    seed=self.seed,
                )
                print("=" * 30)

    def set_embeddings(self):
        item_bias, user_embed, item_embed = self.sess.run(
            [self.item_bias_var, self.user_embed_var, self.item_embed_var]
        )
        # to be compatible with cython version, bias is concatenated with embedding
        user_bias = np.ones([len(user_embed), 1], dtype=user_embed.dtype)
        item_bias = item_bias[:, None]
        self.user_embed = np.hstack([user_embed, user_bias])
        self.item_embed = np.hstack([item_embed, item_bias])
