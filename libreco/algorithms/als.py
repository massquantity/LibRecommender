"""Implementation of Alternating Least Squares."""
import logging
import os
from functools import partial

import numpy as np

from ..bases import EmbedBase
from ..evaluation import print_metrics
from ..recommendation import recommend_from_embedding
from ..utils.initializers import truncated_normal
from ..utils.misc import time_block
from ..utils.save_load import save_default_recs, save_params
from ..utils.validate import check_fitting

LOG_FORMAT = "%(asctime)s - %(levelname)s - %(message)s"
logging.basicConfig(format=LOG_FORMAT)


class ALS(EmbedBase):
    """*Alternating Least Squares* algorithm.

    One can use conjugate gradient optimization and set more `n_threads`
    to accelerate training.

    Parameters
    ----------
    task : {'rating', 'ranking'}
        Recommendation task. See :ref:`Task`.
    data_info : :class:`~libreco.data.DataInfo` object
          Object that contains useful information for training and inference.
    embed_size : int, default: 16
        Vector size of embeddings.
    n_epochs : int, default: 10
        Number of epochs for training.
    reg : float or None, default: None
        Regularization parameter, must be non-negative or None.
    alpha : int, default: 10
        Parameter used for increasing confidence level, only applied for `ranking` task.
    use_cg : bool, default: True
        Whether to use *conjugate gradient* optimization. See `reference <http://citeseerx.ist.psu.edu/viewdoc/download?doi=10.1.1.379.6473&rep=rep1&type=pdf>`_.
    n_threads : int, default: 1
        Number of threads to use.
    seed : int, default: 42
        Random seed.
    lower_upper_bound : tuple or None, default: None
        Lower and upper score bound for `rating` task.

    References
    ----------
    [1] *Haoming Li et al.* `Matrix Completion via Alternating Least Square(ALS)
    <https://stanford.edu/~rezab/classes/cme323/S15/notes/lec14.pdf>`_.

    [2] *Yifan Hu et al.* `Collaborative Filtering for Implicit Feedback Datasets
    <http://yifanhu.net/PUB/cf.pdf>`_.

    [3] *Gábor Takács et al.* `Applications of the Conjugate Gradient Method for Implicit Feedback Collaborative Filtering
    <http://citeseerx.ist.psu.edu/viewdoc/download?doi=10.1.1.379.6473&rep=rep1&type=pdf>`_.
    """

    def __init__(
        self,
        task,
        data_info,
        embed_size=16,
        n_epochs=10,
        reg=None,
        alpha=10,
        use_cg=True,
        n_threads=1,
        seed=42,
        lower_upper_bound=None,
    ):
        super().__init__(task, data_info, embed_size, lower_upper_bound)

        self.all_args = locals()
        self.n_epochs = n_epochs
        self.reg = self._check_reg(reg)
        self.alpha = alpha
        self.use_cg = use_cg
        self.n_threads = n_threads
        self.seed = seed

    def build_model(self):
        np.random.seed(self.seed)
        self.user_embed = truncated_normal(
            shape=[self.n_users, self.embed_size], mean=0.0, scale=0.03
        )
        self.item_embed = truncated_normal(
            shape=[self.n_items, self.embed_size], mean=0.0, scale=0.03
        )

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
        **kwargs,
    ):
        """Fit ALS model on the training data.

        Parameters
        ----------
        train_data : :class:`~libreco.data.TransformedSet` object
            Data object used for training.
        neg_sampling : bool
            Whether to perform negative sampling for evaluating data.

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
        """
        try:
            from ._als import als_update
        except (ImportError, ModuleNotFoundError):
            logging.warning("Als cython version is not available")
            raise

        check_fitting(self, train_data, eval_data, neg_sampling, k)
        self.show_start_time()
        self.build_model()
        user_interaction = train_data.sparse_interaction  # sparse.csr_matrix
        item_interaction = user_interaction.T.tocsr()
        if self.task == "ranking":
            user_interaction.data = user_interaction.data * self.alpha + 1
            item_interaction.data = item_interaction.data * self.alpha + 1

        trainer = partial(als_update, task=self.task, use_cg=self.use_cg)
        for epoch in range(1, self.n_epochs + 1):
            with time_block(f"Epoch {epoch}", verbose):
                trainer(
                    interaction=user_interaction,
                    X=self.user_embed,
                    Y=self.item_embed,
                    reg=self.reg,
                    num_threads=self.n_threads,
                )
                trainer(
                    interaction=item_interaction,
                    X=self.item_embed,
                    Y=self.user_embed,
                    reg=self.reg,
                    num_threads=self.n_threads,
                )

            if verbose > 1:
                print_metrics(
                    model=self,
                    neg_sampling=neg_sampling,
                    train_data=train_data,
                    eval_data=eval_data,
                    metrics=metrics,
                    eval_batch_size=eval_batch_size,
                    k=k,
                    sample_user_num=eval_user_num,
                    seed=self.seed,
                )
                print("=" * 30)
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

    @staticmethod
    def _check_reg(reg):
        if not isinstance(reg, float) or reg <= 0.0:
            raise ValueError(f"`reg` must be float and positive, got {reg}")
        return reg

    def save(self, path, model_name, **kwargs):
        """Save model for inference or retraining.

        Parameters
        ----------
        path : str
            File folder path to save model.
        model_name : str
            Name of the saved model file.
        """
        if not os.path.isdir(path):
            print(f"file folder {path} doesn't exists, creating a new one...")
            os.makedirs(path)
        save_params(self, path, model_name)
        save_default_recs(self, path, model_name)
        variable_path = os.path.join(path, model_name)
        np.savez_compressed(
            variable_path, user_embed=self.user_embed, item_embed=self.item_embed
        )

    def set_embeddings(self):  # pragma: no cover
        pass

    def rebuild_model(self, path, model_name):
        """Reconstruct model for retraining.

        Parameters
        ----------
        path : str
            File folder path for saved model.
        model_name : str
            Name of the saved model file.
        """
        variable_path = os.path.join(path, f"{model_name}.npz")
        variables = np.load(variable_path)
        # remove oov values
        old_var = variables["user_embed"][:-1]
        self.user_embed[: len(old_var)] = old_var
        old_var = variables["item_embed"][:-1]
        self.item_embed[: len(old_var)] = old_var


def least_squares(sparse_interaction, X, Y, reg, embed_size, num, mode):
    """Least squares optimization showcase for ALS."""
    indices = sparse_interaction.indices
    indptr = sparse_interaction.indptr
    data = sparse_interaction.data
    if mode == "explicit":
        for m in range(num):
            m_slice = slice(indptr[m], indptr[m + 1])
            interacted = Y[indices[m_slice]]
            labels = data[m_slice]
            A = interacted.T @ interacted + reg * np.eye(embed_size)
            b = interacted.T @ labels
            X[m] = np.linalg.solve(A, b)
    elif mode == "implicit":
        init_A = Y.T @ Y + reg * np.eye(embed_size, dtype=np.float32)
        for m in range(num):
            A = init_A.copy()
            b = np.zeros(embed_size, dtype=np.float32)
            for i in range(indptr[m], indptr[m + 1]):
                factor = Y[indices[i]]
                confidence = data[i]
                # If confidence = 1, r_ui = 0 means no interaction.
                A += (confidence - 1) * np.outer(factor, factor)
                b += confidence * factor
            X[m] = np.linalg.solve(A, b)
    else:
        raise ValueError("mode must either be 'explicit' or 'implicit'")


# O(f^3) * m
def least_squares_cg(sparse_interaction, X, Y, reg, embed_size, num, mode, cg_steps=3):
    """Conjugate Gradient optimization showcase for ALS."""
    indices = sparse_interaction.indices
    indptr = sparse_interaction.indptr
    data = sparse_interaction.data
    if mode == "explicit":
        for m in range(num):
            m_slice = slice(indptr[m], indptr[m + 1])
            interacted = Y[indices[m_slice]]
            labels = data[m_slice]
            A = interacted.T @ interacted + reg * np.eye(embed_size)
            b = interacted.T @ labels
            X[m] = np.linalg.solve(A, b)
    elif mode == "implicit":
        init_A = Y.T @ Y + reg * np.eye(embed_size, dtype=np.float32)
        for m in range(num):
            x = X[m]
            r = -init_A @ x
            # compute r = b - Ax
            for i in range(indptr[m], indptr[m + 1]):
                y = Y[indices[i]]
                confidence = data[i]
                r += (confidence - (confidence - 1) * (y @ x)) * y

            p = r.copy()
            rs_old = r @ r
            if rs_old < 1e-10:
                continue

            for _ in range(cg_steps):
                Ap = init_A @ p
                for i in range(indptr[m], indptr[m + 1]):
                    y = Y[indices[i]]
                    confidence = data[i]
                    Ap += (confidence - 1) * (y @ p) * y

                # standard CG update
                ak = rs_old / (p @ Ap)
                x += ak * p
                r -= ak * Ap
                rs_new = r @ r
                if rs_new < 1e-10:
                    break
                p = r + (rs_new / rs_old) * p
                rs_old = rs_new

            X[m] = x

    else:
        raise ValueError("mode must either be 'explicit' or 'implicit'")
