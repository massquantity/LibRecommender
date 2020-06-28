import abc
import os
import multiprocessing
import numpy as np
import tensorflow as tf
from ..utils.timing import time_block
from ..utils.colorize import colorize
from ..utils.exception import NotSamplingError


class Base(abc.ABC):
    """Base class for all recommendation models.

    Parameters
    ----------
    data_info : `DataInfo` object
        Object that contains useful information for training and predicting.
    lower_upper_bound: list or tuple, optional
        Lower and upper score bound for rating task.
    """

    def __init__(self, task, data_info, lower_upper_bound=None):
        self.task = task
        if task == "rating":
            if lower_upper_bound is not None:
                assert isinstance(lower_upper_bound, (list, tuple)), (
                    "must contain both lower and upper bound if provided")
                self.lower_bound = lower_upper_bound[0]
                self.upper_bound = lower_upper_bound[1]
            else:
                self.lower_bound, self.upper_bound = data_info.min_max_rating
            print(f"lower bound: {self.lower_bound}, "
                  f"upper bound: {self.upper_bound}")

        elif task != "ranking":
            raise ValueError("task must be rating or ranking")

    @abc.abstractmethod
    def fit(self, train_data, **kwargs):
        """Train model on the training data.

        Parameters
        ----------
        train_data : `TransformedSet` object
            Data object used for training.
        """
        raise NotImplementedError

    @abc.abstractmethod
    def predict(self, user, item):
        """Predict score for given user and item.

        Parameters
        ----------
        user: int or array_like
            User id or batch of user ids.
        item: int or array_like
            Item id or batch of item ids.

        Returns
        -------
        prediction: int or array_like
            Predicted scores for each user-item pair.
        """
        raise NotImplementedError

    @abc.abstractmethod
    def recommend_user(self, user, n_rec, **kwargs):
        """Recommend a list of items for given user.

        Parameters
        ----------
        user: int
            User id to recommend.
        n_rec: int
            number of recommendations to return.

        Returns
        -------
        result: list of tuples
            A recommendation list, each recommendation
            contains an (item_id, score) tuple.

        """
        raise NotImplementedError

    def _check_unknown(self, user, item):
        unknown_user_indices = list(
            np.where(np.logical_or(user >= self.n_users, user < 0))[0])
        unknown_item_indices = list(
            np.where(np.logical_or(item >= self.n_items, item < 0))[0])

        unknown_user = list(user[unknown_user_indices]) if (
            unknown_user_indices) else None
        unknown_item = list(item[unknown_item_indices]) if (
            unknown_item_indices) else None
        unknown_index = list(set(unknown_user_indices) |
                             set(unknown_item_indices))
        unknown_num = len(unknown_index)

        if unknown_num > 0:
            user[unknown_index] = 0   # temp conversion
            item[unknown_index] = 0
            unknown_str = (f"detect {unknown_num} unknown interaction(s), "
                           f"including user: {unknown_user}, "
                           f"item: {unknown_item}, "
                           f"will be handled as default prediction")
            print(f"{colorize(unknown_str, 'red')}")

        return unknown_num, unknown_index, user, item

    def _check_unknown_user(self, user):
        if 0 <= user < self.n_users:
            return user
        else:
            unknown_str = (f"detect unknown user {user}, "
                           f"return cold start recommendation")
            print(f"{colorize(unknown_str, 'red')}")
            return

    def _check_has_sampled(self, data, verbose):
        if not data.has_sampled and verbose > 1:
            exception_str = (f"When using batch sampling, "
                             f"one must do whole data sampling "
                             f"before evaluating on epochs.")
            raise NotSamplingError(f"{colorize(exception_str, 'red')}")

    def _decide_dense_values(self, data_info):
        return False if not data_info.dense_col else True

    def _sparse_feat_size(self, data_info):
        if (data_info.user_sparse_unique is not None
                and data_info.item_sparse_unique is not None):
            return max(np.max(data_info.user_sparse_unique),
                       np.max(data_info.item_sparse_unique)) + 1
        elif data_info.user_sparse_unique is not None:
            return np.max(data_info.user_sparse_unique) + 1
        elif data_info.item_sparse_unique is not None:
            return np.max(data_info.item_sparse_unique) + 1

    def _sparse_field_size(self, data_info):
        return len(data_info.sparse_col.name)

    def _dense_field_size(self, data_info):
        return len(data_info.dense_col.name)


class TfMixin(object):
    def __init__(self, tf_sess_config=None):
        self.cpu_num = multiprocessing.cpu_count()
        self.sess = self._sess_config(tf_sess_config)

    def _sess_config(self, tf_sess_config=None):
        if not tf_sess_config:
            # Session config based on:
            # https://software.intel.com/content/www/us/en/develop/articles/tips-to-improve-performance-for-popular-deep-learning-frameworks-on-multi-core-cpus.html
            tf_sess_config = {
                "intra_op_parallelism_threads": 0,
                "inter_op_parallelism_threads": 0,
                "allow_soft_placement": True,
                "device_count": {"CPU": self.cpu_num}
            }
        #    os.environ["OMP_NUM_THREADS"] = f"{self.cpu_num}"

        config = tf.ConfigProto(**tf_sess_config)
        return tf.Session(config=config)

    def train_pure(self, data_generator, verbose, shuffle, eval_data, metrics):
        for epoch in range(1, self.n_epochs + 1):
            with time_block(f"Epoch {epoch}", verbose):
                train_total_loss = []
                for user, item, label in data_generator(shuffle=shuffle):
                    train_loss, _ = self.sess.run(
                        [self.loss, self.training_op],
                        feed_dict={self.user_indices: user,
                                   self.item_indices: item,
                                   self.labels: label,
                                   self.is_training: True})

                    train_total_loss.append(train_loss)

            if verbose > 1:
                train_loss_str = "train_loss: " + str(
                    round(np.mean(train_total_loss), 4)
                )
                print(f"\t {colorize(train_loss_str, 'green')}")

                class_name = self.__class__.__name__.lower()
                if class_name.startswith("svd"):
                    # set up parameters for prediction evaluate
                    self._set_latent_factors()

                self.print_metrics(eval_data=eval_data, metrics=metrics)
                print("="*30)

    def train_feat(self, data_generator, verbose, shuffle, eval_data, metrics):
        for epoch in range(1, self.n_epochs + 1):
            with time_block(f"Epoch {epoch}", verbose):
                train_total_loss = []
                for si, di, dv, label in data_generator(shuffle=shuffle):
                    feed_dict = self._get_feed_dict(si, di, dv, label, True)
                    train_loss, _ = self.sess.run(
                        [self.loss, self.training_op], feed_dict)
                    train_total_loss.append(train_loss)

            if verbose > 1:
                train_loss_str = "train_loss: " + str(
                    round(np.mean(train_total_loss), 4)
                )
                print(f"\t {colorize(train_loss_str, 'green')}")
                self.print_metrics(eval_data=eval_data, metrics=metrics)
                print("="*30)

    def _get_feed_dict(self, sparse_indices, dense_indices,
                       dense_values, label, is_training):
        feed_dict = {self.sparse_indices: sparse_indices,
                     self.is_training: is_training}
        if self.dense:
            feed_dict.update({self.dense_indices: dense_indices,
                              self.dense_values: dense_values})
        if label is not None:
            feed_dict.update({self.labels: label})
        return feed_dict
