import abc
import numpy as np
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


class TfMixin(object):
    def __init__(self, config=None, reg=None):
        self.reg_ = None

    def _reg_config(self):
        pass

    def train_pure(self, data_generator, verbose, shuffle, eval_data, metrics):
        for epoch in range(1, self.n_epochs + 1):
            with time_block(f"Epoch {epoch}", verbose):
                train_total_loss = []
                for user, item, label in data_generator(shuffle=shuffle):
                    train_loss, _ = self.sess.run([
                        self.loss, self.training_op
                    ],
                        feed_dict={self.user_indices: user,
                                   self.item_indices: item,
                                   self.labels: label})

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













