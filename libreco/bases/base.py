import abc
import time

from ..utils.constants import PURE_MODELS, FEAT_MODELS, SEQUENCE_MODELS
from ..utils.misc import colorize


class Base(abc.ABC):
    """Base class for all recommendation models.

    Parameters
    ----------
    task : str
        Specific task, either rating or ranking.
    data_info : `DataInfo` object
        Object that contains useful information for training and predicting.
    lower_upper_bound : list or tuple, optional
        Lower and upper score bound for rating task.
    """

    def __init__(self, task, data_info, lower_upper_bound=None):
        self.model_name = self.__class__.__name__
        self.model_category = self.get_model_category()
        self.task = task
        self.data_info = data_info
        self.n_users = data_info.n_users
        self.n_items = data_info.n_items
        self.user_consumed = data_info.user_consumed
        if task == "rating":
            self.global_mean = data_info.global_mean
            if lower_upper_bound is not None:
                assert isinstance(
                    lower_upper_bound, (list, tuple)
                ), "must contain both lower and upper bound if provided"
                self.lower_bound = lower_upper_bound[0]
                self.upper_bound = lower_upper_bound[1]
            else:
                self.lower_bound, self.upper_bound = data_info.min_max_rating

        elif task != "ranking":
            raise ValueError("task must either be rating or ranking")

        self.default_prediction = data_info.global_mean if task == "rating" else 0.0

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
    def predict(self, user, item, **kwargs):
        """Predict score for given user and item.

        Parameters
        ----------
        user : int or array_like
            User id or batch of user ids.
        item : int or array_like
            Item id or batch of item ids.

        Returns
        -------
        prediction : int or array_like
            Predicted scores for each user-item pair.
        """
        raise NotImplementedError

    @abc.abstractmethod
    def recommend_user(self, user, n_rec, **kwargs):
        """Recommend a list of items for given user.

        Parameters
        ----------
        user : int
            User id to recommend.
        n_rec : int
            number of recommendations to return.

        Returns
        -------
        result : list of tuples
            A recommendation list, each recommendation
            contains an (item_id, score) tuple.
        """
        raise NotImplementedError

    @abc.abstractmethod
    def save(self, path, model_name, **kwargs):
        """save model for inference or retraining.

        Parameters
        ----------
        path : str
            file folder path to save model.
        model_name : str
            name of the saved model file.
        """
        raise NotImplementedError

    @classmethod
    @abc.abstractmethod
    def load(cls, path, model_name, data_info, **kwargs):
        """load saved model for inference.

        Parameters
        ----------
        path : str
            file folder path to save model.
        model_name : str
            name of the saved model file.
        data_info : `DataInfo` object
            Object that contains some useful information.
        """
        raise NotImplementedError

    @staticmethod
    def show_start_time():
        start_time = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())
        print(f"Training start time: {colorize(start_time, 'magenta')}")

    def get_model_category(self):
        if self.model_name in SEQUENCE_MODELS:
            return "sequence"
        elif self.model_name in FEAT_MODELS:
            return "feat"
        elif self.model_name in PURE_MODELS:
            return "pure"
        else:
            raise ValueError(f"unknown model: {self.model_name}")
