"""Recommendation model base class."""
import abc
import time

from ..utils.constants import FEAT_MODELS, PURE_MODELS, SEQUENCE_MODELS
from ..utils.misc import colorize


class Base(abc.ABC):
    """Base class for all recommendation models.

    Parameters
    ----------
    task : {'rating', 'ranking'}
        Recommendation task. See :ref:`Task`.
    data_info : :class:`~libreco.data.DataInfo` object
        Object that contains useful information for training and inference.
    lower_upper_bound : list or tuple, default: None
        Lower and upper score bound for rating task.
    """

    def __init__(self, task, data_info, lower_upper_bound=None):
        self.model_name = self.__class__.__name__
        self.model_category = self._get_model_category()
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

        self.default_pred = data_info.global_mean if task == "rating" else 0.0
        self.default_recs = None

    @abc.abstractmethod
    def fit(self, train_data, neg_sampling, **kwargs):
        """Fit model on the training data.

        Parameters
        ----------
        train_data : :class:`~libreco.data.TransformedSet` object
            Data object used for training.
        neg_sampling : bool
            Whether to perform negative sampling for training or evaluating data.
        """
        raise NotImplementedError

    @abc.abstractmethod
    def predict(self, user, item, **kwargs):
        """Predict score for given user and item.

        Parameters
        ----------
        user : int or str or array_like
            User id or batch of user ids.
        item : int or str or array_like
            Item id or batch of item ids.

        Returns
        -------
        prediction : float or numpy.ndarray
            Predicted scores for each user-item pair.
        """
        raise NotImplementedError

    @abc.abstractmethod
    def recommend_user(self, user, n_rec, **kwargs):
        """Recommend a list of items for given user.

        Parameters
        ----------
        user : int or str or array_like
            User id or batch of user ids to recommend.
        n_rec : int
            Number of recommendations to return.

        Returns
        -------
        recommendation : dict of {Union[int, str, array_like] : numpy.ndarray}
            Recommendation result with user ids as keys
            and array_like recommended items as values.
        """
        raise NotImplementedError

    @abc.abstractmethod
    def save(self, path, model_name, **kwargs):
        """Save model for inference or retraining.

        Parameters
        ----------
        path : str
            File folder path to save model.
        model_name : str
            Name of the saved model file.

        See Also
        --------
        load
        """
        raise NotImplementedError

    @classmethod
    @abc.abstractmethod
    def load(cls, path, model_name, data_info, **kwargs):
        """Load saved model for inference.

        Parameters
        ----------
        path : str
            File folder path to save model.
        model_name : str
            Name of the saved model file.
        data_info : :class:`~libreco.data.DataInfo` object
            Object that contains some useful information.

        Returns
        -------
        model : type(cls)
            Loaded model.

        See Also
        --------
        save
        """
        raise NotImplementedError

    @staticmethod
    def show_start_time():
        start_time = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())
        print(f"Training start time: {colorize(start_time, 'magenta')}")

    def _get_model_category(self):
        if self.model_name in SEQUENCE_MODELS:
            return "sequence"
        elif self.model_name in FEAT_MODELS:
            return "feat"
        elif self.model_name in PURE_MODELS:
            return "pure"
        else:
            raise ValueError(f"unknown model: {self.model_name}")
