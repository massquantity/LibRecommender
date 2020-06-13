import abc


class Base(abc.ABC):
    """Base class for all recommendation models.

    Parameters
    ----------
    data_info : `DataInfo` object
        Object that contains some useful information for training and predicting.
    lower_upper_bound: list or tuple, optional
        Lower and upper score bound for rating task.
    """

    def __init__(self, data_info, task, lower_upper_bound=None):
        self.task = task
        if task == "rating":
            if lower_upper_bound is not None:
                assert isinstance(lower_upper_bound, (list, tuple)), (
                    "must contain both lower and upper bound if provided")
                self.lower_bound = lower_upper_bound[0]
                self.upper_bound = lower_upper_bound[1]
            else:
                self.lower_bound, self.upper_bound = data_info.min_max_rating
            print(f"lower bound: {self.lower_bound}, upper bound: {self.upper_bound}")
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

