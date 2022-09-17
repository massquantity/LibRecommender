import pytest

from libreco.bases import Base


class NCF(Base):
    def __init__(self, task, data_info, lower_upper_bound=None):
        super().__init__(task, data_info, lower_upper_bound)

    def fit(self, train_data, **kwargs):
        raise NotImplementedError

    def predict(self, user, item, **kwargs):
        raise NotImplementedError

    def recommend_user(self, user, n_rec, **kwargs):
        raise NotImplementedError

    def save(self, path, model_name, **kwargs):
        raise NotImplementedError

    @classmethod
    def load(cls, path, model_name, data_info, **kwargs):
        raise NotImplementedError


def test_base(prepare_pure_data):
    _, train_data, _, data_info = prepare_pure_data
    with pytest.raises(ValueError):
        _ = NCF(task="unknown", data_info=data_info)
    with pytest.raises(AssertionError):
        _ = NCF(task="rating", data_info=data_info, lower_upper_bound=1)

    model = NCF(task="rating", data_info=data_info, lower_upper_bound=[1, 5])
    with pytest.raises(NotImplementedError):
        model.fit(train_data)
    with pytest.raises(NotImplementedError):
        model.predict(1, 2)
    with pytest.raises(NotImplementedError):
        model.recommend_user(1, 7)
    with pytest.raises(NotImplementedError):
        model.save("path", "model_name")
    with pytest.raises(NotImplementedError):
        NCF.load("path", "model_name", data_info)
