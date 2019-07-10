from abc import ABCMeta, abstractmethod


class Base(object):
    __metaclass__ = ABCMeta

    @abstractmethod
    def build_model(self):
        pass

    @abstractmethod
    def fit(self):
        raise NotImplementedError

    @abstractmethod
    def predict(self):
        raise NotImplementedError
