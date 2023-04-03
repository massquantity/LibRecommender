import tensorflow as tf

from libreco.bases import TfBase
from libreco.data import DataInfo
from tests.utils_data import SAVE_PATH


def save_load_model(cls, model, data_info):
    model_name = cls.__name__.lower() + "_model"
    data_info.save(path=SAVE_PATH, model_name=model_name)
    model.save(SAVE_PATH, model_name, manual=True, inference_only=True)

    if issubclass(cls, TfBase) or hasattr(model, "sess"):
        tf.compat.v1.reset_default_graph()
    loaded_data_info = DataInfo.load(path=SAVE_PATH, model_name=model_name)
    loaded_model = cls.load(SAVE_PATH, model_name, loaded_data_info, manual=True)
    return loaded_model, loaded_data_info
