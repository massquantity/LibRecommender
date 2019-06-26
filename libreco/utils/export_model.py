import pickle
from sklearn.externals import joblib
import tensorflow as tf


def export_model_pickle(path, model):
    with open(path, 'wb') as f:
        pickle.dump(model, f)


def export_model_joblib(path, model):
    with open(path, 'wb') as f:
        joblib.dump(model, f, compress=True)


def export_model_tf():
    export_path = "."
    builder = tf.saved_model.builder.SavedModelBuilder(export_path)











