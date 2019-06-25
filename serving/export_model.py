import pickle
import tensorflow as tf


def export_model():
    pickle.dump(...)


def export_model_tf():
    export_path = "."
    builder = tf.saved_model.builder.SavedModelBuilder(export_path)