import os
import shutil
import logging
import pickle
import json
from sklearn.externals import joblib
import tensorflow as tf


def export_feature_transform(fb_path, conf_path, feat_builder, conf):
    with open(fb_path, 'wb') as f:
        joblib.dump(feat_builder, f)
    with open(conf_path, 'wb') as f:
        joblib.dump(conf, f)


def export_model_pickle(path, model):
    with open(path, 'wb') as f:
        pickle.dump(model, f)


def export_model_joblib(path, model):
    with open(path, 'wb') as f:
        joblib.dump(model, f, compress=True)


def export_model_tf(model, model_name, version, simple_save=False):
    model_base_path = os.path.realpath("..")
    export_path = os.path.join(model_base_path, "serving", "models", model_name, version)
    if os.path.isdir(export_path):
        logging.warning("\tModel path \"%s\" already exists, removing..." % export_path)
        shutil.rmtree(export_path)
    if simple_save:
        print("simple_save is deprecated, it will be removed in tensorflow xxx...")
        tf.saved_model.simple_save(model.sess, export_path,
                                   inputs={'fi': model.feature_indices,
                                           'fv': model.feature_values},
                                   outputs={'y_prob': model.y_prob})
    else:
        builder = tf.saved_model.builder.SavedModelBuilder(export_path)
        input_fi = tf.saved_model.utils.build_tensor_info(model.feature_indices)
        input_fv = tf.saved_model.utils.build_tensor_info(model.feature_values)
        #    input_label = tf.saved_model.utils.build_tensor_info(self.labels)
        input_y = tf.saved_model.utils.build_tensor_info(model.y_prob)

        prediction_signature = (
            tf.saved_model.signature_def_utils.build_signature_def(
                inputs={'fi': input_fi,
                        'fv': input_fv},
                outputs={'y_prob': input_y},
                method_name=tf.saved_model.signature_constants.PREDICT_METHOD_NAME))

        builder.add_meta_graph_and_variables(
            model.sess, [tf.saved_model.tag_constants.SERVING],
            signature_def_map={'predict': prediction_signature}
            # tf.saved_model.signature_constants.DEFAULT_SERVING_SIGNATURE_DEF_KEY: prediction_signature},
            #    main_op=tf.tables_initializer(),
            #    strip_default_attrs=True
        )

        builder.save()
    logging.warning('\tDone exporting!')


def export_TFRecord():
    example = tf.parse_single_example()




