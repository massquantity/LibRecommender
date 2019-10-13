import os
import pickle
import json
from collections import defaultdict
import numpy as np
import pandas as pd
import requests
from sklearn.externals import joblib
from flask import Flask, jsonify, request
from flask_restful import reqparse, abort, Api, Resource
import sys
from pathlib import Path
sys.path.append(os.pardir)


app = Flask(__name__)
api = Api(app)

algo_name = ['fm', 'FM']


class predict(Resource):
    def get(self):
        return algo_name

    def post(self, algo):
        test_json = request.get_json(force=True)
        print(test_json)
        test_data = pd.DataFrame(test_json)
        orig_cols = ["user", "item", "sex", "age", "occupation", "title", "genre1", "genre2", "genre3"]
        test_data = test_data.reindex(columns=orig_cols)
        feature_builder_path = os.path.join(os.curdir, "models/others/feature_builder.jb")
        conf_path = os.path.join(os.curdir, "models/others/conf.jb")
        feat_indices, feat_values = feature_transform(test_data, feature_builder_path, conf_path)

        samples = []
        for fi, fv in zip(feat_indices, feat_values):
            samples.append({"fi": fi.tolist(), "fv": fv.tolist()})

        data = {"signature_name": "predict", "instances": samples}
        response = requests.post("http://localhost:8501/v1/models/%s:predict" % algo, data=json.dumps(data))
        return response.text


api.add_resource(predict, '/algo')


def feature_transform(data, fb_path, conf_path):
    """
    transform data into model input format
    :param data: original data
    :param feat_builder_path: saved feature_builder object path
    :param conf_path: saved configure file path
    :return: feature indices, feature values
    """
    with open(fb_path, 'rb') as f:
        feat_builder = joblib.load(f)
    with open(conf_path, 'rb') as f:
        conf = joblib.load(f)

    categorical_features = dict()
    numerical_features = dict()
    merge_cat_features = defaultdict(list)
    merge_list = dict()
    data_size = len(data)
    user_indices = data['user'].values
    item_indices = data['item'].values

    if conf['categorical_col'] is not None:
        for cat_feat in conf['categorical_col']:
            categorical_features[cat_feat] = data.iloc[:, cat_feat - 1]  # doesn't include label col

    if conf['numerical_col'] is not None:
        for num_feat in conf['numerical_col']:
            numerical_features[num_feat] = data.iloc[:, num_feat - 1]

    if conf['merged_categorical_col'] is not None:
        for merge_feat in conf['merged_categorical_col']:
            for mft in merge_feat:
                merge_list[mft] = data.iloc[:, mft - 1]
        for merge_feat in conf['merged_categorical_col']:
            merge_col_index = merge_feat[0]
            for mft in merge_feat:
                merge_cat_features[merge_col_index].extend(merge_list[mft])

    feat_indices, feat_values = feat_builder.transform(categorical_features,
                                                       numerical_features,
                                                       merge_cat_features,
                                                       data_size,
                                                       user_indices,
                                                       item_indices)

    return feat_indices, feat_values


if __name__ == "__main__":
    app.run(debug=True, port=5000)  #  host="0.0.0.0"
    # export FLASK_APP=deploy_feat_flask.py | export FLASK_ENV=development | flask run

