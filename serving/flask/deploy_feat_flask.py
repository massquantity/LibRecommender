import os
import pickle
import time
import json
import itertools
from collections import defaultdict, OrderedDict
import numpy as np
import pandas as pd
import requests
from sklearn.externals import joblib
from flask import Flask, jsonify, request
import sys
from pathlib import Path, PurePath
sys.path.append(os.pardir)

path = str(Path.joinpath(PurePath("."), "models", "others", "fm_dataset.jb"))
print(path)

with open(str(Path.joinpath(Path("."), "models", "others", "fm_dataset.jb")), 'rb') as f:
    dataset = joblib.load(f)
with open(str(Path.joinpath(PurePath("."), "models", "others", "fm_unique_items.jb")), 'rb') as f:
    items_unique = joblib.load(f)
with open(os.path.join(os.curdir, "models/others/feature_builder.jb"), 'rb') as f:
    feat_builder = joblib.load(f)
with open(os.path.join(os.curdir, "models/others/fm_conf.jb"), 'rb') as f:
    conf = joblib.load(f)


app = Flask(__name__)


@app.route("/<algo>/predict", methods=['POST'])
def api_call(algo):
    test_json = request.get_json(force=True)
    print(test_json)
    test_data = pd.DataFrame(test_json)
    orig_cols = ["user", "item", "sex", "age", "occupation", "title", "genre1", "genre2", "genre3"]
    test_data = test_data.reindex(columns=orig_cols)
    feat_indices, feat_values = feature_transform(test_data)

    samples = []
    for fi, fv in zip(feat_indices, feat_values):
        samples.append({"fi": fi.tolist(), "fv": fv.tolist()})

    data = {"signature_name": "predict", "instances": samples}
    if algo.lower() == 'fm':
        model = algo
    else:
        raise ValueError("algorithm %s is not allowed" % algo)
    response = requests.post("http://localhost:8501/v1/models/%s:predict" % model, data=json.dumps(data))
    return response.text


def feature_transform(data):
    """
    transform data into model input format
    :param data: original data
    :param feat_builder_path: saved feature_builder object path
    :param conf_path: saved configure file path
    :return: feature indices, feature values
    """
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


def get_recommend_indices_and_values(data, user, items_unique):
    user_col = data.train_feat_indices.shape[1] - 2
    item_col = data.train_feat_indices.shape[1] - 1

    user_repr = user + data.user_offset
    user_cols = data.user_feature_cols + [user_col]
    user_features = data.train_feat_indices[:, user_cols]
    user_unique = user_features[user_features[:, -1] == user_repr][0]
    users = np.tile(user_unique, (data.n_items, 1))

    #   np.unique is sorted based on the first column, so put item column first
    item_cols = [item_col] + data.item_feature_cols
    orig_cols = user_cols + item_cols
    col_reindex = np.array(range(len(orig_cols)))[np.argsort(orig_cols)]

    assert users.shape[0] == items_unique.shape[0], "user shape must equal to num of candidate items"
    concat_indices = np.concatenate([users, items_unique], axis=-1)[:, col_reindex]

    #   construct feature values, mainly fill numerical columns
    feat_values = np.ones(shape=(data.n_items, concat_indices.shape[1]))
    if data.numerical_col is not None:
        numerical_dict = OrderedDict()
        for col in range(len(data.numerical_col)):
            if col in data.user_feature_cols:
                user_indices = np.where(data.train_feat_indices[:, user_col] == user_repr)[0]
                numerical_values = data.train_feat_values[user_indices, col][0]
                numerical_dict[col] = numerical_values
            elif col in data.item_feature_cols:
                # order according to item indices
                numerical_map = OrderedDict(
                                    sorted(
                                        zip(data.train_feat_indices[:, -1],
                                            data.train_feat_values[:, col]), key=lambda x: x[0]))
                numerical_dict[col] = [v for v in numerical_map.values()]

        for k, v in numerical_dict.items():
            feat_values[:, k] = np.array(v)

    return concat_indices, feat_values


@app.route("/<algo>/recommend", methods=['POST'])
def recommend(algo):
    test_json = request.get_json(force=True)
#    print(type(test_json))
#    print(test_json)
    test_data = pd.DataFrame(test_json)
    user = test_data["user"][0]
    n_rec = test_data["n_rec"][0]
    consumed = dataset.train_user[user]
    count = n_rec + len(consumed)
    feat_indices, feat_values = get_recommend_indices_and_values(dataset, user, items_unique)

    samples = []
    for fi, fv in zip(feat_indices, feat_values):
        samples.append({"fi": fi.tolist(), "fv": fv.tolist()})
    data = {"signature_name": "predict", "instances": samples}
    if algo.lower() == 'fm':
        model = algo
    else:
        raise ValueError("algorithm %s is not allowed" % algo)
    response = requests.post("http://localhost:8501/v1/models/%s:predict" % model, data=json.dumps(data))
#    response = json.loads(response.text)
    response = response.json()
    preds = pd.DataFrame(response).values.ravel()
    ids = np.argpartition(preds, -count)[-count:]
    rank = sorted(zip(ids, preds[ids]), key=lambda x: -x[1])
    rank_list = itertools.islice((rec for rec in rank if rec[0] not in consumed), int(n_rec))
    rank_list = [(int(r[0]), r[1]) for r in rank_list]
    rec = {"recommend list for user %s" % str(user): rank_list}
    return json.dumps(rec, indent=4)


@app.errorhandler(400)
def bad_request(error=None):
    message = {
        'status': 400,
        'message': 'Bad Request: ' + request.url + " --> Please provide more data information...",
    }
    resp = jsonify(message)
    resp.status_code = 400
    return resp


if __name__ == "__main__":
    app.run(debug=True, port=5000)  #  host="0.0.0.0"
    # export FLASK_APP=deploy_feat_flask.py | export FLASK_ENV=development | flask run

