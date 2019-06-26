import os
import pickle
from collections import defaultdict
import numpy as np
import pandas as pd
from sklearn.externals import joblib
from flask import Flask, jsonify, request
import sys
from pathlib import Path
sys.path.append(os.pardir)
import libreco

app = Flask(__name__)


@app.route("/<algo>", methods=['POST'])
def api_call(algo):
    test_json = request.get_json(force=True)
    print(test_json)
    test_data = pd.DataFrame(test_json)
    orig_cols = ["user", "item", "sex", "age", "occupation", "title", "genre1", "genre2", "genre3"]
    test_data = test_data.reindex(columns=orig_cols)


    for col in test_data.columns:
        if col in ['u', 'user']:
            u = test_data[col][0]
        elif col in ['i', 'item']:
            i = test_data[col][0]
    try:
        n_rec = test_data["n_rec"][0]
        k = test_data["k"][0]
    except KeyError:
        n_rec = k = None

    if algo in ["user_knn", "userKNN"]:
        algo = "user_knn"
    model_path = os.path.join(os.getcwd(), "models", "%s.jb" % algo)
    with open(model_path, 'rb') as f:
        model = joblib.load(f)

    if n_rec and k:
        reco_list = model.topN(u, k, n_rec)
        response = jsonify({'user': str(u), 'recommend list': str(reco_list)})
    elif u and i:
        pred = model.predict(u, i)
        response = jsonify({'user': str(u), 'item': str(i), 'pred': str(pred)})
    else:
        return bad_request()
    return response


@app.errorhandler(400)
def bad_request(error=None):
    message = {
        'status': 400,
        'message': 'Bad Request: ' + request.url + " --> Please provide more data information...",
    }
    resp = jsonify(message)
    resp.status_code = 400
    return resp


def feature_transform(data, feat_builder_path, conf_path):
    with open(feat_builder_path, 'rb') as f:
        feat_builder = joblib.load(f)
    with open(conf_path, 'rb') as f:
        conf = joblib.load(f)

    categorical_features = dict()
    numerical_features = dict()
    merge_cat_features = defaultdict(list)
    merge_list = list()
    data_size = len(data)
    user_indices = data['user'].values
    item_indices = data['item'].values

    for cat_feat in conf['categorical_col']:
        categorical_features[cat_feat] = data.iloc[:, cat_feat - 1]  # doesn't include label col
    for num_feat in conf['numerical_col']:
        numerical_features[num_feat] = data.iloc[:, num_feat - 1]
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
