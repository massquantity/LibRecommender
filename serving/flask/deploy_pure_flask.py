import os
import pickle
import pandas as pd
from sklearn.externals import joblib
from flask import Flask, jsonify, request
import sys
from pathlib import Path
sys.path.append(os.pardir)


app = Flask(__name__)


@app.route("/<algo>", methods=['POST'])
def api_call(algo):
    test_json = request.get_json(force=True)
    print(test_json)
    test_data = pd.DataFrame(test_json)
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


if __name__ == "__main__":
    app.run(debug=True, port=5000)  #  host="0.0.0.0"
