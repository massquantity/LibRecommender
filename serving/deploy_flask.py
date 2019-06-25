import os
import pickle
import pandas as pd
from sklearn.externals import joblib
from flask import Flask, jsonify, request
import sys
from pathlib import Path
sys.path.append(os.pardir)
import libreco

app = Flask(__name__)


@app.route("/predict", methods=['POST'])
def api_call():
    try:
        test_json = request.get_json(force=True)
        print(test_json)
        test_data = pd.DataFrame(test_json)
        for col in test_data.columns:
            if col in ['u', 'user']:
                u = test_data[col][0]
            elif col in ['i', 'item']:
                i = test_data[col][0]

    except Exception as e:
        raise e

    model_path = os.path.join(os.getcwd(), "models", "user_knn.jb")
    with open(model_path, 'rb') as f:
        model = joblib.load(f)
    u = str(u)
    i = str(i)
    pred = model.predict(u, i)
    print(u, i, pred)
    response = jsonify({'user': u, 'item': i, 'pred': pred})  # user=u, item=i,
    return response

'''
if __name__ == "__main__":
    app.run(debug=True)
'''