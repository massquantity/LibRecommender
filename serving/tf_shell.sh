#!/bin/sh

sudo docker run -p 8501:8501 --mount type=bind,source=/home/massquantity/Workspace/LibRecommender/serving/models/FM,target=/models/FM -e MODEL_NAME=FM -t tensorflow/serving
export FLASK_APP=deploy_feat_flask.py
export FLASK_ENV=development
flask run