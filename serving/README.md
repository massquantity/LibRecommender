# LibRecommender Serving Guide

## Introduction

This guide mainly describes how to use [`flask`](<https://flask.palletsprojects.com/en/1.1.x/>) to serve a trained model in LibRecommender. From serving's perspective, currently there are three kinds of models in LibRecommender: 

+ KNN-based model
+ vector-based model 
+ tensorflow-based model. 

The following is the main serving workflow: 

1. Serialize trained model to disk.
2. Load model and save to redis.
3. Run flask server.
4. Make http request to the server and get recommendation.

Here we choose NOT to save the trained model directly to redis, since:  1) Even you save the model to redis in the first place, you'll end up with saving to disk anyway :)  2) We try to keep the requirements of the main `libreco` module as minimal as possible.

So during serving, one should start redis server first: 

```bash
$ redis-server
```



## Saving Format

In LibRecommender, the primary data serialization format is [`JSON`](<https://www.json.org/json-en.html>) rather than pickle, because pickle is relatively slow and it is declared in the official [pickle](<https://docs.python.org/3.6/library/pickle.html>) documentation that:

> Warning: The `pickle` module is not secure against erroneous or maliciously constructed data. Never unpickle data received from an untrusted or unauthenticated source.

Aside from JSON, models built upon tensorflow are saved using its own [`tf.saved_model`](<https://tensorflow.google.cn/versions/r1.15/api_docs/python/tf/saved_model>) API, which will basically be transformed to `protocol buffer` format.



## KNN-based model

KNN-based model refers to the classic `userCF` and `itemCF` algorithms, which leverages a similarity matrix to find similar users/items to recommend. Due to the large number of users/items, it is often impractical to store the whole similarity matrix, so here we may only save the most similar `K` neighbors for each user/item. 

Below is an example usage which saves 20 neighbors per item using itermCF. One should also specify model-saving `path` : 

```python
>>> from libreco.algorithms import UserCF, ItemCF
>>> from libreco.data import DatasetPure
>>> from libreco.utils import save_knn
>>> from serving.flask import sim2redis, user_consumed2redis

>>> train_data, data_info = DatasetPure.build_trainset(...)
>>> model = ItemCF(...)
>>> model.fit(...)  # train model
>>> path = "knn_model"  # specify model saving directory
>>> save_knn(path, model, train_data, k=20)  # save model
>>> sim2redis(path)	  # save similarity info to redis
>>> user_consumed2redis(path)  # save user_consumed to redis, in order to prevent from recommending items that the user has consumed
```

```bash
$ cd LibRecommender/serving/flask/knn
$ export FLASK_APP=knn_deploy.py
$ export FLASK_ENV=development  # optional debug mode, never use it in production 
$ flask run  # run flask server

# make requests
$ python knn_request.py --user 1 --k_neighbors 10 --n_rec 10  
$ curl -d '{"user": "1", "n_rec": 10, "k_neighbors": 10}' -X POST http://127.0.0.1:5000/item_cf/recommend
# get item id and score: {'recommend list for user (1)': [[3168, 9.421334058046341], [2538, 8.726857960224152], [505, 8.711400210857391], [530, 7.293927997350693], [1339, 7.1917658150196075], [4270, 7.149620413780212], [601, 7.130850255489349], [3808, 6.961166977882385], [2004, 6.635882019996643], [1300, 6.460416287183762]]}
```



## Vector-based model

Vector-based model relies on the dot product of two vectors to make recommendation, so we only need to save a bunch of vectors. This kind of model includes `SVD`, `SVD++`, `ALS`, `BPR` and `YouTubeMatch`.

In practice, to speed up serving, some ANN(Approximate Nearest Neighbors) libraries are often used to find similar vectors. Here in LibRecommender, we use [faiss](<https://github.com/facebookresearch/faiss>) to do such thing.

Below is an example usage which uses `ALS`. One should also specify model-saving `path` : 

```python
>>> from libreco.algorithms import ALS
>>> from libreco.data import DatasetPure
>>> from libreco.utils import save_vector
>>> from serving.flask import vector2redis, user_consumed2redis, save_faiss_index

>>> train_data, data_info = DatasetPure.build_trainset(...)
>>> model = ALS(...)
>>> model.fit(...)  # train model
>>> path = "vector_model"  # specify model saving directory
>>> save_vector(path, model, train_data)  # save model
>>> vector2redis(path)	  # save vector info to redis
>>> user_consumed2redis(path)  # save user_consumed to redis, in order to prevent from recommending items that the user has consumed
>>> save_faiss_index(path)   # save faiss index if you want to use faiss
```

```bash
$ cd LibRecommender/serving/flask/vector
$ export FLASK_APP=vector_deploy.py
$ export FLASK_ENV=development  # optional debug mode, never use it in production 
$ flask run  # run flask server

# make requests
$ python vector_request.py --user 1 --n_rec 10 --use_faiss false
$ curl -d '{"user": "1", "n_rec": 10, "use_faiss": true}' -X POST http://127.0.0.1:5000/vector/recommend
```



## Tensorflow-based model 

As stated above, tf-based model will typically be saved in `protocol buffer` format. These model mainly contains neural networks, including `Wide & Deep`,  `FM`,  `DeepFM`, `YouTubeRanking` , `AutoInt` , `DIN` . 

We use `tensorflow-serving` to serve tf-based models, and typically it is installed through Docker, see [official page](<https://github.com/tensorflow/serving>) for reference. After successfully starting the docker container, we post request  to the serving model inside the flask app and get the recommendation.

Below is an example usage which uses `DIN`. Since `DIN` makes use of user past interacted items, so we also need to save item sequence to redis. One should also specify model-saving `path` : 

```python
>>> from libreco.algorithms import DIN
>>> from libreco.data import DatasetFeat
>>> from libreco.utils import save_info, save_model_tf_serving
>>> from serving.flask import data_info2redis, user_consumed2redis, seq2redis

>>> train_data, data_info = DatasetFeat.build_trainset(...)
>>> model = DIN(...)
>>> model.fit(...)  # train model

>>> path = "tf_model"  # specify model saving directory
>>> save_info(path, model, train_data, data_info)  # save data_info
>>> save_model_tf_serving(path, model, "din")  # save tf model
>>> data_info2redis(path)	  # save feature info to redis
>>> user_consumed2redis(path)  # save user_consumed to redis, in order to prevent from recommending items that the user has consumed
>>> seq2redis(path)   # save item sequence to redis
```

```bash
$ sudo docker run --rm -t -p 8501:8501 --mount type=bind,source=$(pwd)/tf_model/din,target=/models/din -e MODEL_NAME=din tensorflow/serving   # start tensorflow-serving, make sure that model is in "tf_model/din" directory, or you can change to other directory

$ cd LibRecommender/serving/flask/tf
$ export FLASK_APP=tf_deploy.py
$ export FLASK_ENV=development  # optional debug mode, never use it in production 
$ flask run  # run flask server

# make requests
$ python tf_request.py --user 1 --n_rec 10 --algo din
$ curl -d '{"user": "1", "n_rec": 10, "algo": "din"}' -X POST http://127.0.0.1:5000/din/recommend
```




