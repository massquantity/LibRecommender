Online Computing Serving
========================

The previous :doc:`python` does have one drawback: all the information are precomputed offline.
So recommendations can't be generated based on real-time features or user behavior sequences during serving,
which makes the service less flexible.

Starting from version ``1.2.0``, we have introduced `online computing serving` to address this limitation.
The procedure is quite similar, except that you can specify the user features and sequences in the request.

Please note that not all models support online computing:

+ Supporting user features: ``WideDeep``, ``FM``, ``DeepFM``, ``AutoInt``, ``TwoTower``.
+ Supporting sequences: ``RNN4Rec``, ``Caser``, ``WaveNet``.
+ Supporting both: ``YouTubeRetrieval``, ``YouTubeRanking``, ``DIN``.


Seq model
---------

Below is an example usage that uses ``RNN4Rec`` with dynamic behavior sequence:

.. code-block:: python3

    >>> from libreco.algorithms import RNN4Rec
    >>> from libreco.data import DatasetPure
    >>> from libserving.serialization import save_online, online2redis

    >>> train_data, data_info = DatasetPure.build_trainset(...)
    >>> model = RNN4Rec(...)
    >>> model.fit(...)  # train model
    >>> path = "online_model"  # specify model saving directory
    >>> save_online(path, model, version=1)  # save model in json format
    >>> online2redis(path, host="localhost", port=6379, db=0)  # load json from path and save model to redis

Online computing models all depend on `TensorFlow Serving <https://github.com/tensorflow/serving>`_ for inference, so we should start tf serving first.
For more details see :ref:`TensorFlow Model <tf-models>`:

.. code-block:: bash

    $ MODEL_NAME=rnn4rec
    $ MODEL_PATH=online_model
    $ sudo docker run --rm -it -p 8501:8501 --mount type=bind,source=$(pwd),target=$(pwd) -e MODEL_BASE_PATH=$(pwd)/${MODEL_PATH} -e MODEL_NAME=${MODEL_NAME} tensorflow/serving:2.8.2


When making request, the ``seq`` parameter should be a list:

.. code-block:: bash

    $ sanic sanic_serving.online_deploy:app --dev --access-logs -v --workers 1  # run sanic server

    # make requests
    $ python request.py --user 1 --n_rec 10 --algo online --seq '[1, 2, 3]'
    $ curl -d '{"user": 1, "n_rec": 10, "seq": [1, 2, 3]}' -X POST http://127.0.0.1:8000/online/recommend
    # {'Recommend result for user 1': ['1196', '480', '260', '2028', '1198', '1214', '780', '1387', '1291', '1197']}


Feat model
----------

Below is an example usage that uses ``TwoTower`` with dynamic user features:

.. code-block:: python3

    >>> from libreco.algorithms import TwoTower
    >>> from libreco.data import DatasetFeat
    >>> from libserving.serialization import save_online, online2redis

    >>> train_data, data_info = DatasetFeat.build_trainset(...)
    >>> model = TwoTower(...)
    >>> model.fit(...)
    >>> path = "online_model"
    >>> save_online(path, model, version=1)
    >>> online2redis(path, host="localhost", port=6379, db=0)

.. code-block:: bash

    $ MODEL_NAME=twotower
    $ MODEL_PATH=online_model
    $ sudo docker run --rm -it -p 8501:8501 --mount type=bind,source=$(pwd),target=$(pwd) -e MODEL_BASE_PATH=$(pwd)/${MODEL_PATH} -e MODEL_NAME=${MODEL_NAME} tensorflow/serving:2.8.2


When making request, the ``user_feats`` parameter should be a dict:

.. code-block:: bash

    $ sanic sanic_serving.online_deploy:app --dev --access-logs -v --workers 1  # run sanic server

    # make requests
    $ python request.py --user 1 --n_rec 10 --algo online --user_feats '{"sex": "F", "age": 10}'
    $ curl -d '{"user": 1, "n_rec": 10, "user_feats": {"sex": "F", "age": 10}}' -X POST http://127.0.0.1:8000/online/recommend
    # {'Recommend result for user 1': ['1196', '480', '260', '2028', '1198', '1214', '780', '1387', '1291', '1197']}


Feat & Seq model
----------------

Below is an example usage that uses ``DIN`` with dynamic user features and sequences:

.. code-block:: python3

    >>> from libreco.algorithms import DIN
    >>> from libreco.data import DatasetFeat
    >>> from libserving.serialization import save_online, online2redis

    >>> train_data, data_info = DatasetFeat.build_trainset(...)
    >>> model = DIN(...)
    >>> model.fit(...)
    >>> path = "online_model"
    >>> save_online(path, model, version=1)
    >>> online2redis(path, host="localhost", port=6379, db=0)

.. code-block:: bash

    $ MODEL_NAME=din
    $ MODEL_PATH=online_model
    $ sudo docker run --rm -it -p 8501:8501 --mount type=bind,source=$(pwd),target=$(pwd) -e MODEL_BASE_PATH=$(pwd)/${MODEL_PATH} -e MODEL_NAME=${MODEL_NAME} tensorflow/serving:2.8.2

.. code-block:: bash

    $ sanic sanic_serving.online_deploy:app --dev --access-logs -v --workers 1  # run sanic server

    # make requests
    $ python request.py --user 1 --n_rec 10 --algo online --user_feats '{"sex": "F", "age": 10}' --seq '[1, 2, 3]'
    $ curl -d '{"user": 1, "n_rec": 10, "user_feats": {"sex": "F", "age": 10}, "seq": [1, 2, 3]}' -X POST http://127.0.0.1:8000/online/recommend
    # {'Recommend result for user 1': ['1196', '480', '260', '2028', '1198', '1214', '780', '1387', '1291', '1197']}

----------

.. tip::

    If the recommendation results don't change after sending the request, there could be several possible reasons:

    1. You may have misspelled the feature name, e.g., "sex" -> "sax".
    2. It's possible that you specified a category that doesn't exist in the training data, e.g., "sex" -> "bisexual".
    3. Not all models support assigning features or sequences. For example,
       If you're using a ``pure`` model such as ``RNn4Rec`` or a model that doesn't utilize behavior sequence
       such as ``WideDeep``, passing a feature or a sequence won't have an impact on the results.
    4. The features you modified might not be considered important by the trained model.
       Consequently, the ranking scores didn't show significant changes.
