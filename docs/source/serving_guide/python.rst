.. _Serving Guide:

Python Serving Guide
====================

Introduction
------------

This guide mainly describes how to serve a trained model using the `libserving <https://github.com/massquantity/LibRecommender/tree/master/libserving>`_ module
in LibRecommender. Prior to LibRecommender version ``0.10.0``, `Flask <https://flask.palletsprojects.com/en/2.0.x/>`_
was used to construct the serving web server. However, to take full advantage of the asynchronous
feature and modern ``async/await`` syntax in Python, we've decided to switch to `Sanic <https://github.com/sanic-org/sanic>`_ framework.
Unlike flask, a sanic server can run in production directly, and the typical command is like this:

.. code-block:: bash

    $ sanic server:app --host=127.0.0.1 --port=8000 --dev --access-logs -v --workers 1  # develop mode
    $ sanic server:app --no-access-logs --workers 10  # production mode

Refer to `Running Sanic <https://sanic.dev/en/guide/deployment/running.html>`_ for more details.

.. admonition:: Rust
   :class: rust

   Beyond Python, one can also use Rust to serve a model. See :doc:`rust`.

-----------

From model serving's perspective, currently there are three kinds of models in LibRecommender:

+ knn-based model
+ embed-based model
+ tensorflow-based model.

As for models trained with PyTorch, they all belong to embed-based model.

The following is the main serving workflow:

1. Serialize the trained model to disk.
2. Load model and save to redis.
3. Run the sanic server.
4. Make http request to the server and gain recommendation.

Here we choose NOT to save the trained model directly to redis, since:
1) Even you save the model to redis in the first place, you'll end up with saving to disk anyway :)
2) We try to keep the requirements of the main ``libreco`` module as minimal as possible.

So during serving, one should start redis server first:

.. code-block:: bash

    $ redis-server


.. Error::

    Note that sometimes using redis in model serving can be error-prone:

    For example, you served a ``DeepFM`` model at first, and later on you decided to use
    another ``pure`` model, say ``NCF``.  Since ``DeepFM`` is a ``feat`` model,
    some feature information may have been saved into redis. If you forget to remove
    these feature information before using ``NCF``, the server may mistakenly load it
    and eventually causing an error.

.. Attention::

    In this guide we assume the following codes are all executed in ``LibRecommender/libserving`` folder,
    so one needs to clone the repository first:

    .. code-block:: bash

        $ git clone https://github.com/massquantity/LibRecommender.git
        $ cd LibRecommender/libserving




Note about Dependencies
-----------------------

The serving related dependencies are listed in `main README <https://github.com/massquantity/LibRecommender#optional-dependencies-for-libserving>`_.

+ `redis-py <https://github.com/redis/redis-py>`_ introduced async support since 4.2.0.

+ Pydantic has introduced breaking changes in `V2 <https://docs.pydantic.dev/latest/migration/>`_.
  Consider upgrading to ``pydantic >= 2.0`` if you encounter validation errors.

+ According to the `official doc <https://github.com/facebookresearch/faiss/blob/main/INSTALL.md>`__, faiss can't be installed from pip directly.
  But someone has built wheel packages, refer to `faiss-wheels <https://github.com/kyamagu/faiss-wheels>`_.
  So now the pip option is available for faiss.

+ We use `TensorFlow Serving <https://github.com/tensorflow/serving>`_ to serve
  tensorflow-based models, and typically it is installed through Docker.
  However, The latest TensorFlow Serving might not work in some cases,
  and we have encountered similar error described in this
  `issue <https://github.com/tensorflow/serving/issues/2048>`_ on TensorFlow Serving 2.9 and 2.10.
  So for now the workable version is 2.8.2, and one should pull docker image like this:

.. code-block:: bash

    $ sudo docker pull tensorflow/serving:2.8.2


Saving Format
-------------

In ``libserving``, the primary data serialization format is `JSON <https://www.json.org/json-en.html>`_.

Aside from JSON, models built upon TensorFlow are saved using its own
`tf.saved_model <https://www.tensorflow.org/versions/r1.15/api_docs/python/tf/saved_model>`_ API.
The ``SavedModel`` format provides a language-neutral format to save machine-learning models.


KNN-based Model
---------------

KNN-based models refer to the classic ``UserCF`` and ``ItemCF`` algorithms, which leverage
a similarity matrix to find similar users/items for recommendation.
Due to the large number of users/items, it is often impractical to store the whole
similarity matrix, so here we may only save the most similar k neighbors for each user/item.

Below is an example usage which saves 10 neighbors per item using ItermCF.
One should also specify model-saving ``path`` :

.. code-block:: python3

    >>> from libreco.algorithms import ItemCF
    >>> from libreco.data import DatasetPure
    >>> from libserving.serialization import knn2redis, save_knn

    >>> train_data, data_info = DatasetPure.build_trainset(...)
    >>> model = ItemCF(...)
    >>> model.fit(...)  # train model
    >>> path = "knn_model"  # specify model saving directory
    >>> save_knn(path, model, k=10)  # save model in json format
    >>> knn2redis(path, host="localhost", port=6379, db=0)  # load json from path and save model to redis

.. code-block:: bash

    $ sanic sanic_serving.knn_deploy:app --dev --access-logs -v --workers 1  # run sanic server

    # make requests
    $ python request.py --user 1 --n_rec 10 --algo knn
    $ curl -d '{"user": 1, "n_rec": 10}' -X POST http://127.0.0.1:8000/knn/recommend
    # {'Recommend result for user 1': ['480', '589', '2571', '260', '2028', '1198', '1387', '1214', '1291', '1197']}



Embed-based Model
-----------------

Embed-based models perform similarity searching on embeddings to make recommendation,
so we only need to save a bunch of embeddings. This kind of model includes
``SVD``, ``SVD++``, ``ALS``, ``BPR``, ``YouTubeRetrieval``, ``Item2Vec``, ``DeepWalk``,
``RNN4Rec``, ``Caser``, ``WaveNet``, ``NGCF``, ``LightGCN``, ``GraphSage``, ``PinSage``,
``TwoTower``.

In practice, to speed up serving, some ANN(Approximate Nearest Neighbors) libraries
are often used to find similar embeddings. Here in ``libserving``, we use
`faiss <https://github.com/facebookresearch/faiss>`_ to do such thing.

Below is an example usage which uses ``ALS``. One should also specify model-saving ``path``:

.. code-block:: python3

    >>> from libreco.algorithms import ALS
    >>> from libreco.data import DatasetPure
    >>> from libserving.serialization import embed2redis, save_embed

    >>> train_data, data_info = DatasetPure.build_trainset(...)
    >>> model = ALS(...)
    >>> model.fit(...)  # train model
    >>> path = "embed_model"  # specify model saving directory
    >>> save_embed(path, model)  # save model in json format
    >>> embed2redis(path, host="localhost", port=6379, db=0)  # load json from path and save model to redis

The following code will train faiss index on model's item embeddings and save to disk as file name
``faiss_index.bin``. The saved index will be loaded in sanic server.

.. code-block:: python3

    >>> from libserving.serialization import save_faiss_index
    >>> save_faiss_index(path, model)

.. code-block:: bash

    $ sanic sanic_serving.embed_deploy:app --dev --access-logs -v --workers 1  # run sanic server

    # make requests
    $ python request.py --user 1 --n_rec 10 --algo embed
    $ curl -d '{"user": 1, "n_rec": 10}' -X POST http://127.0.0.1:8000/embed/recommend
    # {'Recommend result for user 1': ['593', '1270', '318', '2858', '1196', '2571', '1617', '260', '1200', '457']}

.. _tf-models:

TensorFlow-based Model
----------------------

As stated above, tensorflow-based model will typically be saved in ``SavedModel`` format.
These model mainly contains neural networks, including ``NCF``, ``WideDeep``,  ``FM``,
``DeepFM``, ``YouTubeRanking`` , ``AutoInt`` , ``DIN``.

We assume TensorFlow Serving has already been installed through Docker.
After successfully starting the docker container, we can post request to the
serving model inside the sanic server and get the recommendation.

Below is an example usage which uses ``DIN``. One should also specify model-saving ``path``:

.. code-block:: python3

    >>> from libreco.algorithms import DIN
    >>> from libreco.data import DatasetFeat
    >>> from libserving.serialization import save_tf, tf2redis

    >>> train_data, data_info = DatasetFeat.build_trainset(...)
    >>> model = DIN(...)
    >>> model.fit(...)  # train model
    >>> path = "tf_model"  # specify model saving directory
    >>> save_tf(path, model, version=1)  # save model in json format
    >>> tf2redis(path, host="localhost", port=6379, db=0)  # load json from path and save model to redis

The directory of ``SavedModel`` format for a ``DIN`` model has the following structure and note
that 1 is the version number:

::

    din/
        1/
            variables/
                variables.data-?????-of-?????
                variables.index
            saved_model.pb


We can inspect the saved ``DIN`` model by using ``SavedModel CLI`` described in
`official doc <https://www.tensorflow.org/guide/saved_model#details_of_the_savedmodel_command_line_interface>`__.
By default, it is bundled with TensorFlow. The following command will output:

.. code-block:: bash

    $ saved_model_cli show --dir tf_model/din/1 --all

.. code-block:: bash

    MetaGraphDef with tag-set: 'serve' contains the following SignatureDefs:

    signature_def['predict']:
      The given SavedModel SignatureDef contains the following input(s):
        inputs['dense_values'] tensor_info:
            dtype: DT_FLOAT
            shape: (-1, 1)
            name: Placeholder_6:0
        inputs['item_indices'] tensor_info:
            dtype: DT_INT32
            shape: (-1)
            name: Placeholder_1:0
        inputs['sparse_indices'] tensor_info:
            dtype: DT_INT32
            shape: (-1, 5)
            name: Placeholder_5:0
        inputs['user_indices'] tensor_info:
            dtype: DT_INT32
            shape: (-1)
            name: Placeholder:0
        inputs['user_interacted_len'] tensor_info:
            dtype: DT_FLOAT
            shape: (-1)
            name: Placeholder_3:0
        inputs['user_interacted_seq'] tensor_info:
            dtype: DT_INT32
            shape: (-1, 10)
            name: Placeholder_2:0
      The given SavedModel SignatureDef contains the following output(s):
        outputs['logits'] tensor_info:
            dtype: DT_FLOAT
            shape: (-1)
            name: Reshape_4:0
      Method name is: tensorflow/serving/predict


The above result shows this ``DIN`` model needs 6 inputs, i.e. ``user_indices``, ``item_indices``,
``sparse_indices``, ``dense_values``, ``user_interacted_seq``, ``user_interacted_len``.
But this only applies to ``DIN`` and other models may have different inputs.

+ For ``NCF`` model, only ``user_indices`` and ``item_indices`` are needed since it's a
  collaborative-filtering algorithm.

+ For ``WideDeep``,  ``FM``,  ``DeepFM``, ``AutoInt``, since they don't use behavior sequence
  information, 4 inputs are needed: ``user_indices``, ``item_indices``, ``sparse_indices``, ``dense_values``.

+ Finally, ``YouTubeRanking`` has same inputs as ``DIN``. They both use behavior sequence information.

However, these are just general cases. Suppose your data doesn't have any sparse feature,
then it would be a mistake to feed the ``sparse_indices`` input, so these matters should
be taken into account. This is exactly where a library fits in, and LibRecommender can
dynamically handle these different feature situations. So as a library user, all you
need to do is specifying the correct model path.

|

Using ``SavedModel CLI``, we can even pass some inputs to the model and get outputs
(note the inputs num should match the model requirement):

.. code-block:: bash

    $ inputs="user_indices=np.int32([2,3]);item_indices=np.int32([2,3]);sparse_indices=np.int32([[1,1,1,1,1],[1,1,1,1,1]]);dense_values=np.float32([[1],[2]]);user_interacted_len=np.float32([2,3]);user_interacted_seq=np.int32([[1,2,3,4,5,6,7,8,9,10],[1,2,3,4,5,6,7,8,9,10]])"

    $ saved_model_cli run --dir tf_model/din/1 --tag_set serve --signature_def predict --input_exprs $inputs

.. code-block:: bash

    Result for output key logits:
    [-0.51893234 -0.569685  ]


|

Now let's start TensorFlow Serving service through docker. Note that the ``MODEL_NAME`` should be
lowercase of the model class name. For instance, ``DIN`` -> ``din``, ``YouTubeRanking`` -> ``youtuberanking``, ``WideDeep`` -> ``widedeep``.

.. code-block:: bash

    $ MODEL_NAME=din
    $ MODEL_PATH=tf_model
    $ sudo docker run --rm -t -p 8501:8501 --mount type=bind,source=$(pwd),target=$(pwd) -e MODEL_BASE_PATH=$(pwd)/${MODEL_PATH} -e MODEL_NAME=${MODEL_NAME} tensorflow/serving:2.8.2

Get model status from TensorFlow Serving service using RESTful API:

.. code-block:: bash

    $ curl http://localhost:8501/v1/models/din


.. code-block:: bash

    {
     "model_version_status": [
      {
       "version": "1",
       "state": "AVAILABLE",
       "status": {
        "error_code": "OK",
        "error_message": ""
       }
      }
     ]
    }


Make predictions for two samples from TensorFlow Serving service:

.. code-block:: bash

    $ curl -d '{"signature_name": "predict", "inputs": {"user_indices": [1, 2], "item_indices": [3208, 2], "sparse_indices": [[1, 19, 32, 59, 71], [1, 19, 32, 59, 71]], "dense_values": [22.0, 56.0], "user_interacted_seq": [[996, 1764, 2083, 520, 2759, 334, 304, 1110, 2013, 1415],[996, 1764, 2083, 520, 2759, 334, 304, 1110, 2013, 1415]], "user_interacted_len": [3, 10]}}' -X POST http://localhost:8501/v1/models/din:predict


.. code-block:: bash

    {
        "outputs": [
            -0.65978992,
            -0.759211063
        ]
    }


Now we can start the corresponding sanic server. According to the `official doc <https://www.tensorflow.org/tfx/serving/api_rest#request_format_2>`__,
the input tensors can use either row format or column format. In `tf_deploy.py <https://github.com/massquantity/LibRecommender/tree/master/libserving/sanic_serving/tf_deploy.py>`_
we use column format since it's more compact.

.. code-block:: bash

    $ sanic sanic_serving.tf_deploy:app --dev --access-logs -v --workers 1  # run sanic server

    # make requests
    $ python request.py --user 1 --n_rec 10 --algo tf
    $ curl -d '{"user": 1, "n_rec": 10}' -X POST http://127.0.0.1:8000/tf/recommend
    # {'Recommend result for user 1': ['1196', '480', '260', '2028', '1198', '1214', '780', '1387', '1291', '1197']}
