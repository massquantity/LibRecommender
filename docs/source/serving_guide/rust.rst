Rust Serving Guide
==================

Introduction
------------

This guide mainly describes how to serve a trained model in LibRecommender with `Rust <https://www.rust-lang.org/>`_.
A Rust web server is typically much faster than its Python counterpart. In the `libserving <https://github.com/massquantity/LibRecommender/tree/master/libserving>`_ module,
we use `Actix <https://github.com/actix/actix-web>`_, one of the fastest web frameworks in the world.
The overall serving procedure in this guide resembles :doc:`python`,
so you'll need a Redis too. If you are already familiar with Rust, then follow the steps below.
Otherwise, you can also use `Docker Compose`_.

Users need to provide three environment variables before starting the server:

+ ``PORT`` assigned to the server. In Actix 8080 is commonly used.
+ ``MODE_TYPE`` specifies model category used in server. According to Python Serving Guide, the value should be one of ``knn``, ``embed``, ``tf``.
+ ``WORKERS`` specifies number of workers used in server.

KNN-based Model
---------------

.. code-block:: bash

    $ cd LibRecommender/libserving


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

    $ cd actix_serving
    $ export PORT=8080 MODEL_TYPE=knn WORKERS=8  # assign environment variables

Use either develop or production mode:

.. code-block:: bash

    $ cargo run --package actix_serving --bin actix_serving  # develop mode

.. code-block:: bash

    $ cargo build --release  # production mode
    $ ./target/release/actix_serving

Note that the ``"user"`` parameter in request should be string and the route is ``knn/recommend`` :

.. code-block:: bash

    $ curl -d '{"user": "1", "n_rec": 10}' -H "Content-Type: application/json" -X POST http://0.0.0.0:8080/knn/recommend
    # {"rec_list":["1806","1687","3799","807","1303","1860","3847","3616","1696","1859"]}


Embed-based Model
-----------------

Similar to Python Serving Guide,
we use faiss to find similar embeddings. However, faiss can't be installed directly as in Python,
so we use `Rust bindings <https://github.com/Enet4/faiss-rs>`_ to Faiss. According to the
`instructions <https://github.com/Enet4/faiss-rs#installing-as-a-dependency>`_,
one should fork the `c_api_head branch <https://github.com/Enet4/faiss/tree/c_api_head>`_
and build faiss from source manually before including the crate. We should warn you first,
this process can be pretty frustrating, and if you get stuck, you can view the
`Dockerfile-rs <https://github.com/massquantity/LibRecommender/blob/master/libserving/Dockerfile-rs>`_
file to get some hints:). Or you can just save all these troubles and use :ref:`Docker Compose <Docker Compose>`.

After successfully installing Rust faiss, i.e. copy the ``c_api/libfaiss_c.so`` and ``faiss/libfaiss.so``
to ``LD_LIBRARY_PATH``, the rest is straightforward:

.. code-block:: bash

    $ cd LibRecommender/libserving


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

.. code-block:: python3

    >>> from libserving.serialization import save_faiss_index
    >>> save_faiss_index(path, model)  # save faiss index to disk, note this uses python faiss, not rust faiss


.. code-block:: bash

    $ cd actix_serving
    $ export PORT=8080 MODEL_TYPE=embed WORKERS=8  # assign environment variables

Use either develop or production mode:

.. code-block:: bash

    $ cargo run  # develop mode, this uses rust faiss


.. code-block:: bash

    $ cargo build --release  # production mode
    $ ./target/release/actix_serving

Note that the ``"user"`` parameter in request should be string and the route is ``embed/recommend`` :

.. code-block:: bash

    $ curl -d '{"user": "1", "n_rec": 10}' -H "Content-Type: application/json" -X POST http://0.0.0.0:8080/embed/recommend
    # {"rec_list":["858","260","2355","1287","527","2371","1220","377","1968","3362"]}


TensorFlow-based Model
----------------------

.. code-block:: bash

    $ cd LibRecommender/libserving


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

.. code-block:: bash

    $ MODEL_NAME=din # variables for tensorflow serving
    $ MODEL_PATH=tf_model
    $ sudo docker run --rm -t -p 8501:8501 --mount type=bind,source=$(pwd),target=$(pwd) -e MODEL_BASE_PATH=$(pwd)/${MODEL_PATH} -e MODEL_NAME=${MODEL_NAME} tensorflow/serving:2.8.2  # start tensorflow serving

.. code-block:: bash

    $ cd actix_serving
    $ export PORT=8080 MODEL_TYPE=tf WORKERS=8 # assign environment variables

Use either develop or production mode:

.. code-block:: bash

    $ cargo run  # develop mode, this uses rust faiss

.. code-block:: bash

    $ cargo build --release  # production mode
    $ ./target/release/actix_serving

.. _Docker Compose:

Serving with Docker Compose
---------------------------

In `docker-compose-rs.yml <https://github.com/massquantity/LibRecommender/blob/master/libserving/docker-compose-rs.yml>`_ file,
change the corresponding ``MODEL_TYPE`` environment variable. You may also need to
change the path of volumes if model is stored in other place. Also, Redis has already
been included, so you don't need a Redis server locally. But one should start the docker
compose *before* saving data to Redis. For example in embed-based models:

.. code-block:: bash

    $ cd LibRecommender/libserving


.. code-block:: python3

    >>> from libreco.algorithms import ALS
    >>> from libreco.data import DatasetPure
    >>> from libserving.serialization import embed2redis, save_embed

    >>> train_data, data_info = DatasetPure.build_trainset(...)
    >>> model = ALS(...)
    >>> model.fit(...)  # train model
    >>> path = "embed_model"  # specify model saving directory
    >>> save_embed(path, model)  # save model in json format

    >>> from libserving.serialization import save_faiss_index
    >>> save_faiss_index(path, model)  # save faiss index to disk


.. code-block:: bash

    $ sudo docker compose -f docker-compose-rs.yml up  # start docker compose, which will load faiss index

.. code-block:: python3

    >>> embed2redis(path, host="0.0.0.0", port=6379, db=0)  # now load json from path and save model to redis


.. code-block:: bash

    $ curl -d '{"user": "1", "n_rec": 10}' -H "Content-Type: application/json" -X POST http://0.0.0.0:8080/embed/recommend
    # {"rec_list":["2628","1552","260","969","2193","733","1573","1917","1037","10"]}

For TensorFlow-based models, TensorFlow Serving config is in
`docker-compose-tf-serving.yml <https://github.com/massquantity/LibRecommender/blob/master/libserving/docker-compose-tf-serving.yml>`_ file,
so we need to start them both, and don't forget to change the ``MODEL_BASE_PATH`` and ``MODEL_NAME``
environment variables in the file:

.. code-block:: bash

    $ sudo docker compose -f docker-compose-rs.yml -f docker-compose-tf-serving.yml up

.. code-block:: bash

    $ curl -d '{"user": "1", "n_rec": 10}' -H "Content-Type: application/json" -X POST http://0.0.0.0:8080/tf/recommend
    # {"rec_list":["858","260","2355","1287","527","2371","1220","377","1968","3362"]}
