Installation
============

From `pypi <https://pypi.org/project/LibRecommender/>`_ :

.. code-block:: bash

    $ pip install LibRecommender

To build from source, you will first need `Cython <https://cython.org/>`_ and `Numpy <https://numpy.org/>`_:

.. code-block:: bash

    $ # pip install numpy cython
    $ git clone https://github.com/massquantity/LibRecommender.git
    $ cd LibRecommender
    $ pip install .

-----------------------

Dependencies
++++++++++++

.. HINT::

    LibRecommender contains two modules: `libreco <https://github.com/massquantity/LibRecommender/tree/master/libreco>`_
    for ``training`` and  `libserving <https://github.com/massquantity/LibRecommender/tree/master/libserving>`_
    for ``serving``. If one only wants to train a model, dependencies for `libserving` are not needed.

.. Attention::

    The following dependencies will **NOT** be installed along with LibRecommender to
    avoid messing up your local dependencies. Make sure your dependencies meet the version requirements.


Dependencies for libreco:
^^^^^^^^^^^^^^^^^^^^^^^^^

+ Python >= 3.6
+ TensorFlow >= 1.15
+ PyTorch >= 1.10
+ Numpy >= 1.19.5
+ Cython >= 0.29.0
+ Pandas >= 1.0.0
+ Scipy >= 1.2.1
+ scikit-learn >= 0.20.0
+ gensim >= 4.0.0
+ tqdm
+ `nmslib <https://github.com/nmslib/nmslib>`_ (optional, see :doc:`user_guide/embedding`)
+ `DGL <https://github.com/dmlc/dgl>`_ (optional, see :ref:`Implementation Details <pinsage>`)

.. NOTE::

    If you are using Python 3.6, you also need to install `dataclasses <https://github.com/ericvsmith/dataclasses>`_, which was first introduced in Python 3.7.

**Known issue**: Sometimes one may encounter errors like
``ValueError: numpy.ndarray size changed, may indicate binary incompatibility. Expected 88 from C header, got 80 from PyObject.``
In this case try upgrading numpy, and version 1.22.0 or higher is probably a safe option.

Dependencies for libserving:
^^^^^^^^^^^^^^^^^^^^^^^^^^^^

+ Python >= 3.7
+ sanic >= 22.3
+ requests
+ aiohttp
+ pydantic
+ `ujson <https://github.com/ultrajson/ultrajson>`_
+ `redis <https://redis.io/>`_
+ `redis-py <https://github.com/redis/redis-py>`_ >= 4.2.0
+ `faiss <https://github.com/facebookresearch/faiss>`_ >= 1.5.2
+ `TensorFlow Serving <https://github.com/tensorflow/serving>`_ == 2.8.2
