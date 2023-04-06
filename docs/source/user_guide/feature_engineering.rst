Feature Engineering
===================

Sparse and Dense Features
+++++++++++++++++++++++++

Sparse features are typically categorical features such as sex, location, year, etc.
These features are projected into low dimension vectors through an embedding layer,
which is by far the most common way of handling sparse features.

Dense features are typically numerical features such as age, price, length, etc.
Unfortunately, there is no common way of handling these features, so in LibRecommender
we mainly use the method described in the `AutoInt <https://arxiv.org/pdf/1810.11921.pdf>`_ paper.

Specifically, every dense feature are also projected into low dimension vectors through
an embedding layer, then the vectors are multiplied by the dense feature value itself.
In this way, the authors of the paper argued that sparse and dense features can have
interactions in models such as FM, DeepFM and of course, AutoInt.

.. image:: /_static/autoint_feature.jpg
   :align: center
   :width: 80%
   :target: ../../html/_static/autoint_feature.jpg

So to be clear, for one dense feature, all samples of this feature will be projected into a same embedding vector. This is different from a sparse feature, where all samples of it will have different embedding vectors based on its concrete category.

Apart from ``sparse`` and ``dense`` features, ``user`` and ``item`` features should also be provided. Since in order to make predictions and recommendations, the model needs to know whether a feature belongs to user or item. So, in short, these parameters are [``sparse_col``, ``dense_col``, ``user_col``, ``item_col``].

Multi-Sparse Features
++++++++++++++++++++++

Often times categorical features can be multi-valued. For example, a movie may have multiple genres, as shown in the ``genre`` feature in the ``MovieLens-1m`` dataset:

.. code-block:: rust

    1::Toy Story (1995)::Animation|Children's|Comedy
    2::Jumanji (1995)::Adventure|Children's|Fantasy
    3::Grumpier Old Men (1995)::Comedy|Romance
    4::Waiting to Exhale (1995)::Comedy|Drama
    5::Father of the Bride Part II (1995)::Comedy


Usually we can handle this kind of feature by using multi-hot encoding,
so in LibRecommender they are called ``multi_sparse`` features. After some transformation,
the data can become like this (just for illustration purpose):

+---------+------------------------------------+------------+------------+---------+
| item_id | movie_name                         | genre1     | genre2     | genre3  |
+=========+====================================+============+============+=========+
| 1       | Toy Story (1995)                   | Animation  | Children's | Comedy  |
+---------+------------------------------------+------------+------------+---------+
| 2       | Jumanji (1995)                     | Adventure  | Children's | Fantasy |
+---------+------------------------------------+------------+------------+---------+
| 3       | Grumpier Old Men (1995)            | Comedy     | Romance    | missing |
+---------+------------------------------------+------------+------------+---------+
| 4       | Waiting to Exhale (1995)           | Comedy     | Drama      | missing |
+---------+------------------------------------+------------+------------+---------+
| 5       | Father of the Bride Part II (1995) | Comedy     | missing    | missing |
+---------+------------------------------------+------------+------------+---------+

In this case, a ``multi_sparse_col`` can be used:

.. code-block:: python3

   multi_sparse_col = [["genre1", "genre2", "genre3"]]


Note it's a list of list, because there are possibly many multi_sparse features,  for instance, ``[[a1, a2, a3], [b1, b2]]`` .

When you specify a feature as ``multi_sparse`` feature like this, each sub-feature, i.e. ``genre1``, ``genre2``, ``genre3`` in the table above, will share the same embedding of the original feature ``genre``. Whether the embedding sharing would improve the model performance is data-dependent. But one thing is certain, it will reduce the total number of model parameters.

LibRecommender provides multiple ways of dealing with ``multi_sparse`` features,
i.e. ``normal``, ``sum`` , ``mean`` and ``sqrtn``. ``normal`` means treating
each sub-feature's embedding separately, and in most cases they will be concatenated at last.
``sum`` and ``mean`` means computing the sum or mean of each sub-feature's embedding,
in this case they are combined as one feature. ``sqrtn`` means the result of ``sum``
divided by the square root of sub-feature number, e.g. sqrt(3) in ``genre`` feature.
I'm not sure about this, but I think this ``sqrtn`` idea originally came from `SVD++ <https://people.engr.tamu.edu/huangrh/Spring16/papers_course/matrix_factorization.pdf>`_,
and it was also used in *Scaled Dot-Product Attention* part of `Transformer <https://arxiv.org/pdf/1706.03762.pdf>`_.
Generally the four methods described here have similar functionality as in `tf.nn.embedding_lookup_sparse <https://www.tensorflow.org/versions/r1.15/api_docs/python/tf/nn/embedding_lookup_sparse>`_,
but we didn't use it directly in our implementation since it has no ``normal`` option.

So in general you should choose a strategy in parameter ``multi_sparse_combiner`` when
building models with ``multi_sparse`` features:

.. code-block:: python3

   >>> model = DeepFM(..., multi_sparse_combiner="sqrtn")  # other options: normal, sum, mean

Note the ``genre`` feature above has different number of sub-features among all the samples.
Some movie only has one genre, whereas others may have three. So the value "missing" is used to pad them into same length.
However, when using ``sum``, ``mean`` or ``sqrtn`` to combine these sub-features,
the padding value should be excluded. Thus, you can pass the ``pad_val`` parameter when
building the data, and the model will do all the work. Otherwise, the padding value will
be included in the transformed features.

.. code-block:: python3

   >>> train_data, data_info = DatasetFeat.build_trainset(multi_sparse_col=[["genre1", "genre2", "genre3"]], pad_val=["missing"])


Although here we use "missing" as the padding value, this is not always appropriate.
It is fine with ``str`` type, but with numerical features, a value with corresponding type should be used.
e.g. 0 or -999.99.

Also be aware that the ``pad_val`` parameter is a list and should have
the same length as the number of ``multi_sparse`` features, and the reason for this is obvious.
So all in all an example script is enough to illustrate the usage of ``multi_sparse`` features,
see `multi_sparse_example.py <https://github.com/massquantity/LibRecommender/blob/master/examples/multi_sparse_example.py>`_.



LibRecommender also provides a convenient function (``split_multi_value``) to transform the original ``multi_sparse`` features to the divided sub-features illustrated above.

.. literalinclude:: ../../../examples/multi_sparse_processing_example.py
   :caption: From file `examples/multi_sparse_processing_example.py <https://github.com/massquantity/LibRecommender/blob/master/examples/multi_sparse_processing_example.py>`_
   :name: multi_sparse_processing_example.py
   :lines: 24-32
