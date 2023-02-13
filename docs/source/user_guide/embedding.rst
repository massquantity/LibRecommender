Embedding
=========

According to the `algorithm list <https://github.com/massquantity/LibRecommender#references>`_,
there are some algorithms that can generate user and item embeddings after training.
So LibRecommender provides public APIs to get them:

.. code-block:: python3

   >>> model = RNN4Rec(task="ranking", ...)
   >>> model.fit(train_data, ...)
   >>> model.get_user_embedding(user=1)  # get user embedding for user 1
   >>> model.get_item_embedding(item=2)  # get item embedding for item 2

One can also search for similar users/items based on embeddings. By default,
we use `nmslib <https://github.com/nmslib/nmslib>`_ to do approximate similarity
searching since it's generally fast, but some people may find it difficult
to build and install the library, especially on Windows platform or Python >= 3.10.
So one can fall back to numpy similarity calculation if nmslib is not available.

.. code-block:: python3

   >>> model = RNN4Rec(task="ranking", ...)
   >>> model.fit(train_data, ...)
   >>> model.init_knn(approximate=True, sim_type="cosine")
   >>> model.search_knn_users(user=1, k=3)
   >>> model.search_knn_items(item=2, k=3)

Before searching, one should call :class:`~libreco.bases.EmbedBase.init_knn` to initialize the index.
Set ``approximate=True`` if you can use nmslib, otherwise set ``approximate=False``.
The ``sim_type`` parameter should either be ``cosine`` or ``inner-product``.

.. SeeAlso::

   `knn_embedding_example.py <https://github.com/massquantity/LibRecommender/blob/master/examples/knn_embedding_example.py>`_

