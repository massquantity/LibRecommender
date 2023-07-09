Recommendation
==============

By default, the recommendation result returned by ``model.recommend_user()`` method will
filter out items that a user has previously consumed.

However, if you use a very large ``n_rec`` and number of consumed items for this user plus ``n_rec`` exceeds number of total items,
i.e. ``len(user_consumed) + n_rec > n_items``, the consumed items will not be filtered out
since there are not enough items to recommend. If you don't want to filter out consumed items,
set ``filter_consumed=False``.

LibRecommender also supports random recommendation by setting ``random_rec=True``
(By default it is False). Of course, it's not completely random, but random sampling
based on each item's prediction scores. It's basically a trade-off between accuracy and diversity.

Finally, batch recommendation is also supported by simply passing a list to the ``user`` parameter.
The returned result will be a dict, with users as keys and ``numpy.array`` as values.

.. code-block:: python3

   >>> model.recommend_user(user=[1, 2, 3], n_rec=3, filter_consumed=True, random_rec=False)
   # returns {1: array([2529, 1196, 2916]), 2: array([ 541,  750, 1299]), 3: array([3183, 2722, 2672])}

Cold Start
----------

It is very common to encounter new users or items that doesn't exist in training data,
which is hard to make recommendations for them. This is the notorious "cold-start" problem in recommender system.

There are two strategies in LibRecommender to handle the cold-start problem: ``popular`` and ``average``.
The ``popular`` strategy simply returns the most popular items in training data.

The ``average`` strategy means using the average of all the user/item embeddings as the
representation of the cold-start user/item. Once we have the embedding, we can make
predictions and recommendations. This strategy indicates that a cold-start user/item's
behavior is treated as the "average" behavior of all the known users/items.

Likewise, the new category of one feature are also handled as an average embedding of the
known categories of this feature. See `pure_example.py <https://github.com/massquantity/LibRecommender/blob/master/examples/pure_example.py>`_,
`feat_example.py <https://github.com/massquantity/LibRecommender/blob/master/examples/feat_example.py>`_
for cold-start usage.

.. SEEALSO::
   :doc:`embedding`

Dynamic Features & Sequences
----------------------------

In real-world scenarios, users' features are very likely to change every time we make recommendations
for them. For example, a user's location may change many times a day, and we may need to take this
into account. This feature issue can actually be combined with the cold-start issue. For example,
a user has appeared in training data, but his/her location doesn't exist in training data's ``location``
feature.

On the other hand, user behavior sequences can also play a crucial role in recommender systems.
So how do we handle these dynamic feature and sequence problems? Fortunately, LibRecommender can deal with them elegantly.

If you want to predict or recommend with specific features, the usage is pretty straightforward.
For prediction, just pass the ``feats`` argument, which only accepts ``dict`` type:

.. code-block:: python3

   >>> model.predict(user=1, item=110, feats={"sex": "F", "occupation": 2, "age": 23})


There is no need to specify a feature belongs to user or item, because this information
has already been stored in model's :class:`~libreco.data.DataInfo` object. Note if you misspelled some feature names,
e.g. "sex" -> "sax", the model will simply ignore this feature. If you pass a feature category
that doesn't appear in training data, e.g. "sex" -> "bisexual", then it will be ignored too.

To make recommendation for a user based on dynamic features, pass the user features to the ``user_feats`` argument and the user's
item sequence to the ``seq`` argument. Only ``feat`` models support the assignment of user features,
and the models support sequence recommendations are ``RNN4Rec``, ``Caser``, ``WaveNet``,
``YouTubeRetrieval``, ``YouTubeRanking`` and ``DIN``.

.. code-block:: python3

   >>> model.recommend_user(user=1, n_rec=7, cold_start="popular",
   >>>                      user_feats={"sex": "F", "occupation": 2, "age": 23},
   >>>                      seq=[1, 22, 333])

.. SeeAlso::

    `seq_example.py <https://github.com/massquantity/LibRecommender/blob/master/examples/seq_example.py>`_

.. error::

    Please note that batch recommendation cannot be used with dynamic features and sequences, so the following code will raise an error:

    .. code-block:: python3

       >>> model.recommend_user(user=[1, 2, 3, 0], n_rec=7, user_feats={"sex": "F", "occupation": 2, "age": 23})
