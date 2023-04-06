Data Processing
===============

Data Format
-----------

Just normal data format, with each line representing a sample. One thing is important, the model assumes
the column index for the ``user``, ``item``, and ``label`` to be 0, 1 and 2, respectively. You may wish to change the column order if that's not the case.

If you only have one dataset, there are several ways to split it:

+ ``random_split``. Split the data randomly.
+ ``split_by_ratio``. For each user, assign a certain ratio of items to the test data.
+ ``split_by_num``.  For each user, assign a certain number of items to the test data.
+ ``split_by_ratio_chrono``. For each user, assign a certain ratio of items to the test data, where the items are sorted by time first. In this case, the data should contain a ``time`` column.
+ ``split_by_num_chrono``. For each user, assign certain number of items to test_data, where the items are sorted by time first. In this case, the data should contain a ``time`` column.

.. SeeAlso::

    `split_data_example.py <https://github.com/massquantity/LibRecommender/blob/master/examples/split_data_example.py>`_

.. CAUTION::
    **Some caveats about the data:**

    1. Your data should not contain any missing value. Otherwise, it may lead to unexpected behavior.
    2. If your data size is small (less than 10,000 rows), the four ``split_by_*`` function may not be suitable. Since the number of interacted items for each user may be only one or two, which makes it difficult to split the whole data. In this case ``random_split`` is more suitable.
    3. Some data may contain duplicate samples, e.g., a user may have clicked an item multiple times. In this case, the training and possible negative sampling will be done multiple times for the same sample. If you don't want this, consider using functions such as `drop_duplicates <https://pandas.pydata.org/docs/reference/api/pandas.DataFrame.drop_duplicates.html>`_ in Pandas before training.

.. _Task:

Task
----

In general, there are two kinds of tasks in LibRecommender: ``rating`` and ``ranking`` task. The ``rating`` task deals with explicit data such as ``MovieLens`` or ``Netflix`` dataset, whereas the ``ranking`` task deals with implicit data such as `Last.FM <https://grouplens.org/datasets/hetrec-2011/>`_ dataset. The main difference on usage between these two tasks are:

1. The ``task`` parameter must be specified when building a model.
2. Obviously the metrics used for evaluating should be different. For ``rating`` task, the available metrics are [``rmse``, ``mae``, ``r2``] , and for ``ranking`` task the available metrics are [``loss``, ``balanced_accuracy``, ``roc_auc``, ``pr_auc``, ``precision``, ``recall``, ``map``, ``ndcg``] .

For example, using the ``SVD`` model with ``rating`` task:

.. code-block:: python3

   >>> model = SVD(task="rating", ...)
   >>> model.fit(..., metrics=["rmse", "mae", "r2"])

For the ``ranking`` task with implicit datasets, models treat 1 as positive labels and 0 as negative ones; no other label format is allowed.
However, implicit datasets typically **only** contain positive feedback, so negative sampling is necessary to train a model effectively.
In this scenario, labels can be arbitrary values, as all samples will be treated as positive.

By the way, some models such as ``BPR`` , ``YouTubeRetrieval``, ``YouTubeRanking``, ``Item2Vec``, ``DeepWalk``, ``LightGCN`` etc. ,
can only be used for ``ranking`` tasks since they are specially designed for that.
Errors might be raised if one use them for ``rating`` task.

.. _Negative Sampling:

Negative Sampling
-----------------

Negative sampling is commonly used in model training for ranking tasks with implicit datasets. However, certain models,
such as ``UserCF``, ``ItemCF``, ``BPR``, ``YouTubeRetrieval``, and ``RNN4Rec`` with bpr loss, do not require negative sampling during training.

Despite this, negative labels are necessary when evaluating these models using metrics like ``cross_entropy`` loss, ``roc_auc``, ``pr_auc``.
Therefore, it is recommended to perform negative sampling on all training, evaluation, and test data,
provided that **your data is implicit and only contains positive labels**.

It is important to note that negative sampling is not needed for rating tasks or when your original data already includes negative samples.
The parameter for controlling this behavior is ``neg_sampling`` in ``fit()`` and ``evaluate()`` functions.

.. code-block:: python3

    >>> model.fit(train_data, neg_sampling=True, eval_data=eval_data, ...)
    >>> evaluate(model, test_data, neg_sampling=True, ...)

.. _negative-samplers:

Different sampling strategies are available for users, with most models supporting "random", "unconsumed", and "popular" options.
For more information on these strategies, please refer to the models' API reference:

.. code-block:: python3

    >>> model = RNN4Rec("ranking", data_info, sampler="popular")

.. warning::

    The function ``build_negative_samples()`` was previously used for performing negative sampling on data.
    It has been deprecated since version ``1.1.0`` and will be removed in the future:

    .. code-block:: python3

       >>> train_data.build_negative_samples(data_info, num_neg=1)
