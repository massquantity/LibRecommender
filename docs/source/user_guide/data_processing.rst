Data Processing
===============

Data Format
-----------

Just normal data format, each line represents a sample. One thing is important, the model assumes that ``user``, ``item``, and ``label`` column index are 0, 1, and 2, respectively. You may wish to change the column order if that's not the case.

If you have only one data, you can split the data in following ways:

+ ``random_split``. Split the data randomly.
+ ``split_by_ratio``. For each user, assign a certain ratio of items to test_data.
+ ``split_by_num``.  For each user, assign a certain number of items to test_data.
+ ``split_by_ratio_chrono``. For each user, assign certain ratio of items to test_data, where items are sorted by time first. In this case, data should contain a ``time`` column.
+ ``split_by_num_chrono``. For each user, assign certain number of items to test_data, where items are sorted by time first. In this case, data should contain a ``time`` column.

.. SeeAlso::

    `split_data_example.py <https://github.com/massquantity/LibRecommender/blob/master/examples/split_data_example.py>`_.

.. CAUTION::
    **Some caveats about the data:**

    1. Your data should not contain any missing value. Otherwise, it may lead to unexpected behavior.
    2. If your data size is small (less than 10,000 rows), the four ``split_by_*`` function may not be suitable. Since the number of interacted items for each user may be only one or two, which makes it difficult to split the whole data. In this case ``random_split`` is more suitable.
    3. Some data may contain duplicate samples, e.g., a user may have clicked an item multiple times. In this case, the training and possible negative sampling will be done multiple times for the same sample. If you don't want this, consider using functions such as `drop_duplicates <https://pandas.pydata.org/docs/reference/api/pandas.DataFrame.drop_duplicates.html>`_ in Pandas before training.

.. _Task:

Task
----

There are generally two kinds of tasks in LibRecommender , i.e. ``rating`` and ``ranking`` task. The ``rating`` task deals with explicit data such as ``MovieLens`` or ``Netflix`` dataset, whereas the ``ranking`` task deals with implicit data such as `Last.FM <https://grouplens.org/datasets/hetrec-2011/>`_ dataset. The main difference on usage between these two tasks are:

1. The ``task`` parameter must be specified when building a model.
2. Obviously the metrics used for evaluating should be different. For ``rating`` task, the available metrics are [``rmse``, ``mae``, ``r2``] , and for ``ranking`` task the available metrics are [``loss``, ``balanced_accuracy``, ``roc_auc``, ``pr_auc``, ``precision``, ``recall``, ``map``, ``ndcg``] .

For example, using the ``SVD`` model with ``rating`` task:

.. code-block:: python3

   >>> model = SVD(task="rating", ...)
   >>> model.fit(..., metrics=["rmse", "mae", "r2"])


The implicit data typically may only contain positive feedback, i.e. only has samples that labeled as 1.
In this case, negative sampling is needed to effectively train a model.

By the way, some models such as ``BPR`` , ``YouTubeRetrieval``, ``YouTubeRanking``, ``Item2Vec``, ``DeepWalk``, ``LightGCN`` etc. ,
can only be used for ``ranking`` tasks since they are specially designed for that.
Errors will be raised if one use them for ``rating`` task.

Negative Sampling
-----------------

For implicit data with only positive labels, negative sampling is typically used in model training.
There are some special cases, such as ``UserCF``, ``ItemCF``, ``BPR``, ``YouTubeRetrieval``, ``RNN4Rec with bpr loss``,
where these models do not need to do negative sampling during training.
However, when evaluating these models using some metrics such as ``cross_entropy loss``, ``roc_auc``, ``pr_auc``,
negative labels are indeed needed.

For PyTorch-based models, **only eval or test data needs negative sampling**. These models includes ``NGCF``, ``LightGCN``, ``GraphSage``, ``GraphSageDGL``, ``PinSage``, ``PinSageDGL`` , see `torch_ranking_example.py <https://github.com/massquantity/LibRecommender/blob/master/examples/torch_ranking_example.py>`_.

For other models, performing negative sampling on all the train, eval and test data is recommended as long as **your data is implicit and only contains positive labels**.

.. code-block:: python3

   >>> train_data.build_negative_samples(data_info, item_gen_mode="random", num_neg=1, seed=2020)
   >>> test_data.build_negative_samples(data_info, item_gen_mode="random", num_neg=1, seed=2222)

In the future, we plan to remove this explicit negative sampling part before training. This requires encapsulating the sampling process into the batch training, so that users won't undertake the ambiguity above. Some other sampling methods apart from ``random`` will also be added.
