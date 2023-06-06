Model & Train
=============

Pure and Feat Models
--------------------

LibRecommender is a hybrid recommender system, which means you can choose whether to use
features other than user behaviors or not. For models only use user behaviors, we classify
them as ``pure`` models. This category includes ``UserCF``, ``ItemCF``, ``SVD``, ``SVD++``,
``ALS``, ``NCF``, ``BPR``, ``RNN4Rec``, ``Item2Vec``, ``Caser``, ``WaveNet``, ``DeepWalk``,
``NGCF``, ``LightGCN``.

Then for models that can use other features (e.g., age, sex, name etc.), we call
them ``feat`` models. This category includes ``WideDeep``, ``FM``, ``DeepFM``, ``YouTubeRetrieval``,
``YouTubeRanking``, ``AutoInt``, ``DIN``, ``GraphSage``, ``PinSage``, ``TwoTower``.

The main difference on usage between these two kinds of models are:

1.  ``pure`` models should use :class:`~libreco.data.dataset.DatasetPure` to process data,
and ``feat`` models should use :class:`~libreco.data.dataset.DatasetFeat`.

2. When using ``feat`` models, four parameters should be provided,
i.e. [``sparse_col``, ``dense_col``, ``user_col``, ``item_col``], as otherwise the model will
have no idea how to deal with all kinds of features.

The ``fit()`` method is the sole method for training a model in LibRecommender.
You can find some typical usages in these examples:

.. SeeAlso::

    + `pure_rating_example.py <https://github.com/massquantity/LibRecommender/blob/master/examples/pure_rating_example.py>`_
    + `pure_ranking_example.py <https://github.com/massquantity/LibRecommender/blob/master/examples/pure_ranking_example.py>`_
    + `feat_rating_example.py <https://github.com/massquantity/LibRecommender/blob/master/examples/feat_rating_example.py>`_
    + `feat_ranking_example.py <https://github.com/massquantity/LibRecommender/blob/master/examples/feat_ranking_example.py>`_

In fact, there exists two other kinds of model categories in LibRecommender, and we call them
``sequence`` and ``graph`` models. You can find them in the `algorithm list <https://github.com/massquantity/LibRecommender#references>`_.

Sequence models leverage information of user behavior sequence, whereas Graph models leverage information from graph.
As you can see, these models overlap with ``pure`` and ``feat`` models. But no need to worry,
the APIs remain the same, and you can just refer to the examples above.

Multiprocess data loading
-------------------------

For most models, users can leverage `torch.utils.data.DataLoader <https://pytorch.org/docs/stable/data.html#torch.utils.data.DataLoader>`_
to enable multiprocess data loading and speed up training.
Users can specify the number of workers to be used during data loading using the ``num_workers`` parameter. Please refer to the corresponding
`PyTorch documentation <https://pytorch.org/docs/stable/data.html#single-and-multi-process-data-loading>`_ for more information.

.. code-block:: python3

    >>> model.fit(train_data, neg_sampling=False, num_workers=2)

Loss
----

LibRecommender provides some options on loss type for *ranking* :ref:`task <Task>`.
The default loss type for most models is *cross entropy* loss. Since version ``0.10.0``,
focal loss was added into the library. First introduced in `Lin et al., 2018 <https://arxiv.org/pdf/1708.02002.pdf>`_,
focal loss down-weights well-classified examples and focuses on hard examples to get better
training performance, and here is the `implementation <https://github.com/massquantity/LibRecommender/blob/master/libreco/tfops/loss.py#L34>`_.

In order to choose which loss to use, simply set the ``loss_type`` parameter:

.. code-block:: python3

   >>> model = Caser(task="ranking", loss_type="cross_entropy", ...)
   >>> model = Caser(task="ranking", loss_type="focal", ...)

The table below lists the losses and :ref:`negative samplers <negative-samplers>` that can be used for `ranking` task in each algorithm:

+--------------------------------------------------------------------------------------+---------------------------------------+----------------------------------------+
|                                      Algorithm                                       |                 Loss                  |                Sampler                 |
+======================================================================================+=======================================+========================================+
|                       UserCF, ItemCF, ALS, Item2Vec, DeepWalk                        |                   /                   |                   /                    |
+--------------------------------------------------------------------------------------+---------------------------------------+----------------------------------------+
|                                         BPR                                          |                  bpr                  |      random, unconsumed, popular       |
+--------------------------------------------------------------------------------------+---------------------------------------+----------------------------------------+
|                                   YouTubeRetrieval                                   |          sampled_softmax, nce         |             uniform, other             |
+--------------------------------------------------------------------------------------+---------------------------------------+----------------------------------------+
| SVD, SVD++, NCF, Wide&Deep, FM, DeepFM, YouTubeRanking, AutoInt, DIN, Caser, WaveNet |         cross_entropy, focal          |      random, unconsumed, popular       |
+--------------------------------------------------------------------------------------+---------------------------------------+----------------------------------------+
|                                       RNN4Rec                                        |       cross_entropy, focal, bpr       |      random, unconsumed, popular       |
+--------------------------------------------------------------------------------------+---------------------------------------+----------------------------------------+
|                                   NGCF, LightGCN                                     | cross_entropy, focal, bpr, max_margin |      random, unconsumed, popular       |
+--------------------------------------------------------------------------------------+---------------------------------------+----------------------------------------+
|                     GraphSage, GraphSageDGL, PinSage, PinSageDGL                     | cross_entropy, focal, bpr, max_margin | random, unconsumed, popular, out-batch |
+--------------------------------------------------------------------------------------+---------------------------------------+----------------------------------------+
|                                       TwoTower                                       |   cross_entropy, max_margin, softmax  |      random, unconsumed, popular       |
+--------------------------------------------------------------------------------------+---------------------------------------+----------------------------------------+

.. caution::

    *bpr* and *max_margin* belong to pairwise loss, so they must be used with negative sampling,
    which means your data should only contains positive samples when using these losses.
