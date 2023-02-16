Model & Train
=============

``Pure`` and ``Feat`` model
---------------------------

LibRecommender is a hybrid recommender system, which means you can choose whether to use
features other than user behaviors or not. For models only use user behaviors, we classify
them as ``pure`` models. This category includes ``UserCF``, ``ItemCF``, ``SVD``, ``SVD++``,
``ALS``, ``NCF``, ``BPR``, ``RNN4Rec``, ``Item2Vec``, ``Caser``, ``WaveNet``, ``DeepWalk``,
``NGCF``, ``LightGCN``.

Then for models that can include other features (e.g., age, sex, name etc.), we call
them ``feat`` models. This category includes ``WideDeep``, ``FM``, ``DeepFM``, ``YouTubeRetrieval``,
``YouTubeRanking``, ``AutoInt``, ``DIN``, ``GraphSage``, ``PinSage``.

The main difference on usage between these two kinds of models are:

1.  ``pure`` models should use :class:`~libreco.data.dataset.DatasetPure` to process data,
and ``feat`` models should use :class:`~libreco.data.dataset.DatasetFeat` to process data.

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

Loss
----

LibRecommender provides some options on loss type for ``ranking`` :ref:`task <Task>`.
The default loss type for ``ranking`` is cross entropy loss. Since version ``0.10.0``,
focal loss was added into the library. First introduced in `Lin et al., 2018 <https://arxiv.org/pdf/1708.02002.pdf>`_,
focal loss down-weights well-classified examples and focuses on hard examples to get better
training performance, and here is the `implementation <https://github.com/massquantity/LibRecommender/blob/master/libreco/tfops/loss.py#L34>`_.

In order to choose which loss to use, simply set the ``loss_type`` parameter:

.. code-block:: python3

   >>> model = Caser(task="ranking", loss_type="cross_entropy", ...)
   >>> model = Caser(task="ranking", loss_type="focal", ...)

There are some special cases:

+ Some algorithms are hard to assign explicit loss type,
  including ``UserCF``, ``ItemCF``, ``ALS``, ``Item2Vec``, ``DeepWalk``,
  so they don't have ``loss_type`` parameter.

+ As its name suggests, ``BPR`` can only use ``bpr`` loss.

+ The ``YouTubeRetrieval`` algorithm is also different, its ``loss_type`` is either
  ``sampled_softmax`` or ``nce``.

+ Finally, with ``RNN4Rec`` algorithm, one can choose three ``loss_type``,
  i.e. ``cross_entropy``, ``focal``, ``bpr``.

We are aware that these loss restrictions are hard to remember at once, so this leaves room
for further improvements:)
