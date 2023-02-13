Evaluation & Save/Load
======================

Evaluate During Training
------------------------

The standard procedure in LibRecommender is evaluating during training.
However, for some complex models doing full evaluation on eval data can be very
time-consuming, so you can specify some evaluation parameters to speed this up.

The default value of ``eval_batch_size`` is 8192, and you can use a higher value if
you have enough machine or GPU memory. On the contrary, if you encounter memory error during
evaluation, try reducing ``eval_batch_size``.

The ``eval_user_num`` parameter controls how many users to use in evaluation.
By default, it is ``None``, which uses all the users in eval data.
You can use a smaller value if the evaluation is slow, and this will sample ``eval_user_num``
users randomly from eval data.

.. code-block:: python3

    model.fit(
        train_data,
        verbose=2,
        shuffle=True,
        eval_data=eval_data,
        metrics=metrics,
    	k=10,  # parameter of metrics, e.g. recall at k, ndcg at k
    	eval_batch_size=8192,
    	eval_user_num=100,
    )


Evaluate After Training
-----------------------

After the training, one can use the :class:`~libreco.evaluation.evaluate` function to evaluate on test data directly.

By default, it also won't update features stored in :class:`~libreco.data.DataInfo`,
but you can choose ``update_features=True`` to achieve that.
Also note if your evaluation data **is implicit and only contains positive label**,
then negative sampling is needed by passing ``neg_sample=True``:

.. code-block:: python3

    eval_result = evaluate(
        model,
        data,
        eval_batch_size=8192,
        k=10,
        metrics=["roc_auc", "precision", "ndcg"],
        sample_user_num=2048,
        neg_sample=True,
        update_features=False,
        seed=2222,
    )

Save/Load Model
---------------

In general, we may want to save/load a model for two reasons:

1. Save the model, then load it to make some predictions and recommendations. This is called inference.
2. Save the model, then load it to retrain the model when we get some new data.

The ``save/load`` API mainly deal with the first one, and the retraining problem is quite
different, which will be covered in the :doc:`model_retrain`.
When making predictions and recommendations, it may be unnecessary to save all the model
variables. So one can pass ``inference_only=True`` to only save the essential model part.

After loading the model, one can also evaluate the model directly,
see `save_load_example.py <https://github.com/massquantity/LibRecommender/blob/master/examples/save_load_example.py>`_ for typical usages.
