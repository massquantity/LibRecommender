Data Info
=========

The :class:`~libreco.data.DataInfo` object stores almost all the useful information from the original data.
We admit there may be too much information in this object, but for the ease of use of the library,
we've decided not to split it.
So almost every model has a ``data_info`` attribute that is used to make recommendations.
Additionally, when saving and loading a model, the corresponding *DataInfo* should also be saved and loaded.

When using a ``feat`` model, the :class:`~libreco.data.DataInfo` object stores the unique features of
all users/items in the training data. However, if a user/item has different categories or values
in the training data (which may be unlikely if the data is clean :)), only the last one will be stored.
For example, if in one sample a user's age is 20, and in another sample this user's age becomes 25,
then only 25 will be kept. So here we basically assume the data is always sorted by time,
and you should do so if it doesn't.

Therefore, when you call ``model.predict(user=..., item=...)`` or ``model.recommend_user(user=...)``
for a feat model, the model will use the stored feature information in DataInfo.

The :class:`~libreco.data.DataInfo` object also stores users' consumed items, which can be useful in sequence models
and ``unconsumed`` sampler.

Changing User/Item Features
---------------------------
It is also possible to change the unique user/item feature values stored in *DataInfo*,
then the new features would be used in prediction and recommendation.

.. code-block:: python3

   >>> data_info.assign_user_features(user_data=data)
   >>> data_info.assign_item_features(item_data=data)

The passed ``data`` argument is a ``pandas.DataFrame`` that contains the user/item information.
Be careful with this assign operation if you are not sure if the features in ``data`` are useful.

.. SeeAlso::

    `changing_feature_example.py <https://github.com/massquantity/LibRecommender/blob/master/examples/changing_feature_example.py>`_
