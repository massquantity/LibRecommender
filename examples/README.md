# User Guide

The purpose of this guide is to illustrate some of the main features that LibRecommender provides. Example usages are all listed in this `examples/` folder. 

This guide only demonstrates the data processing, feature engineering and model training parts. For how to serve a trained model in LibRecommender, see [Serving Guide](https://github.com/massquantity/LibRecommender/tree/master/libserving) .



## Task

There are generally two kinds of tasks in LibRecommender , i.e. `rating` and `ranking` task. The `rating` task deals with explicit data such as `MovieLens` or `Netflix` dataset, whereas the `ranking` task deals with implicit data such as [`Last.FM`](https://grouplens.org/datasets/hetrec-2011/) dataset. The main difference on usage between these two tasks are:

1. The `task` parameter must be specified when building a model.
2. Obviously the metrics used for evaluating should be different. For `rating` task, the available metrics are [`rmse`, `mae`, `r2`] , and for `ranking` task the available metrics are [`loss`, `balanced_accuracy`, `roc_auc`, `pr_auc`, `precision`, `recall`, `map`, `ndcg`] .

For example, using the `SVD` model with `rating` task:

```python
>>> model = SVD(task="rating", ...)
>>> svd.fit(..., metrics=["rmse", "mae", "r2"])
```

The implicit data typically may only contains positive feedback, i.e. only has samples that labeled as 1. In this case negative sampling is needed to effectively train a model. We'll cover negative sampling issue in the [section below](#negative-sampling).

By the way, some models such as `BPR` , `YouTubeRetrieval`, `YouTubeRanking`, `Item2Vec`, `DeepWalk`, `LightGCN` etc. , can only be used for `ranking` tasks since they are specially designed for that. 



## `Pure` and `Feat` model

LibRecommender is a hybrid recommender system, which means you can choose whether to use features other than user behaviors or not. For models only use user behaviors, we classify them as  `pure` models. This category includes `UserCF`, `ItemCF`, `SVD`, `SVD++`, `ALS`, `NCF`, `BPR`, `RNN4Rec`, `Item2Vec`, `Caser`, `WaveNet`, `DeepWalk`, `NGCF`, `LightGCN`. 

Then for models that can include other features (e.g., age, sex, name etc.), we call them `feat` models. This category includes `WideDeep`, `FM`, `DeepFM`, `YouTubeRetrieval`, `YouTubeRanking`, `AutoInt`, `DIN`.

 The main difference on usage between these two kinds of models are:

1.  `pure` models should use `DatasetPure` to process data, and `feat` models should use `DatasetFeat` to process data.
2. When using `feat` models, four parameters should be provided, i.e. [`sparse_col`, `dense_col`, `user_col`, `item_col`], as otherwise the model will have no idea how to deal with all kinds of features. We'll discuss more about features in next section.

You can find some typical usages in these files: 

+ [`pure_rating_example.py`](https://github.com/massquantity/LibRecommender/blob/master/examples/pure_rating_example.py)
+ [`pure_ranking_example.py`](https://github.com/massquantity/LibRecommender/blob/master/examples/pure_ranking_example.py)
+ [`feat_rating_example.py`](https://github.com/massquantity/LibRecommender/blob/master/examples/feat_rating_example.py)
+ [`feat_ranking_example.py`](https://github.com/massquantity/LibRecommender/blob/master/examples/feat_ranking_example.py)

In fact, there exists two other kinds of model categories in LibRecommender, and we call them `sequence` and `graph` models. You can find them in the [algorithm list](https://github.com/massquantity/LibRecommender#references).

Sequence models leverage information of user behavior sequence, whereas Graph models leverage information from graph. As you can see, these models overlap with `pure` and `feat` models. But no need to worry, the APIs remain the same, and you can just refer to the examples above.



## Feature Engineering

### `sparse` and `dense` features

Sparse features are typically categorical features such as sex, location, year, etc. These features are projected into low dimension vectors by using an embedding layer, and this is by far the most common way of handling these kinds of features.

Dense features are typically numerical features such as age, price, length, etc. Unfortunately, there is no common way of handling these features, so in LibRecommender we mainly use the method described in the [AutoInt](https://arxiv.org/pdf/1810.11921.pdf) paper. Specifically, every dense feature are also projected into low dimension vectors through an embedding layer, then the vectors are multiplied by the dense feature value itself. In this way, the authors of the paper argued that sparse and dense features can have interactions in models such as FM, DeepFM and of course, AutoInt.  

<br>

<div align="center">
 <img src="https://s3.ax1x.com/2020/11/16/DAq0PO.jpg">
</div>

<br>

So to be clear, for one dense feature, all samples of this feature will be projected into a same embedding vector. This is different from a sparse feature, where all samples of it will have different embedding vectors based on its concrete category.

Apart from `sparse` and `dense` features, `user` and `item` features should also be provided. Since in order to make predictions and recommendations, the model needs to know whether a feature belongs to user or item. So, in short, these parameters are [`sparse_col`, `dense_col`, `user_col`, `item_col`].

### `multi_sparse` features

Often times categorical features can be multi-valued. For example, a movie may have multiple genres, as shown in the `genre` feature in the `MovieLens-1m` dataset:

```
1::Toy Story (1995)::Animation|Children's|Comedy
2::Jumanji (1995)::Adventure|Children's|Fantasy
3::Grumpier Old Men (1995)::Comedy|Romance
4::Waiting to Exhale (1995)::Comedy|Drama
5::Father of the Bride Part II (1995)::Comedy
```

Usually we can handle this kind of feature by using multi-hot encoding, so in LibRecommender they are called `multi_sparse` features. After some transformation, the data can become (just for illustration purpose):

| item_id | movie_name                         | genre1    | genre2     | genre3  |
| ------- | ---------------------------------- | --------- | ---------- | ------- |
| 1       | Toy Story (1995)                   | Animation | Children's | Comedy  |
| 2       | Jumanji (1995)                     | Adventure | Children's | Fantasy |
| 3       | Grumpier Old Men (1995)            | Comedy    | Romance    | missing |
| 4       | Waiting to Exhale (1995)           | Comedy    | Drama      | missing |
| 5       | Father of the Bride Part II (1995) | Comedy    | missing    | missing |

In this case, a `multi_sparse_col` can be used:  

```python
multi_sparse_col = [["genre1", "genre2", "genre3"]]
```

Note it's a list of list, because there are possibly many multi_sparse features,  for instance, `[[a1, a2, a3], [b1, b2]]` .

When you specify a feature as `multi_sparse` feature like this, each sub-feature, i.e. `genre1`, `genre2`, `genre3` in the table above, will share the same embedding of the original feature `genre`. Whether the embedding sharing would improve the model performance is data-dependent. But one thing is certain, it will reduce the total number of model parameters.

LibRecommender provides multiple ways of dealing with `multi_sparse` features, i.e. `normal`, `sum` , `mean` and `sqrtn`. `normal` means treating each sub-feature's embedding separately, and in most cases they will be concatenated at last. `sum` and `mean` means computing the sum or mean of each sub-feature's embedding, in this case they are combined as one feature. `sqrtn` means the result of `sum` divided by the square root of sub-feature number, e.g. sqrt(3) in `genre` feature. I'm not sure about this, but I think this `sqrtn` idea originally came from [SVD++](https://people.engr.tamu.edu/huangrh/Spring16/papers_course/matrix_factorization.pdf). Generally the four methods described here have similar functionality as in [tf.nn.embedding_lookup_sparse](https://www.tensorflow.org/versions/r1.15/api_docs/python/tf/nn/embedding_lookup_sparse), but we didn't use it directly in our implementation since it has no `normal` choice.

So in general you should choose a strategy in parameter `multi_sparse_combiner` when building models with `multi_sparse` features:

```python
>>> model = DeepFM(..., multi_sparse_combiner="sqrtn")  # other options: normal, sum, mean
```

Note the `genre` feature above has different number of sub-features among all the samples. Some movie only has one genre, whereas others may have three. So the value "missing" is used to pad them into same length. However, when using `sum`, `mean` or `sqrtn` to combine these sub-features, the padding value should be excluded. Thus you can pass the `pad_val` parameter when building the data, and the model will do all the work. Otherwise the padding value will be included in the transformed features.

```python
>>> train_data, data_info = DatasetFeat.build_trainset(multi_sparse_col=[["genre1", "genre2", "genre3"]], pad_val=["missing"])
```

Although here we use "missing" as the padding value, this is not always true. It is fine with `str` type, but with features that is numerical type, a value with corresponding type should be used. e.g. 0 or -999.99. Also be aware that the `pad_val` parameter is a list and should have the same length as the number of `multi_sparse` features, and the reason for this is obvious. So all in all an example script is enough to illustrate the usage of `multi_sparse` features, see [`multi_sparse_example.py`](https://github.com/massquantity/LibRecommender/blob/master/examples/multi_sparse_example.py).



LibRecommender also provides a convenient function (`split_multi_value`) to transform the original `multi_sparse` features to the divided sub-features illustrated above. See [`multi_sparse_processing_example.py`](https://github.com/massquantity/LibRecommender/blob/master/examples/multi_sparse_processing_example.py)

```python
multi_value_col = ["genre"]
multi_sparse_col, multi_user_col, multi_item_col = split_multi_value(
    data, multi_value_col, sep="|", max_len=[3], pad_val=["missing"],
    user_col=user_col, item_col=item_col
)
```



## Data Format

JUST normal data format, each line represents a sample. One thing is important, the model assumes that `user`, `item`, and `label` column index are 0, 1, and 2, respectively. You may wish to change the column order if that's not the case.

If you have only one data, you can split the data in following ways:

+ `random_split`. Split the data randomly.
+ `split_by_ratio`. For each user, assign certain ratio of items to test_data.
+ `split_by_num`.  For each user, assign certain number of items to test_data.
+ `split_by_ratio_chrono`. For each user, assign certain ratio of items to test_data, where items are sorted by time first. In this case, data should contain a `time` column.
+ `split_by_num_chrono`. For each user, assign certain number of items to test_data, where items are sorted by time first. In this case, data should contain a `time` column.

**Note that your data should not contain any missing value.**

See [`split_data_example.py`](https://github.com/massquantity/LibRecommender/blob/master/examples/split_data_example.py) .



## Negative Sampling

For implicit data with only positive labels, negative sampling is typically needed for model training. There are some special cases, such as `UserCF`, `ItemCF`, `BPR`, `YouTubeRetrieval`, `RNN4Rec with bpr loss`, because these models do not need negative sampling during training. However, when evaluating these models using some metrics such as `cross_entropy loss`, `roc_auc`, `pr_auc`, negative labels are indeed needed. So we recommend doing negative sampling as long as the data is implicit and only contains positive labels, no matter which model you choose. Also note that train_data and test_data should use different sampling seed.

```python
>>> train_data.build_negative_samples(data_info, item_gen_mode="random", num_neg=1, seed=2020)
>>> test_data.build_negative_samples(data_info, item_gen_mode="random", num_neg=1, seed=2222)
```



## Loss

LibRecommender provides some options on loss type for `ranking` task. The default loss type for `ranking` is cross entropy loss. Since version `0.10.0`, focal loss was added into the library. First introduced in [Lin et al., 2018](https://arxiv.org/pdf/1708.02002.pdf), focal loss down-weights well-classified examples and focuses on hard examples to get better training performance, and here is the [implementation](https://github.com/massquantity/LibRecommender/blob/master/libreco/tfops/loss.py#L34). In order to choose which loss to use, simply set the `loss_type` parameter:

```python
>>> model = Caser(task="ranking", loss_type="cross_entropy", ...)
>>> model = Caser(task="ranking", loss_type="focal", ...)
```

There are some special cases. Some algorithms are hard to assign explicit loss type, including `UserCF`, `ItemCF`, `ALS`, `Item2Vec`, `DeepWalk`, so they don't have `loss_type` parameter. Some algorithms can only use bpr loss, including `BPR`, `NGCF`, `LightGCN`, so don't bother to choose loss for them either.

The `YouTubeRetrieval` algorithm is also different, its `loss_type` is either `sampled_softmax` or `nce`. Finally, with `RNN4Rec` algorithm, one can choose three `loss_type`, i.e. `cross_entropy`, `focal`, `bpr`.

We are aware that these loss restrictions are hard to remember at once, so this leaves room for further improvement.



## Embedding

According to the [algorithm list](https://github.com/massquantity/LibRecommender#references), there are some algorithms that can generate final user and item embeddings. So LibRecommender provides public APIs to get them:

```python
>>> model = RNN4Rec(task="ranking", ...)
>>> model.fit(train_data, ...)
>>> model.get_user_embedding(user=1)  # get user embedding for user 1
>>> model.get_item_embedding(item=2)  # get item embedding for item 2
```

One can also search for similar users/items based on embeddings. By default we use [nmslib](https://github.com/nmslib/nmslib) to do approximate similarity searching since it's generally fast, but some people may find it difficult to build and install the library, especially on Windows platform or Python >= 3.10. So one can fall back to numpy similarity calculation if nmslib is not available. 

```python
>>> model = RNN4Rec(task="ranking", ...)
>>> model.fit(train_data, ...)
>>> model.init_knn(approximate=True, sim_type="cosine")
>>> model.search_knn_users(user=1, k=3)
>>> model.search_knn_items(item=2, k=3)
```

Before searching, one should call `init_knn` to initialize index. Set `approximate=True` if you can use nmslib, otherwise set `approximate=False`. The `sim_type` parameter should either be `cosine` or `inner-product`. 

Also see [`knn_embedding_example.py`](https://github.com/massquantity/LibRecommender/blob/master/examples/knn_embedding_example.py).



## Save/Load Model

In general, we may want to save/load a model for two reasons:

1. Save the model, then load it to make some predictions and recommendations.
2. Save the model, then load it to retrain the model when we get some new data.

The `save/load` API mainly deal with the first one, and the retrain problem is quite different, which will be covered in the [model retrain part](#model-retrain). When making predictions and recommendations, it may be unnecessary to save all the model variables. So one can pass `inference_only=True` to only save the essential model part.

After loading the model, one can also evaluate the model directly, see [save_load_example.py](https://github.com/massquantity/LibRecommender/blob/master/examples/save_load_example.py) for typical usages.



## Practical Issues in Recommendation System

Since the open source of LibRecommender, a lot of people have raised questions to us, either through Email or Github Issues. From our view, these questions can mainly be divided into three categories:

+ **Cold-Start problem**:  It is very common to encounter new users or items that doesn't exist in training data, so how can we make recommendations for them?

+ **Changing-Feature problem**:  In real-world scenarios, users' features are very likely to change every time we make recommendations for them. For example, a user's location may change many times a day, and we may need to take this into account. This feature problem can actually be combined with the cold-start problem. For example, a user has appeared in training data, but his/her location doesn't exist in training data's `location` feature, so what do we do?

+ **Model-retrain problem**:  When we get some new data, we definitely want to retrain the old model with these new data, but this is actually not easy for some deep learning models. The reason lies in the new users/items that may appear in the new data. In deep learning models, embedding variables are usually used, and their shape are preassigned and fixed during and after training. The same problem also goes with new features. If we load the old model and want to train it with new users/items/features, these embedding shape must be expanded, which is not allowed in TensorFlow (at least to my knowledge). One workaround is to combine the new data with the old data, then retrain the model with all of the data. But it would be a waste of time and resources to retrain the whole data every time we get some new data, so what do we do?

The next few sections will try to explain how we can handle these problems in LibRecommender. But first let us introduce the inner data structure inside the library, namely, the `DataInfo` object.

### Data Info

The `DataInfo` object saves almost all the useful information in the data. We admit there may be too much information in this object, but for ease of use of the library, we decided not to split it. So almost every model contains a `data_info` property to make recommendations. And when you save and load a model, you should also save/load the corresponding `DataInfo` .

When we are using a `feat` model, the `DataInfo` object will store the unique features of all the users/items in training data. If a user/item has different categories or values in training data (which may be unlikely if the data is clean :)), only the last one will be stored. For example, if in one sample a user's age is 20, and in another sample this user's age becomes 25, then only 25 will be kept. So here we basically assume the data is always sorted by time, and you should do so if it doesn't. 

Therefore when you call `model.predict(user=..., item=...)` or `model.recommend_user(user=...)` for a `feat` model, the model will use the stored feature information in `DataInfo`. It is also possible to change the feature values when making predictions and recommendations. See [Changing Feature](#changing-feature) section.



## Cold Start

There are two strategies in the library to handle cold-start problem: `popular` and `average`. The `popular` strategy simply returns the most popular items in training data. 

The `average` strategy means using the average of all the user/item embeddings as the representation of the cold-start user/item. Once we have the embedding, we can make predictions and recommendations. This strategy indicates that a cold-start user/item's behavior is treated as the "average" behavior of all the known users/items. 

Likewise, the new category of one feature are also handled as an average embedding of the known categories of this feature. See [pure_example.py](https://github.com/massquantity/LibRecommender/blob/master/examples/pure_example.py), [feat_example.py](https://github.com/massquantity/LibRecommender/blob/master/examples/feat_example.py) for cold-start usage.



## Changing Feature

If you want to predict or recommend with specific features, the usage is pretty straight forward.  For prediction, just pass the `feats` argument, which only accepts `dict` or `pands.Series` type: 

```python
>>> model.predict(user=1, item=110, feats={"sex": "F", "occupation": 2, "age": 23})
```

There is no need to specify a feature belongs to user or item, because these information has already been stored in model's `DataInfo` object. Note if you misspelled some feature names, e.g. "sex" -> "sax", the model will simply ignore this feature. If you pass a feature category that doesn't appear in training data, e.g. "sex" -> "bisexual", then it will be ignored too.

If you want to predict on a whole dataset with features, you can use the `predict_data_with_feats` function. By setting `batch_size` to `None`, the model will treat all the data as one batch, which may cause memory issues: 

```python
 >>> from libreco.prediction import predict_data_with_feats
 >>> predict_data_with_feats(model, data=dataset, batch_size=1024, cold_start="average")
```

To make recommendation for one user, we can pass the user features to `user_feats` argument. It actually doesn't make much sense to change the item features when making recommendation for only one user, but we provide an `item_data` argument anyway, which can change the item features. The type of `item_data` must be `pandas.DataFrame` . We assume one may want to change the features of multiple items, since it nearly makes no difference to the recommendation result if only one item's features have been changed.

```python
>>> model.recommend_user(user=1, n_rec=7, cold_start="popular", 
                         user_feats=pd.Series({"sex": "F", "occupation": 2, "age": 23}), 
                         item_data=item_features)
```

Note the three functions described above doesn't change the unique user/item features inside the `DataInfo` object. So the next time you call `model.predict(user=1, item=110)` , it will still use the features stored in `DataInfo`. However, if you do want to change the features in `DataInfo`, then you can use `assign_user_features` and `assign_item_features` :

```python
>>> data_info.assign_user_features(user_data=data)
>>> data_info.assign_item_features(item_data=data)
```

The passed `data` argument is a `pandas.DataFrame` that contains the user/item information. Be careful with this assign operation if you are not sure if the features in `data` are useful.

During evaluation, one can also evaluate directly on one data. By default it also won't update features in `DataInfo`, but you can choose `update_features=True` to achieve that. Also note that if your evaluation data is implicit and only contains positive label, then negative sampling is needed by passing `neg_sample=True` :

```python
eval_result = evaluate(model, data, eval_batch_size=8192, k=10,
                       metrics=["roc_auc", "precision", "ndcg"],
                       sample_user_num=2048, neg_sample=True,
                       update_features=False, seed=2222)
```

See [changing_feature_example.py](https://github.com/massquantity/LibRecommender/blob/master/examples/changing_feature_example.py) 



## Model Retrain 

When we want to retrain a deep learning model with new data that contains new users/items, the embedding variables' shapes need to be expanded. As described above, this is not allowed in TensorFlow. So how can we retrain a model if we can't change the shape of the variables in TensorFlow? Well, if we can't alter it, we create a new one, then explicitly assign the old one to the new one. Specifically, in TensorFlow, we build a new graph with variables with new shape, then assign the old values to the correct indices of new variables. For the new indices, i.e. the new user/item part, they are initialized randomly as usual.

The problem with this solution is that we can not use TensorFlow's default method such as [`tf.train.Saver` ](https://www.tensorflow.org/api_docs/python/tf/compat/v1/train/Saver)or [`tf.saved_model` ](https://tensorflow.google.cn/versions/r1.15/api_docs/python/tf/saved_model)to save the deep learning models, since it can only load to the exact same model with same shapes. So our solution is extracting all the variables to numpy array format, then save them using the save method in numpy. After that the variables are loaded from numpy, we then build the new graph and update the new variables with old ones. 

So it's crucial to set `manual=True, inference_only=False` when you save the model, which means leveraging the numpy way. If you set `manual=False`, the model may use the `tf.train.Saver` to save the model, which is OK if you are certain that there will be no new user/item in new data.

Before retraining the model, we also should build the new data. Since the old `data_info` already exists, we just need to update some information to the `data_info` by passing `revolution=True`. During recommendation, we always want to filter some items that a user has consumed, which are also stored in `DataInfo` object. So if you want to combine the user-consumed information in old data with that in new data, you can pass `merge_behavior=True`:

```python
train_data, data_info = DatasetFeat.build_trainset(
    train, revolution=True, data_info=data_info, merge_behavior=True
)
```

See [model_retrain_example.py](https://github.com/massquantity/LibRecommender/blob/master/examples/model_retrain_example.py).



## Common Parameters

+ `task` : choose rating or ranking task.
+ `data_info` : an object which contains useful data information.
+ `loss_type`: type of loss function.
+ `embed_size` : vector size used in embedding layer.
+ `n_epochs` : number of total training epochs.
+ `lr` : learning rate.
+ `lr_decay` : whether to use learning rate decay.
+ `reg` : L2 regularization parameter.
+ `batch_size` : training batch size.
+ `num_neg` : number of negative sampling items.
+ `use_bn` : whether to use batch normalization.
+ `dropout_rate` : specify dropout rate.
+ `hidden_units` : specify number of layers and hidden units, typically comma-separated string, such as `"128,64,32"`.
+ `recent_num` : used in sequence models, specify number of recent items to consider.



