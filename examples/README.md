# User Guide

The purpose of this guide is to illustrate some of the main features that `LibRecommender` provides. Example usages are all listed in this `examples/` folder. 

This guide only demonstrates the data processing, feature engineering and model training parts. For how to serve a trained model in `LibRecommender`, see [Serving Guide](https://github.com/massquantity/LibRecommender/tree/master/serving) .



## Task

There are generally two kinds of tasks in `LibRecommender` , i.e. `rating` and `ranking` task. The `rating` task deals with explicit data such as `MovieLens` or `Netflix` dataset, whereas the `ranking` task deals with implicit data such as [`Last.FM`](https://grouplens.org/datasets/hetrec-2011/) dataset. The main difference on usage between these two tasks are:

1. The `task` parameter must be specified when building a model.
2. Obviously the metrics used for evaluating should be different. For `rating` task, the available metrics are `["rmse", "mae", "r2"]` , and for `ranking` task the available metrics are `["loss", "balanced_accuracy", "roc_auc", "pr_auc", "precision", "recall", "map", "ndcg"]` .

For example, using the `SVD` model with `rating` task:

```python
>>> model = SVD(task="rating", ...)
>>> svd.fit(..., metrics=["rmse", "mae", "r2"])
```

The implicit data typically may only contains positive feedback, i.e. only has samples that labeled as 1. In this case negative sampling is needed to effectively train a model. We'll cover negative sampling issue in the last paragraph.

By the way, some models such as `BPR` , `KNNEmbedding`,  `YouTubeMatch` and `YouTubeRanking`, can only be used for `ranking` tasks because they are specially designed for that. 



## `Pure` and `Feat` model

`LibRecommender` is a hybrid recommender system, which means you can choose whether to use features other than user behaviors or not. For models only use user behaviors, we classify them as  `pure` models. This category includes `userCF, itemCF, SVD, SVD++, ALS, NCF, BPR, KnnEmbedding, RNN4Rec`. 

Then for models that can include other features (e.g., age, sex, name etc.), we call them `feat` models. This category includes `Wide & Deep, FM, DeepFM, YouTubeMatch, YouTubeRanking, AutoInt, DIN`.

 The main difference on usage between these two kinds of models are:

1.  `pure` models should use `DatasetPure` to process the data, and `feat` models should use `DatasetFeat` to process the data.
2. When using `feat` models, four parameters should be provided, i.e. `[sparse_col, dense_col, user_col, item_col]`, as otherwise the model will have no idea how to deal with all kinds of features. We'll discuss more details about features in next section.

You can find some typical usages in these files: 

+ [`pure_rating_example.py`](https://github.com/massquantity/LibRecommender/blob/master/examples/pure_rating_example.py)
+ [` pure_ranking_example.py ` ](https://github.com/massquantity/LibRecommender/blob/master/examples/pure_ranking_example.py)
+ [`feat_rating_example.py `](https://github.com/massquantity/LibRecommender/blob/master/examples/feat_rating_example.py)
+ [`feat_ranking_example.py`](https://github.com/massquantity/LibRecommender/blob/master/examples/feat_ranking_example.py)

Actually, there exists a third model category in `LibRecommender`, which we call them `seq` models. These models typically leverage some sequence or graph information of users. This category includes `KnnEmbedding, RNN4Rec, YouTuBeMatch, YouTubeRanking, DIN`. As you can see, `seq` models overlap with `pure` and `feat` models. But no need to worry, the APIs still remains the same, and you can just use the examples above.



## Feature Engineering

### `sparse` and `dense` features

Sparse features are typically categorical features such as sex, location, year, etc. These features are projected into low dimension vectors by using an embedding layer, and this is by far the most common way of handling these kinds of features.

Dense features are typically numerical features such as age, price, length, etc. Unfortunately, there is no common way of handling these features, so in `LibRecommender` we mainly use the method described in the [AutoInt](https://arxiv.org/pdf/1810.11921.pdf) paper. Specifically, every dense feature are also projected into low dimension vectors through an embedding layer, then the vectors are multiplied by the dense feature value itself. In this way, the authors of the paper argued that sparse and dense features can have interactions in models such as FM, DeepFM and of course, AutoInt.  

<br>

<div align="center">
 <img src="https://s3.ax1x.com/2020/11/16/DAq0PO.jpg">
</div>

<br>

So to be clear, for one dense feature, all samples of this feature will be projected into a same embedding vector. This is different from a sparse feature, where all samples of it will have different embedding vectors based on its concrete category.

Apart from `sparse` and `dense` features, `user` and `item` features should also be provided. Since in order to make predictions and recommendations, the model needs to know a feature belongs to user or item. So, in short, these parameters are `[sparse_col, dense_col, user_col, item_col]`  in `LibRecommender`.

### `multi_sparse` features

Often times categorical features can be multi-valued. For example, a movie may have multiple genres, as shown in the `genre` feature in the `MovieLens-1m` dataset:

```
1::Toy Story (1995)::Animation|Children's|Comedy
2::Jumanji (1995)::Adventure|Children's|Fantasy
3::Grumpier Old Men (1995)::Comedy|Romance
4::Waiting to Exhale (1995)::Comedy|Drama
5::Father of the Bride Part II (1995)::Comedy
```

Usually we can handle this kind of feature by using multi-hot encoding, so in `LibRecommender` they are called `multi_sparse` features. After some transformation, the data can become (just for illustration purpose):

| item_id | movie_name                         | genre1    | genre2     | genre3  |
| ------- | ---------------------------------- | --------- | ---------- | ------- |
| 1       | Toy Story (1995)                   | Animation | Children's | Comedy  |
| 2       | Jumanji (1995)                     | Adventure | Children's | Fantasy |
| 3       | Grumpier Old Men (1995)            | Comedy    | Romance    | missing |
| 4       | Waiting to Exhale (1995)           | Comedy    | Drama      | missing |
| 5       | Father of the Bride Part II (1995) | Comedy    | missing    | missing |

In this case, a `multi_sparse_col` should be provided, see [`multi_sparse_example.py`](https://github.com/massquantity/LibRecommender/blob/master/examples/multi_sparse_example.py). 

```python
multi_sparse_col = [["genre1", "genre2", "genre3"]]
```

Note it's a list of list, because there are possibly many multi_sparse features,  for instance, `[[a1, a2, a3], [b1, b2]]` .

`LibRecommender` also provides a convenient function (`split_multi_value`) to transform the original `multi_sparse` features to the divided features illustrated above. See [`multi_sparse_processing_example.py`](https://github.com/massquantity/LibRecommender/blob/master/examples/multi_sparse_processing_example.py)

```python
multi_value_col = ["genre"]
multi_sparse_col, multi_user_col, multi_item_col = split_multi_value(
    data, multi_value_col, sep="|", max_len=[3], pad_val="missing",
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

See [`split_data_example.py`](https://github.com/massquantity/LibRecommender/blob/master/examples/split_data_example.py) .



## Negative Sampling

For implicit data with only positive labels, negative sampling is typically needed for model training. There are some special cases, such as `user_cf, item_cf, BPR, YouTubeMatch, RNN4Rec with bpr loss`, because these models do not need negative sampling during training. However, when evaluating these models using some metrics such as `cross_entropy loss, roc_auc, pr_auc`, negative labels are indeed needed. So we recommend doing negative sampling as long as the data is implicit and only contains positive labels, no matter which model you choose. Also note that train_data and test_data should use different sampling seed.

```python
>>> train_data.build_negative_samples(data_info, item_gen_mode="random", num_neg=1, seed=2020)
>>> test_data.build_negative_samples(data_info, item_gen_mode="random", num_neg=1, seed=2222)
```



## Common Parameters

+ `task` : choose rating or ranking task.
+ `data_info` : an object which contains useful data information.
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



