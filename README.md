# LibRecommender

## Overview

**LibRecommender** is an easy-to-use recommender system focused on end-to-end recommendation. The main features are:

+ Implement a number of popular recommendation algorithms such as SVD, DeepFM, BPR etc.

+ Allow user to use pure behavior features as well as other meta features.

+ Automatically convert categorical features to sparse representation, thus ease the memory usage.

+ Enable negative sampling for implicit dataset.

+ Using Cython or Tensorflow to accelerate model training.

+ Provide end-to-end workflow, i.e. data handling / preprocessing -> model training -> evaluate -> serving.



## Usage
```python
from libreco.dataset import DatasetFeat
from libreco.algorithms import DeepFMFeat

conf = {
    "data_path": "path/to/your/data",
    "length": 500000,
    "user_col": 0,
    "item_col": 1,
    "label_col": 2,
    "numerical_col": [4],
    "categorical_col": [3, 5, 6, 7, 8],
    "merged_categorical_col": None,
    "user_feature_cols": [3, 4, 5],
    "item_feature_cols": [6, 7, 8],
    "convert_implicit": True,
    "build_negative": True,
    "num_neg": 2,
#    "batch_size": 2048,
    "sep": ",",
}

dataset = DatasetFeat(include_features=True)
dataset.build_dataset(**conf)

dfm = DeepFMFeat(lr=0.0002, n_epochs=10000, reg=0.1, embed_size=50,
                    batch_size=2048, dropout=0.0, task="ranking", neg_sampling=True)
dfm.fit(dataset, pre_sampling=False, verbose=1)
print(dfm.predict(1959, 1992))
print(dfm.recommend_user(19500, 7))
```


## Data Format


## Installation & Dependencies 

- Python 3.5 +
- tensorflow >= 1.12
- numpy >= 1.13
- pandas >= 0.21.0
- scipy >= 0.19.1
- scikit-learn >= 0.20.1





## References

|       Algorithm        | Category | Paper                                                        |
| :--------------------: | :------: | :----------------------------------------------------------- |
|   userKNN / itemKNN    |   pure   | [Item-Based Collaborative Filtering Recommendation Algorithms](http://www.ra.ethz.ch/cdstore/www10/papers/pdf/p519.pdf) |
|          SVD           |   pure   | [Matrix Factorization Techniques for Recommender Systems](https://datajobs.com/data-science-repo/Recommender-Systems-[Netflix].pdf) |
|         SVD ++         |   pure   | [Factorization Meets the Neighborhood: a Multifaceted Collaborative Filtering Model](https://dl.acm.org/citation.cfm?id=1401944) |
|        superSVD        |   pure   | [Factorization Meets the Neighborhood: a Multifaceted Collaborative Filtering Model](https://dl.acm.org/citation.cfm?id=1401944) |
|          ALS           |   pure   | 1. [Matrix Completion via Alternating Least Square(ALS)](https://stanford.edu/~rezab/classes/cme323/S15/notes/lec14.pdf) / <br>2. [Collaborative Filtering for Implicit Feedback Datasets](http://yifanhu.net/PUB/cf.pdf) / <br>3. [Applications of the Conjugate Gradient Method for Implicit Feedback Collaborative Filtering](http://citeseerx.ist.psu.edu/viewdoc/download?doi=10.1.1.379.6473&rep=rep1&type=pdf) |
|          NCF           |   pure   | [Neural Collaborative Filtering](https://arxiv.org/pdf/1708.05031.pdf) |
|          BPR           |   pure   | [BPR: Bayesian Personalized Ranking from Implicit Feedback](https://arxiv.org/ftp/arxiv/papers/1205/1205.2618.pdf) |
|      Wide & Deep       |   feat   | [Wide & Deep Learning for Recommender Systems](https://arxiv.org/pdf/1606.07792.pdf) |
|           FM           |   feat   | [Factorization Machines](https://www.csie.ntu.edu.tw/~b97053/paper/Rendle2010FM.pdf) |
|         DeepFM         |   feat   | [DeepFM: A Factorization-Machine based Neural Network for CTR Prediction](https://arxiv.org/pdf/1703.04247.pdf) |
| Youtube-Recommendation |   feat   | [Deep Neural Networks for YouTube Recommendations](<https://static.googleusercontent.com/media/research.google.com/zh-CN//pubs/archive/45530.pdf>) |
