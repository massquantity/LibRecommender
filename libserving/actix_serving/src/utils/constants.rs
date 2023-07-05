use crate::redis_ops::RedisFeatKeys;

pub const KNN_MODELS: [&'static str; 2] = ["UserCF", "ItemCF"];

pub const EMBED_MODELS: [&'static str; 14] = [
    "SVD",
    "SVDpp",
    "ALS",
    "BPR",
    "YouTubeRetrieval",
    "Item2Vec",
    "RNN4Rec",
    "Caser",
    "WaveNet",
    "DeepWalk",
    "NGCF",
    "LightGCN",
    "PinSage",
    "PinSageDGL",
];

pub const CROSS_FEAT_MODELS: [&'static str; 6] = [
    "WideDeep",
    "FM",
    "DeepFM",
    "YouTubeRanking",
    "AutoInt",
    "DIN",
];

pub const SEQ_EMBED_MODELS: [&'static str; 3] = ["RNN4Rec", "Caser", "WaveNet"];

pub const USER_ID_EMBED_MODELS: [&'static str; 2] = ["Caser", "WaveNet"];

pub const SEPARATE_FEAT_MODELS: [&'static str; 1] = ["TwoTower"];

pub const SPARSE_SEQ_MODELS: [&'static str; 1] = ["YouTubeRetrieval"];

pub const CROSS_SEQ_MODELS: [&'static str; 2] = ["YouTubeRanking", "DIN"];

pub const SPARSE_REDIS_KEYS: RedisFeatKeys = RedisFeatKeys {
    user_index: "user_sparse_col_index",
    item_index: "item_sparse_col_index",
    user_value: "user_sparse_values",
    item_value: "item_sparse_values",
};

pub const DENSE_REDIS_KEYS: RedisFeatKeys = RedisFeatKeys {
    user_index: "user_dense_col_index",
    item_index: "item_dense_col_index",
    user_value: "user_dense_values",
    item_value: "item_dense_values",
};
