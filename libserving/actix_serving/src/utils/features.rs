use std::collections::HashMap;

use deadpool_redis::{redis::AsyncCommands, Connection as RedisConnection};
use serde::de::DeserializeOwned;
use serde::Serialize;
use serde_json::Value;
use serde_with::skip_serializing_none;

use crate::constants::{
    CROSS_FEAT_MODELS, CROSS_SEQ_MODELS, DENSE_REDIS_KEYS, SPARSE_REDIS_KEYS, USER_ID_EMBED_MODELS,
};
use crate::errors::{ServingError, ServingResult};
use crate::redis_ops::{get_str, RedisFeatKeys};

pub(crate) struct DynFeats {
    pub(crate) user_feats: Option<HashMap<String, Value>>,
    pub(crate) seq: Option<Vec<Value>>,
    pub(crate) candidate_num: u32,
}

#[skip_serializing_none]
#[derive(Debug, Serialize)]
#[serde(untagged)]
pub(crate) enum Features {
    EmbedSeq {
        user_indices: Option<Vec<u32>>,
        #[serde(rename(serialize = "user_interacted_seq"))]
        seqs: Vec<Vec<u32>>, // shape: (1, seq_len)
        #[serde(rename(serialize = "user_interacted_len"))]
        seq_lens: Vec<u32>,
        #[serde(rename(serialize = "k"))]
        cand_num: Option<u32>,
    },

    SparseSeq {
        user_indices: Vec<u32>,
        user_sparse_indices: Option<Vec<Vec<u32>>>, // shape: (1, feat_len)
        user_dense_values: Option<Vec<Vec<f32>>>,
        item_interaction_indices: Vec<Vec<i64>>, // shape: (seq_len, 2)
        item_interaction_values: Vec<i64>,
        #[serde(rename(serialize = "modified_batch_size"))]
        batch_size: u32,
        #[serde(rename(serialize = "k"))]
        cand_num: Option<u32>,
    },

    Separate {
        user_indices: Vec<u32>,
        item_indices: Vec<u32>,
        user_sparse_indices: Option<Vec<Vec<u32>>>, // shape: (1, feat_len)
        user_dense_values: Option<Vec<Vec<f32>>>,
        item_sparse_indices: Option<Vec<Vec<u32>>>, // shape: (n_items, feat_len)
        item_dense_values: Option<Vec<Vec<f32>>>,
        #[serde(rename(serialize = "k"))]
        cand_num: Option<u32>,
    },

    Cross {
        user_indices: Vec<u32>,
        item_indices: Vec<u32>,
        sparse_indices: Option<Vec<Vec<u32>>>, // shape: (n_items, feat_len)
        dense_values: Option<Vec<Vec<f32>>>,
        #[serde(rename(serialize = "user_interacted_seq"))]
        seqs: Option<Vec<Vec<u32>>>, // shape: (n_items, seq_len)
        #[serde(rename(serialize = "user_interacted_len"))]
        seq_lens: Option<Vec<u32>>,
        #[serde(rename(serialize = "k"))]
        cand_num: Option<u32>,
    },
}

pub(crate) async fn build_seq_embed_features(
    model_name: &str,
    user_id: &str,
    n_items: u32,
    user_consumed: &[u32],
    conn: &mut RedisConnection,
    dyn_feats: Option<&DynFeats>,
) -> ServingResult<Features> {
    let num = n_items as usize;
    let id = user_id.parse::<u32>()?;

    let user_indices = if USER_ID_EMBED_MODELS.contains(&model_name) {
        Some(vec![id])
    } else {
        None
    };

    let (user_seq, cand_num) = match dyn_feats {
        Some(df) => (&df.seq, Some(df.candidate_num)),
        None => (&None, None),
    };
    let (seqs, seq_lens) = get_seq(model_name, user_consumed, num, conn, user_seq).await?;

    let features = Features::EmbedSeq {
        user_indices,
        seqs,
        seq_lens,
        cand_num,
    };

    Ok(features)
}

pub(crate) async fn build_sparse_seq_features(
    user_id: &str,
    n_items: u32,
    user_consumed: &[u32],
    conn: &mut RedisConnection,
    dyn_feats: Option<&DynFeats>,
) -> ServingResult<Features> {
    let id = user_id.parse::<u32>()?;
    let user_indices = vec![id];

    let (user_feats, user_seq, cand_num) = match dyn_feats {
        Some(df) => (&df.user_feats, &df.seq, Some(df.candidate_num)),
        None => (&None, &None, None),
    };

    let (user_sparse_feats, user_dense_feats) = split_user_feats(user_feats, conn).await?;
    let user_sparse_indices = if conn.exists("user_sparse_values").await? {
        Some(get_user_feats("user_sparse_values", user_id, conn, user_sparse_feats, 1).await?)
    } else {
        None
    };
    let user_dense_values = if conn.exists("user_dense_values").await? {
        Some(get_user_feats("user_dense_values", user_id, conn, user_dense_feats, 1).await?)
    } else {
        None
    };

    let (item_interaction_indices, item_interaction_values) =
        get_sparse_seq(user_consumed, n_items, conn, user_seq).await?;

    let features = Features::SparseSeq {
        user_indices,
        user_sparse_indices,
        user_dense_values,
        item_interaction_indices,
        item_interaction_values,
        batch_size: 1,
        cand_num,
    };

    Ok(features)
}

pub(crate) async fn build_separate_features(
    user_id: &str,
    n_items: u32,
    conn: &mut RedisConnection,
    dyn_feats: Option<&DynFeats>,
) -> ServingResult<Features> {
    let num = n_items as usize;
    let id = user_id.parse::<u32>()?;
    let user_indices = vec![id];
    let item_indices = (0..n_items).collect::<Vec<u32>>();

    let (user_feats, cand_num) = match dyn_feats {
        Some(df) => (&df.user_feats, Some(df.candidate_num)),
        None => (&None, None),
    };

    let (user_sparse_feats, user_dense_feats) = split_user_feats(user_feats, conn).await?;
    let user_sparse_indices = if conn.exists("user_sparse_values").await? {
        Some(get_user_feats("user_sparse_values", user_id, conn, user_sparse_feats, 1).await?)
    } else {
        None
    };
    let user_dense_values = if conn.exists("user_dense_values").await? {
        Some(get_user_feats("user_dense_values", user_id, conn, user_dense_feats, 1).await?)
    } else {
        None
    };

    let item_sparse_indices = if conn.exists("item_sparse_values").await? {
        Some(get_item_feats("item_sparse_values", num, conn).await?)
    } else {
        None
    };
    let item_dense_values = if conn.exists("item_dense_values").await? {
        Some(get_item_feats("item_dense_values", num, conn).await?)
    } else {
        None
    };

    let features = Features::Separate {
        user_indices,
        item_indices,
        user_sparse_indices,
        user_dense_values,
        item_sparse_indices,
        item_dense_values,
        cand_num,
    };

    Ok(features)
}

pub(crate) async fn build_cross_features(
    model_name: &str,
    user_id: &str,
    n_items: u32,
    user_consumed: &[u32],
    conn: &mut RedisConnection,
    dyn_feats: Option<&DynFeats>,
) -> ServingResult<Features> {
    let num = n_items as usize;
    let id = user_id.parse::<u32>()?;
    let user_indices = vec![id; num];
    let item_indices = (0..n_items).collect::<Vec<u32>>();

    let (user_feats, user_seq, cand_num) = match dyn_feats {
        Some(df) => (&df.user_feats, &df.seq, Some(df.candidate_num)),
        None => (&None, &None, None),
    };
    let (user_sparse_feats, user_dense_feats) = split_user_feats(user_feats, conn).await?;

    let sparse_indices: Option<Vec<Vec<u32>>> =
        combine_features::<u32>(SPARSE_REDIS_KEYS, user_id, num, conn, user_sparse_feats).await?;

    let dense_values: Option<Vec<Vec<f32>>> =
        combine_features::<f32>(DENSE_REDIS_KEYS, user_id, num, conn, user_dense_feats).await?;

    let (seqs, seq_lens) = if CROSS_SEQ_MODELS.contains(&model_name) {
        let (sq, sql) = get_seq(model_name, user_consumed, num, conn, user_seq).await?;
        (Some(sq), Some(sql))
    } else {
        (None, None)
    };

    let features = Features::Cross {
        user_indices,
        item_indices,
        sparse_indices,
        dense_values,
        seqs,
        seq_lens,
        cand_num,
    };
    // let feature_json = serde_json::to_value(features)?;
    Ok(features)
}

async fn combine_features<T>(
    redis_keys: RedisFeatKeys,
    user_id: &str,
    n_items: usize,
    conn: &mut RedisConnection,
    user_feats: Vec<(usize, T)>,
) -> ServingResult<Option<Vec<Vec<T>>>>
where
    T: Copy + Clone + Default + DeserializeOwned + Serialize,
{
    let has_user_feats: bool = conn.exists(redis_keys.user_index).await?;
    let has_item_feats: bool = conn.exists(redis_keys.item_index).await?;
    let features = match (has_user_feats, has_item_feats) {
        (true, true) => {
            Some(get_cross_feats(redis_keys, user_id, n_items, conn, user_feats).await?)
        }
        (true, false) => {
            Some(get_user_feats(redis_keys.user_value, user_id, conn, user_feats, n_items).await?)
        }
        (false, true) => Some(get_item_feats(redis_keys.item_value, n_items, conn).await?),
        _ => None,
    };
    Ok(features)
}

async fn get_cross_feats<T>(
    redis_keys: RedisFeatKeys,
    user_id: &str,
    n_items: usize,
    conn: &mut RedisConnection,
    user_feats: Vec<(usize, T)>,
) -> ServingResult<Vec<Vec<T>>>
where
    T: Copy + Clone + Default + DeserializeOwned + Serialize,
{
    let user_col_index: Vec<usize> = get_index_from_redis(redis_keys.user_index, conn).await?;
    let item_col_index: Vec<usize> = get_index_from_redis(redis_keys.item_index, conn).await?;
    let mut user_vals: Vec<T> =
        get_one_feat_from_redis(redis_keys.user_value, user_id, conn).await?;
    if !user_feats.is_empty() {
        user_feats
            .into_iter()
            .for_each(|(idx, val)| user_vals[idx] = val)
    }
    let item_vals: Vec<Vec<T>> = get_item_feats(redis_keys.item_value, n_items, conn).await?;

    let dim = user_col_index.len() + item_col_index.len();
    let mut features = vec![vec![T::default(); dim]; n_items];
    for item_id in 0..n_items {
        let user_idx_iter = user_col_index.iter();
        let user_val_iter = user_vals.iter();
        for (i, v) in user_idx_iter.zip(user_val_iter) {
            features[item_id][*i] = *v;
        }

        let item_idx_iter = item_col_index.iter();
        let item_val_iter = item_vals[item_id].iter();
        for (i, v) in item_idx_iter.zip(item_val_iter) {
            features[item_id][*i] = *v;
        }
    }
    Ok(features)
}

async fn get_user_feats<T>(
    value_name: &str,
    user_id: &str,
    conn: &mut RedisConnection,
    user_feats: Vec<(usize, T)>,
    repeat_num: usize,
) -> ServingResult<Vec<Vec<T>>>
where
    T: Copy + DeserializeOwned,
{
    let mut user_vals: Vec<T> = get_one_feat_from_redis(value_name, user_id, conn).await?;
    if !user_feats.is_empty() {
        user_feats
            .into_iter()
            .for_each(|(idx, val)| user_vals[idx] = val)
    }

    Ok(vec![user_vals; repeat_num])
}

async fn get_item_feats<T>(
    value_name: &str,
    n_items: usize,
    conn: &mut RedisConnection,
) -> ServingResult<Vec<Vec<T>>>
where
    T: DeserializeOwned,
{
    get_all_feat_from_redis(value_name, n_items, conn).await
}

async fn get_index_from_redis(
    index_name: &str,
    conn: &mut RedisConnection,
) -> ServingResult<Vec<usize>> {
    let index_str = get_str(conn, index_name, "none", "get").await?;
    let index: Vec<usize> = serde_json::from_str(&index_str)?;
    Ok(index)
}

async fn get_one_feat_from_redis<T>(
    value_name: &str,
    user_id: &str,
    conn: &mut RedisConnection,
) -> ServingResult<Vec<T>>
where
    T: DeserializeOwned,
{
    let value_str = get_str(conn, value_name, user_id, "hget").await?;
    let value: Vec<T> = serde_json::from_str(&value_str)?;
    Ok(value)
}

async fn get_all_feat_from_redis<T>(
    value_name: &str,
    n_items: usize,
    conn: &mut RedisConnection,
) -> ServingResult<Vec<Vec<T>>>
where
    T: DeserializeOwned,
{
    let value_strs: Vec<String> = conn.lrange(value_name, 0, -1).await?;
    let values = value_strs
        .into_iter()
        .take(n_items)
        .map(|i| serde_json::from_str(&i))
        .collect::<Result<Vec<Vec<_>>, _>>()?;
    Ok(values)
}

async fn get_sparse_seq(
    user_consumed: &[u32],
    n_items: u32,
    conn: &mut RedisConnection,
    user_seq: &Option<Vec<Value>>,
) -> ServingResult<(Vec<Vec<i64>>, Vec<i64>)> {
    let max_seq_len: usize = conn.get("max_seq_len").await?;
    let user_seq = json_to_seq(user_seq, n_items, conn).await?;
    let original_seq = if user_seq.is_empty() {
        user_consumed.to_vec()
    } else {
        user_seq
    };

    let (sparse_indices, sparse_values) = if original_seq.is_empty() {
        (vec![vec![0, 0]], vec![-1])
    } else {
        let seq_len = std::cmp::min(max_seq_len, original_seq.len());
        let indices = vec![vec![0, 0]; seq_len];
        let src_start = original_seq.len() - seq_len;
        let values = original_seq[src_start..]
            .iter()
            .map(|&i| if i < n_items { i as i64 } else { -1 })
            .collect();
        (indices, values)
    };

    Ok((sparse_indices, sparse_values))
}

async fn get_seq(
    model_name: &str,
    user_consumed: &[u32],
    n_items: usize,
    conn: &mut RedisConnection,
    user_seq: &Option<Vec<Value>>,
) -> ServingResult<(Vec<Vec<u32>>, Vec<u32>)> {
    if !conn.exists("max_seq_len").await? {
        eprintln!("{} has missing `max_seq_len`", model_name);
        return Err(ServingError::Other("Missing `max_seq_len` attribute"));
    }

    let repeat_num = if CROSS_FEAT_MODELS.contains(&model_name) {
        n_items
    } else {
        1
    };

    let max_seq_len: usize = conn.get("max_seq_len").await?;
    let user_seq = json_to_seq(user_seq, n_items as u32, conn).await?;
    let original_seq = if user_seq.is_empty() {
        user_consumed
    } else {
        user_seq.as_slice()
    };

    let (res_seq, res_seq_len) = if !original_seq.is_empty() {
        let mut seq = vec![n_items as u32; max_seq_len];
        let seq_len = std::cmp::min(max_seq_len, original_seq.len());
        let src_start = original_seq.len() - seq_len;
        seq[0..seq_len].copy_from_slice(&original_seq[src_start..]);
        (seq, seq_len)
    } else {
        (vec![n_items as u32; max_seq_len], max_seq_len)
    };

    Ok((
        vec![res_seq; repeat_num],
        vec![res_seq_len as u32; repeat_num],
    ))
}

async fn split_user_feats(
    user_features: &Option<HashMap<String, Value>>,
    conn: &mut RedisConnection,
) -> ServingResult<(Vec<(usize, u32)>, Vec<(usize, f32)>)> {
    let mut user_sparse_feats: Vec<(usize, u32)> = Vec::new();
    let mut user_dense_feats: Vec<(usize, f32)> = Vec::new();
    if let Some(user_feats) = user_features {
        for (col, val) in user_feats {
            if let Some(sparse_idx_val) = json_to_sparse_index(val, col, conn).await? {
                user_sparse_feats.push(sparse_idx_val);
            } else if let Some(dense_idx_val) = json_to_dense_value(val, col, conn).await? {
                user_dense_feats.push(dense_idx_val);
            }
        }
    }

    Ok((user_sparse_feats, user_dense_feats))
}

fn json_to_str(value: &Value) -> Option<String> {
    if value.is_string() {
        value.as_str().map(str::to_string)
    } else if value.is_i64() {
        value.as_i64().map(|i| i.to_string())
    } else {
        None
    }
}

async fn json_to_sparse_index(
    value: &Value,
    col: &str,
    conn: &mut RedisConnection,
) -> ServingResult<Option<(usize, u32)>> {
    let mut res: Option<(usize, u32)> = None;
    if conn.hexists("user_sparse_fields", col).await? {
        if let Some(val) = json_to_str(value) {
            let mapping_key = format!("user_sparse_idx_mapping__{col}");
            if conn.hexists(&mapping_key, &val).await? {
                let field_index = conn.hget("user_sparse_fields", col).await?;
                let sparse_index = conn.hget(&mapping_key, &val).await?;
                res.replace((field_index, sparse_index));
            } else {
                log::warn!("Unknown value `{val}` in sparse feature `{col}`")
            }
        } else {
            log::warn!("Failed to convert `{value}` from sparse feature `{col}`");
        }
    }

    Ok(res)
}

async fn json_to_dense_value(
    value: &Value,
    col: &str,
    conn: &mut RedisConnection,
) -> ServingResult<Option<(usize, f32)>> {
    let mut res: Option<(usize, f32)> = None;
    if conn.hexists("user_dense_fields", col).await? {
        if let Some(dense_val) = value.as_f64().map(|i| i as f32) {
            let field_index = conn.hget("user_dense_fields", col).await?;
            res.replace((field_index, dense_val));
        } else {
            log::warn!("Failed to convert `{value}` from dense feature `{col}`");
        }
    }

    Ok(res)
}

async fn json_to_seq(
    seq_values: &Option<Vec<Value>>,
    n_items: u32,
    conn: &mut RedisConnection,
) -> ServingResult<Vec<u32>> {
    let mut seq = Vec::new();
    if let Some(seq_vals) = seq_values {
        for v in seq_vals {
            let mut item = n_items;
            if let Some(i) = json_to_str(v) {
                if conn.hexists("item2id", &i).await? {
                    item = conn.hget("item2id", &i).await?
                }
            }
            seq.push(item)
        }
    }

    Ok(seq)
}
