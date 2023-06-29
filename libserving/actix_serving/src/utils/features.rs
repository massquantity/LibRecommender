use deadpool_redis::{redis::AsyncCommands, Connection as RedisConnection};
use serde::de::DeserializeOwned;
use serde::Serialize;
use serde_with::skip_serializing_none;

use crate::constants::TF_SEQ_MODELS;
use crate::errors::{ServingError, ServingResult};
use crate::redis_ops::get_str;

#[skip_serializing_none]
#[derive(Debug, Serialize)]
pub(crate) struct CrossFeats {
    user_indices: Vec<u32>,
    item_indices: Vec<u32>,
    sparse_indices: Option<Vec<Vec<u32>>>,
    dense_values: Option<Vec<Vec<f32>>>,
    #[serde(rename(serialize = "user_interacted_seq"))]
    seqs: Option<Vec<Vec<u32>>>,
    #[serde(rename(serialize = "user_interacted_len"))]
    seq_lens: Option<Vec<u32>>,
}

pub(crate) async fn build_cross_features(
    model_name: &str,
    user_id: &str,
    n_items: u32,
    user_consumed: &[u32],
    conn: &mut RedisConnection,
) -> ServingResult<CrossFeats> {
    let num = n_items as usize;
    let id = user_id.parse::<u32>()?;
    let user_indices = vec![id; num];
    let item_indices = (0..n_items).collect::<Vec<u32>>();

    let sparse_indices: Option<Vec<Vec<u32>>> = combine_features::<u32>(
        "user_sparse_col_index",
        "item_sparse_col_index",
        "user_sparse_values",
        "item_sparse_values",
        user_id,
        num,
        conn,
    )
    .await?;

    let dense_values: Option<Vec<Vec<f32>>> = combine_features::<f32>(
        "user_dense_col_index",
        "item_dense_col_index",
        "user_dense_values",
        "item_dense_values",
        user_id,
        num,
        conn,
    )
    .await?;

    let (seqs, seq_lens) = if TF_SEQ_MODELS.contains(&model_name) {
        let (seq, seq_len) = get_seq(model_name, user_consumed, num, conn).await?;
        (Some(seq), Some(seq_len))
    } else {
        (None, None)
    };

    let features = CrossFeats {
        user_indices,
        item_indices,
        sparse_indices,
        dense_values,
        seqs,
        seq_lens,
    };
    // let feature_json = serde_json::to_value(features)?;
    Ok(features)
}

async fn combine_features<T>(
    user_index_name: &str,
    item_index_name: &str,
    user_value_name: &str,
    item_value_name: &str,
    user_id: &str,
    n_items: usize,
    conn: &mut RedisConnection,
) -> ServingResult<Option<Vec<Vec<T>>>>
where
    T: Copy + Clone + Default + DeserializeOwned + Serialize,
{
    let has_user_feats: bool = conn.exists(user_index_name).await?;
    let has_item_feats: bool = conn.exists(item_index_name).await?;
    let features = match (has_user_feats, has_item_feats) {
        (true, true) => Some(
            get_cross_feats(
                user_index_name,
                item_index_name,
                user_value_name,
                item_value_name,
                user_id,
                n_items,
                conn,
            )
            .await?,
        ),
        (true, false) => Some(get_user_feats(user_value_name, user_id, n_items, conn).await?),
        (false, true) => Some(get_item_feats(item_value_name, n_items, conn).await?),
        _ => None,
    };
    Ok(features)
}

async fn get_cross_feats<T>(
    user_index_name: &str,
    item_index_name: &str,
    user_value_name: &str,
    item_value_name: &str,
    user_id: &str,
    n_items: usize,
    conn: &mut RedisConnection,
) -> ServingResult<Vec<Vec<T>>>
where
    T: Copy + Clone + Default + DeserializeOwned + Serialize,
{
    let user_col_index: Vec<usize> = get_index_from_redis(user_index_name, conn).await?;
    let item_col_index: Vec<usize> = get_index_from_redis(item_index_name, conn).await?;
    let user_vals: Vec<T> = get_one_feat_from_redis(user_value_name, user_id, conn).await?;
    let item_vals: Vec<Vec<T>> = get_all_feat_from_redis(item_value_name, n_items, conn).await?;

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
    n_items: usize,
    conn: &mut RedisConnection,
) -> ServingResult<Vec<Vec<T>>>
where
    T: Copy + DeserializeOwned,
{
    let user_vals: Vec<T> = get_one_feat_from_redis(value_name, user_id, conn).await?;
    Ok(vec![user_vals; n_items])
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

async fn get_seq(
    model_name: &str,
    user_consumed: &[u32],
    n_items: usize,
    conn: &mut RedisConnection,
) -> ServingResult<(Vec<Vec<u32>>, Vec<u32>)> {
    if !conn.exists("max_seq_len").await? {
        eprintln!("{} has missing `max_seq_len`", model_name);
        return Err(ServingError::Other("Missing `max_seq_len` attribute"));
    }

    let max_seq_len: usize = conn.get("max_seq_len").await?;
    let (res_seq, res_seq_len) = if !user_consumed.is_empty() {
        let mut seq = vec![n_items as u32; max_seq_len];
        let seq_len = std::cmp::min(max_seq_len, user_consumed.len());
        let src_start = user_consumed.len() - seq_len;
        seq[0..seq_len].copy_from_slice(&user_consumed[src_start..]);
        (seq, seq_len)
    } else {
        (vec![n_items as u32; max_seq_len], max_seq_len)
    };

    Ok((vec![res_seq; n_items], vec![res_seq_len as u32; n_items]))
}
