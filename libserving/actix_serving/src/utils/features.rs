use std::collections::HashMap;

use deadpool_redis::{redis::AsyncCommands, Connection};
use serde::de::DeserializeOwned;
use serde::Serialize;
use serde_json::{json, Value};

use crate::constants::TF_SEQ_MODELS;
use crate::errors::{ServingError, ServingResult};
use crate::redis_ops::get_str;

pub struct Features {
    user_sparse_col: Option<String>,
    user_sparse_indices: Option<String>,
    item_sparse_col: Option<String>,
    item_sparse_indices: Option<Vec<String>>,
    user_dense_col: Option<String>,
    user_dense_values: Option<String>,
    item_dense_col: Option<String>,
    item_dense_values: Option<Vec<String>>,
}

pub(crate) fn build_features(
    data: &mut HashMap<String, Value>,
    raw_feats: &Features,
    user_id: &str,
    n_items: usize,
) -> ServingResult<()> {
    let user_id = user_id.parse::<isize>()?;
    data.insert(String::from("user_indices"), json!(vec![user_id; n_items]));
    data.insert(
        String::from("item_indices"),
        json!((0..n_items).collect::<Vec<usize>>()),
    );

    // raw_feats.user_sparse_col.ok_or_else(|| error::ErrorInternalServerError("Failed to get user_sparse_col"))?,
    let sparse_key = String::from("sparse_indices");
    if let (Some(user_col), Some(item_col), Some(user_feats), Some(item_feats)) = (
        raw_feats.user_sparse_col.as_ref(),
        raw_feats.item_sparse_col.as_ref(),
        raw_feats.user_sparse_indices.as_ref(),
        raw_feats.item_sparse_indices.as_ref(),
    ) {
        let sparse_features =
            merge_features::<isize>(user_col, item_col, user_feats, item_feats, n_items)?;
        data.insert(sparse_key, sparse_features);
    } else if let (Some(user_col), Some(user_feats)) = (
        raw_feats.user_sparse_col.as_ref(),
        raw_feats.user_sparse_indices.as_ref(),
    ) {
        let user_features = construct_user_features::<isize>(user_col, user_feats, n_items)?;
        data.insert(sparse_key, user_features);
    } else if let (Some(item_col), Some(item_feats)) = (
        raw_feats.item_sparse_col.as_ref(),
        raw_feats.item_sparse_indices.as_ref(),
    ) {
        let item_features = construct_item_features::<isize>(item_col, item_feats, n_items)?;
        data.insert(sparse_key, item_features);
    }

    let dense_key = String::from("dense_values");
    if let (Some(user_col), Some(item_col), Some(user_feats), Some(item_feats)) = (
        raw_feats.user_dense_col.as_ref(),
        raw_feats.item_dense_col.as_ref(),
        raw_feats.user_dense_values.as_ref(),
        raw_feats.item_dense_values.as_ref(),
    ) {
        let dense_features =
            merge_features::<f32>(user_col, item_col, user_feats, item_feats, n_items)?;
        data.insert(dense_key, dense_features);
    } else if let (Some(user_col), Some(user_feats)) = (
        raw_feats.user_dense_col.as_ref(),
        raw_feats.user_dense_values.as_ref(),
    ) {
        let user_features = construct_user_features::<f32>(user_col, user_feats, n_items)?;
        data.insert(dense_key, user_features);
    } else if let (Some(item_col), Some(item_feats)) = (
        raw_feats.item_dense_col.as_ref(),
        raw_feats.item_dense_values.as_ref(),
    ) {
        let item_features = construct_item_features::<f32>(item_col, item_feats, n_items)?;
        data.insert(dense_key, item_features);
    }
    Ok(())
}

fn merge_features<T>(
    user_col: &str,
    item_col: &str,
    user_feats: &str,
    item_feats: &[String],
    n_items: usize,
) -> ServingResult<Value>
where
    T: Copy + Clone + Default + DeserializeOwned + Serialize,
{
    let user_col_index: Vec<usize> = serde_json::from_str(user_col)?;
    let item_col_index: Vec<usize> = serde_json::from_str(item_col)?;
    let user_values: Vec<T> = serde_json::from_str(user_feats)?;
    let item_values: Vec<Vec<T>> = item_feats
        .iter()
        .map(|i| Ok(serde_json::from_str(i))?)
        .collect::<Result<Vec<Vec<_>>, _>>()?;

    let dim = user_col_index.len() + item_col_index.len();
    let mut features = vec![vec![T::default(); dim]; n_items];
    for item_id in 0..n_items {
        for (i, v) in user_col_index.iter().zip(user_values.iter()) {
            features[item_id][*i] = *v
        }
        for (i, v) in item_col_index.iter().zip(item_values[item_id].iter()) {
            features[item_id][*i] = *v
        }
    }
    Ok(json!(features))
}

fn construct_user_features<T>(
    user_col: &str,
    user_feats: &str,
    n_items: usize,
) -> ServingResult<Value>
where
    T: Copy + Clone + Default + DeserializeOwned + Serialize,
{
    let user_col_index: Vec<usize> = serde_json::from_str(user_col)?;
    let user_values: Vec<T> = serde_json::from_str(user_feats)?;
    let mut features = vec![T::default(); user_col_index.len()];
    user_col_index
        .into_iter()
        .zip(user_values.into_iter())
        .for_each(|(i, v)| features[i] = v);
    Ok(json!(vec![features; n_items]))
}

fn construct_item_features<T>(
    item_col: &str,
    item_feats: &[String],
    n_items: usize,
) -> ServingResult<Value>
where
    T: Copy + Clone + Default + DeserializeOwned + Serialize,
{
    let item_col_index: Vec<usize> = serde_json::from_str(item_col)?;
    let item_values: Vec<Vec<T>> = item_feats
        .iter()
        .map(|i| Ok(serde_json::from_str(i))?)
        .collect::<Result<Vec<Vec<_>>, _>>()?;
    let mut features = vec![vec![T::default(); item_col_index.len()]; n_items];
    for item_id in 0..n_items {
        for (i, v) in item_col_index.iter().zip(item_values[item_id].iter()) {
            features[item_id][*i] = *v
        }
    }
    Ok(json!(features))
}

pub(crate) async fn get_raw_features(
    user_id: &str,
    conn: &mut Connection,
) -> ServingResult<Features> {
    let (user_sparse_col, user_sparse_indices) =
        get_one_feat_from_redis(conn, "user_sparse_col_index", "user_sparse_values", user_id)
            .await?;
    let (item_sparse_col, item_sparse_indices) =
        get_all_feats_from_redis(conn, "item_sparse_col_index", "item_sparse_values").await?;
    let (user_dense_col, user_dense_values) =
        get_one_feat_from_redis(conn, "user_dense_col_index", "user_dense_values", user_id).await?;
    let (item_dense_col, item_dense_values) =
        get_all_feats_from_redis(conn, "item_dense_col_index", "item_dense_values").await?;
    Ok(Features {
        user_sparse_col,
        user_sparse_indices,
        item_sparse_col,
        item_sparse_indices,
        user_dense_col,
        user_dense_values,
        item_dense_col,
        item_dense_values,
    })
}

async fn get_one_feat_from_redis(
    conn: &mut Connection,
    index_name: &str,
    value_name: &str,
    id: &str,
) -> ServingResult<(Option<String>, Option<String>)> {
    let mut index: Option<String> = None;
    let mut values: Option<String> = None;
    if conn.exists(index_name).await? {
        index.replace(get_str(conn, index_name, "none", "get").await?);
        values.replace(get_str(conn, value_name, id, "hget").await?);
    }
    Ok((index, values))
}

async fn get_all_feats_from_redis(
    conn: &mut Connection,
    index_name: &str,
    value_name: &str,
) -> ServingResult<(Option<String>, Option<Vec<String>>)> {
    let mut index: Option<String> = None;
    let mut values: Option<Vec<String>> = None;
    if conn.exists(index_name).await? {
        index.replace(get_str(conn, index_name, "none", "get").await?);
        values.replace(conn.lrange(value_name, 0, -1).await?);
    }
    Ok((index, values))
}

pub(crate) async fn get_seq_feature(
    conn: &mut Connection,
    model_name: &str,
) -> ServingResult<Option<usize>> {
    let max_seq_len: Option<usize> = if TF_SEQ_MODELS.contains(&model_name) {
        if !conn.exists("max_seq_len").await? {
            eprintln!("{} uses sequence information", model_name);
            return Err(ServingError::Other("`max_seq_len` doesn't exist in redis"));
        }
        Some(conn.get("max_seq_len").await?)
    } else {
        None
    };
    Ok(max_seq_len)
}

pub(crate) fn build_last_interaction(
    data: &mut HashMap<String, Value>,
    max_seq_len: Option<usize>,
    n_items: usize,
    user_consumed: &[usize],
) {
    if let Some(max_seq_len) = max_seq_len {
        let (u_interacted_len, dst_start, src_start) = if max_seq_len <= user_consumed.len() {
            (max_seq_len, 0, user_consumed.len() - max_seq_len)
        } else {
            (user_consumed.len(), max_seq_len - user_consumed.len(), 0)
        };
        let mut u_last_interacted = vec![n_items; max_seq_len];
        u_last_interacted[dst_start..].copy_from_slice(&user_consumed[src_start..]);
        data.insert(
            String::from("user_interacted_seq"),
            json!(vec![u_last_interacted; n_items]),
        );
        data.insert(
            String::from("user_interacted_len"),
            json!(vec![u_interacted_len; n_items]),
        );
    }
}
