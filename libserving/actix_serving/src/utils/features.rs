use std::collections::HashMap;

use actix_web::error;
use redis::AsyncCommands;
use serde::de::DeserializeOwned;
use serde::Serialize;
use serde_json::{json, Value};

use crate::constants::TF_SEQ_MODELS;
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

pub fn build_features(
    data: &mut HashMap<String, Value>,
    raw_feats: &Features,
    user_id: &str,
    n_items: usize,
) -> actix_web::Result<()> {
    let user_id = user_id
        .parse::<isize>()
        .map_err(error::ErrorInternalServerError)?;
    data.insert(String::from("user_indices"), json!(vec![user_id; n_items]));
    data.insert(
        String::from("item_indices"),
        json!((0..n_items).collect::<Vec<usize>>()),
    );

    let sparse_key = String::from("sparse_indices");
    if raw_feats.user_sparse_col.is_some() && raw_feats.item_sparse_col.is_some() {
        data.insert(sparse_key, merge_feats::<isize>(raw_feats, n_items, true)?);
    } else if raw_feats.user_sparse_col.is_some() {
        data.insert(sparse_key, user_feats::<isize>(raw_feats, n_items, true)?);
    } else if raw_feats.item_sparse_col.is_some() {
        data.insert(sparse_key, item_feats::<isize>(raw_feats, n_items, true)?);
    }

    let dense_key = String::from("dense_values");
    if raw_feats.user_dense_col.is_some() && raw_feats.item_dense_col.is_some() {
        data.insert(dense_key, merge_feats::<f32>(raw_feats, n_items, false)?);
    } else if raw_feats.user_dense_col.is_some() {
        data.insert(dense_key, user_feats::<f32>(raw_feats, n_items, false)?);
    } else if raw_feats.item_dense_col.is_some() {
        data.insert(dense_key, item_feats::<f32>(raw_feats, n_items, false)?);
    }
    Ok(())
}

fn merge_feats<T>(raw_feats: &Features, n_items: usize, is_sparse: bool) -> actix_web::Result<Value>
where
    T: Copy + Clone + Default + DeserializeOwned + Serialize,
{
    let (user_col_index_str, item_col_index_str, user_values_str, item_values_str) =
        if is_sparse {
            (
                raw_feats.user_sparse_col.as_ref().ok_or_else(|| {
                    error::ErrorInternalServerError("Failed to get user_sparse_col")
                })?,
                raw_feats.item_sparse_col.as_ref().ok_or_else(|| {
                    error::ErrorInternalServerError("Failed to get item_sparse_col")
                })?,
                raw_feats.user_sparse_indices.as_ref().ok_or_else(|| {
                    error::ErrorInternalServerError("Failed to get user_sparse_indices")
                })?,
                raw_feats.item_sparse_indices.as_ref().ok_or_else(|| {
                    error::ErrorInternalServerError("Failed to get item_sparse_indices")
                })?,
            )
        } else {
            (
                raw_feats.user_dense_col.as_ref().ok_or_else(|| {
                    error::ErrorInternalServerError("Failed to get user_dense_col")
                })?,
                raw_feats.item_dense_col.as_ref().ok_or_else(|| {
                    error::ErrorInternalServerError("Failed to get item_dense_col")
                })?,
                raw_feats.user_dense_values.as_ref().ok_or_else(|| {
                    error::ErrorInternalServerError("Failed to get user_dense_values")
                })?,
                raw_feats.item_dense_values.as_ref().ok_or_else(|| {
                    error::ErrorInternalServerError("Failed to get item_dense_values")
                })?,
            )
        };
    let user_col_index: Vec<usize> = serde_json::from_str(user_col_index_str)?;
    let item_col_index: Vec<usize> = serde_json::from_str(item_col_index_str)?;
    let user_values: Vec<T> = serde_json::from_str(user_values_str)?;
    let item_values: Vec<Vec<T>> = item_values_str
        .iter()
        .map(|i| Ok(serde_json::from_str(i))?)
        .collect::<Result<Vec<Vec<_>>, _>>()
        .map_err(error::ErrorInternalServerError)?;

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

fn user_feats<T>(raw_feats: &Features, n_items: usize, is_sparse: bool) -> actix_web::Result<Value>
where
    T: Copy + Clone + Default + DeserializeOwned + Serialize,
{
    let (user_col_index_str, user_values_str) =
        if is_sparse {
            (
                raw_feats.user_sparse_col.as_ref().ok_or_else(|| {
                    error::ErrorInternalServerError("Failed to get user_sparse_col")
                })?,
                raw_feats.user_sparse_indices.as_ref().ok_or_else(|| {
                    error::ErrorInternalServerError("Failed to get user_sparse_indices")
                })?,
            )
        } else {
            (
                raw_feats.user_dense_col.as_ref().ok_or_else(|| {
                    error::ErrorInternalServerError("Failed to get user_dense_col")
                })?,
                raw_feats.user_dense_values.as_ref().ok_or_else(|| {
                    error::ErrorInternalServerError("Failed to get user_dense_values")
                })?,
            )
        };
    let user_col_index: Vec<usize> = serde_json::from_str(user_col_index_str)?;
    let user_values: Vec<T> = serde_json::from_str(user_values_str)?;
    let mut features = vec![T::default(); user_col_index.len()];
    user_col_index
        .into_iter()
        .zip(user_values.into_iter())
        .for_each(|(i, v)| features[i] = v);
    Ok(json!(vec![features; n_items]))
}

fn item_feats<T>(raw_feats: &Features, n_items: usize, is_sparse: bool) -> actix_web::Result<Value>
where
    T: Copy + Clone + Default + DeserializeOwned + Serialize,
{
    let (item_col_index_str, item_values_str) =
        if is_sparse {
            (
                raw_feats.item_sparse_col.as_ref().ok_or_else(|| {
                    error::ErrorInternalServerError("Failed to get item_sparse_col")
                })?,
                raw_feats.item_sparse_indices.as_ref().ok_or_else(|| {
                    error::ErrorInternalServerError("Failed to get item_sparse_indices")
                })?,
            )
        } else {
            (
                raw_feats.item_dense_col.as_ref().ok_or_else(|| {
                    error::ErrorInternalServerError("Failed to get item_dense_col")
                })?,
                raw_feats.item_dense_values.as_ref().ok_or_else(|| {
                    error::ErrorInternalServerError("Failed to get item_dense_values")
                })?,
            )
        };
    let item_col_index: Vec<usize> = serde_json::from_str(item_col_index_str)?;
    let item_values: Vec<Vec<T>> = item_values_str
        .iter()
        .map(|i| serde_json::from_str(i).unwrap())
        .collect();
    let mut features = vec![vec![T::default(); item_col_index.len()]; n_items];
    for item_id in 0..n_items {
        for (i, v) in item_col_index.iter().zip(item_values[item_id].iter()) {
            features[item_id][*i] = *v
        }
    }
    Ok(json!(features))
}

pub async fn get_raw_features(
    user_id: &str,
    conn: &mut redis::aio::Connection,
) -> Result<Features, actix_web::Error> {
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

async fn feature_exists(conn: &mut redis::aio::Connection, key: &str) -> actix_web::Result<bool> {
    conn.exists(key)
        .await
        .map_err(error::ErrorInternalServerError)
}

async fn get_one_feat_from_redis(
    conn: &mut redis::aio::Connection,
    index_name: &str,
    value_name: &str,
    id: &str,
) -> actix_web::Result<(Option<String>, Option<String>)> {
    let mut index: Option<String> = None;
    let mut values: Option<String> = None;
    if feature_exists(conn, index_name).await? {
        index.replace(get_str(conn, index_name, "none", "get").await?);
        values.replace(get_str(conn, value_name, id, "hget").await?);
    }
    Ok((index, values))
}

async fn get_all_feats_from_redis(
    conn: &mut redis::aio::Connection,
    index_name: &str,
    value_name: &str,
) -> actix_web::Result<(Option<String>, Option<Vec<String>>)> {
    let mut index: Option<String> = None;
    let mut values: Option<Vec<String>> = None;
    if feature_exists(conn, index_name).await? {
        index.replace(get_str(conn, index_name, "none", "get").await?);
        values.replace(
            conn.lrange(value_name, 0, -1)
                .await
                .map_err(error::ErrorInternalServerError)?,
        );
    }
    Ok((index, values))
}

pub async fn get_seq_feature(
    conn: &mut redis::aio::Connection,
    model_name: &str,
) -> actix_web::Result<Option<usize>> {
    let max_seq_len: Option<usize> = if TF_SEQ_MODELS.contains(&model_name) {
        if !feature_exists(conn, "max_seq_len").await? {
            return Err(error::ErrorInternalServerError(format!(
                "`max_seq_len` doesn't exist in redis, which is used in {model_name}"
            )));
        }
        let seq_len: usize = conn
            .get("max_seq_len")
            .await
            .map_err(error::ErrorInternalServerError)?;
        Some(seq_len)
    } else {
        None
    };
    Ok(max_seq_len)
}

pub fn build_last_interaction(
    data: &mut HashMap<String, Value>,
    max_seq_len: Option<usize>,
    n_items: usize,
    user_consumed: &[usize],
) -> actix_web::Result<()> {
    match max_seq_len {
        Some(max_seq_len) => {
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
        None => (),
    }
    Ok(())
}