use std::collections::{HashMap, HashSet};
use std::fmt::Debug;

use deadpool_redis::{redis::AsyncCommands, Connection as RedisConnection, Pool};
use serde::de::DeserializeOwned;
use tonic::{Request, Response, Status};

use crate::constants::{
    CROSS_FEAT_MODELS, CROSS_SEQ_MODELS, DENSE_REDIS_KEYS, SEPARATE_FEAT_MODELS, SEQ_EMBED_MODELS,
    SPARSE_REDIS_KEYS, SPARSE_SEQ_MODELS, USER_ID_EMBED_MODELS,
};
use crate::redis_ops::{get_multi_str, get_str, get_vec, RedisFeatKeys};

pub mod recommend_proto {
    tonic::include_proto!("recommend");
}

pub mod tensorflow_proto {
    tonic::include_proto!("tensorflow");
}

pub mod tensorflow_serving_proto {
    tonic::include_proto!("tensorflow.serving");
}

use recommend_proto::{
    feature::Value as FeatValue, recommend_server::Recommend, Feature, RecRequest, RecResponse,
};
use tensorflow_proto::{tensor_shape_proto::Dim, DataType, TensorProto, TensorShapeProto};
use tensorflow_serving_proto::prediction_service_client::PredictionServiceClient;
use tensorflow_serving_proto::{ModelSpec, PredictRequest};

pub struct RecommendService {
    pub redis_pool: Pool,
}

#[tonic::async_trait]
impl Recommend for RecommendService {
    async fn get_recommendation(
        &self,
        request: Request<RecRequest>,
    ) -> Result<Response<RecResponse>, Status> {
        let RecRequest {
            user,
            n_rec,
            user_feats,
            seq,
        } = request.into_inner();

        log::info!("recommend {n_rec} items for user {user}");
        if !user_feats.is_empty() {
            log::info!("\nuser features: {:#?}", user_feats);
        }
        if !seq.is_empty() {
            log::info!("\nsequence: {:?}", seq);
        }

        let mut conn = self
            .redis_pool
            .get()
            .await
            .map_err(|e| Status::internal(e.to_string()))?;

        let model_name: &str = &get_str(&mut conn, "model_name", "none", "get")
            .await
            .map_err(|e| Status::internal(e.to_string()))?;
        let n_items: usize = conn
            .get("n_items")
            .await
            .map_err(|e| Status::internal(e.to_string()))?;
        let user_id = if conn
            .hexists("user2id", &user)
            .await
            .map_err(|e| Status::internal(e.to_string()))?
        {
            get_str(&mut conn, "user2id", &user, "hget")
                .await
                .map_err(|e| Status::internal(e.to_string()))?
        } else {
            get_str(&mut conn, "n_users", "none", "get")
                .await
                .map_err(|e| Status::internal(e.to_string()))?
        };

        let user_consumed: Vec<i32> = if conn
            .hexists("user_consumed", &user_id)
            .await
            .map_err(|e| Status::internal(e.to_string()))?
        {
            get_vec(&mut conn, "user_consumed", &user_id)
                .await
                .map_err(|e| Status::internal(e.to_string()))?
        } else {
            Vec::new()
        };

        let user_id = user_id.parse::<i32>().unwrap();
        let candidate_num = std::cmp::min(n_rec as usize + user_consumed.len(), n_items) as i32;
        let features = if SEQ_EMBED_MODELS.contains(&model_name) {
            self.build_seq_embed_features(
                model_name,
                user_id,
                n_items as i32,
                &user_consumed,
                &mut conn,
                seq,
                candidate_num,
            )
            .await
        } else if SPARSE_SEQ_MODELS.contains(&model_name) {
            self.build_sparse_seq_features(
                user_id,
                n_items as i32,
                &user_consumed,
                &mut conn,
                user_feats,
                seq,
                candidate_num,
            )
            .await
        } else if SEPARATE_FEAT_MODELS.contains(&model_name) {
            self.build_separate_features(
                user_id,
                n_items as i32,
                &mut conn,
                user_feats,
                candidate_num,
            )
            .await
        } else {
            self.build_cross_features(
                model_name,
                user_id,
                n_items as i32,
                &user_consumed,
                &mut conn,
                user_feats,
                seq,
                candidate_num,
            )
            .await
        };

        let ranked_items = self
            .request_tf_serving(features, model_name)
            .await?;
        let item_ids = convert_items(&ranked_items, n_rec, &user_consumed);
        let recs = get_multi_str(&mut conn, "id2item", &item_ids)
            .await
            .map_err(|e| Status::internal(e.to_string()))?;
        Ok(Response::new(RecResponse { items: recs }))
    }
}

fn convert_items(ranked_items: &[i32], n_rec: i32, user_consumed: &[i32]) -> Vec<usize> {
    let user_consumed_set: HashSet<&i32> = HashSet::from_iter(user_consumed);
    ranked_items
        .iter()
        .filter_map(|i| {
            if !user_consumed_set.contains(i) {
                Some(*i as usize)
            } else {
                None
            }
        })
        .take(n_rec as usize)
        .collect::<Vec<_>>()
}

impl RecommendService {
    async fn request_tf_serving(
        &self,
        features: HashMap<String, TensorProto>,
        model_name: &str,
    ) -> Result<Vec<i32>, Status> {
        let mut grpc_client = PredictionServiceClient::connect("http://[::1]:8500")
            .await
            .map_err(|e| Status::internal(e.to_string()))?;
        let model_spec = ModelSpec {
            name: model_name.to_lowercase(),
            signature_name: String::from("topk"),
            version_choice: None,
        };

        let request = Request::new(PredictRequest {
            model_spec: Some(model_spec),
            inputs: features,
            output_filter: Vec::default(),
        });
        let response = grpc_client.predict(request).await?;
        let items = match response.into_inner().outputs.get("topk") {
            Some(output) => output.int_val.to_vec(),
            None => {
                return Err(Status::internal(
                    "Missing `topk` field in tf-serving response.",
                ))
            }
        };
        Ok(items)
    }

    async fn build_seq_embed_features(
        &self,
        model_name: &str,
        user_id: i32,
        n_items: i32,
        user_consumed: &[i32],
        conn: &mut RedisConnection,
        user_seq: Vec<i32>,
        candidate_num: i32,
    ) -> HashMap<String, TensorProto> {
        let mut features = HashMap::new();
        if USER_ID_EMBED_MODELS.contains(&model_name) {
            features.insert(
                String::from("user_indices"),
                make_int_bytes_tensor_proto(&[user_id], &[1]),
            );
        }

        let (seqs, seq_lens) = get_seq(model_name, user_consumed, n_items, conn, &user_seq).await;
        features.insert(
            String::from("user_interacted_seq"),
            make_int_bytes_tensor_proto(&seqs, &[1, seqs.len() as i64]),
        );
        features.insert(
            String::from("user_interacted_len"),
            make_int_bytes_tensor_proto::<i64>(&seq_lens, &[1]),
        );
        features.insert(
            String::from("k"),
            make_int_bytes_tensor_proto(&[candidate_num], &[]),
        );
        features
    }

    async fn build_sparse_seq_features(
        &self,
        user_id: i32,
        n_items: i32,
        user_consumed: &[i32],
        conn: &mut RedisConnection,
        user_feats: HashMap<String, Feature>,
        user_seq: Vec<i32>,
        candidate_num: i32,
    ) -> HashMap<String, TensorProto> {
        let mut features = HashMap::new();
        let (user_sparse_feats, user_dense_feats) = split_user_feats(user_feats, conn).await;
        if conn.exists("user_sparse_values").await.unwrap() {
            let user_sparse_indices =
                get_user_feats("user_sparse_values", user_id, conn, user_sparse_feats, 1).await;
            let shape = [1, user_sparse_indices.len() as i64];
            features.insert(
                String::from("user_sparse_indices"),
                make_int_bytes_tensor_proto(&user_sparse_indices, &shape),
            );
        }
        if conn.exists("user_dense_values").await.unwrap() {
            let user_dense_values =
                get_user_feats("user_dense_values", user_id, conn, user_dense_feats, 1).await;
            let shape = [1, user_dense_values.len() as i64];
            features.insert(
                String::from("user_dense_values"),
                make_float_tensor_proto(&user_dense_values, &shape),
            );
        }

        let (seq_sparse_indices, seq_sparse_values) =
            get_sparse_seq(user_consumed, n_items, conn, &user_seq).await;
        features.insert(
            String::from("item_interaction_indices"),
            make_int_bytes_tensor_proto(&seq_sparse_indices, &[seq_sparse_values.len() as i64, 2]),
        );
        features.insert(
            String::from("item_interaction_values"),
            make_int_bytes_tensor_proto(&seq_sparse_values, &[seq_sparse_values.len() as i64]),
        );
        features.insert(
            String::from("modified_batch_size"),
            make_int_bytes_tensor_proto(&[1], &[]),
        );
        features.insert(
            String::from("k"),
            make_int_bytes_tensor_proto(&[candidate_num], &[]),
        );
        features
    }

    async fn build_separate_features(
        &self,
        user_id: i32,
        n_items: i32,
        conn: &mut RedisConnection,
        user_feats: HashMap<String, Feature>,
        candidate_num: i32,
    ) -> HashMap<String, TensorProto> {
        let num = n_items as i64;
        let mut features = HashMap::new();
        features.insert(
            String::from("user_indices"),
            make_int_bytes_tensor_proto(&[user_id], &[1]),
        );
        let item_indices = (0..n_items).collect::<Vec<i32>>();
        features.insert(
            String::from("item_indices"),
            make_int_bytes_tensor_proto(&item_indices, &[num]),
        );

        let (user_sparse_feats, user_dense_feats) = split_user_feats(user_feats, conn).await;
        if conn.exists("user_sparse_values").await.unwrap() {
            let user_sparse_indices =
                get_user_feats("user_sparse_values", user_id, conn, user_sparse_feats, 1).await;
            let shape = [1, user_sparse_indices.len() as i64];
            features.insert(
                String::from("user_sparse_indices"),
                make_int_bytes_tensor_proto(&user_sparse_indices, &shape),
            );
        }
        if conn.exists("user_dense_values").await.unwrap() {
            let user_dense_values =
                get_user_feats("user_dense_values", user_id, conn, user_dense_feats, 1).await;
            let shape = [1, user_dense_values.len() as i64];
            features.insert(
                String::from("user_dense_values"),
                make_float_tensor_proto(&user_dense_values, &shape),
            );
        }

        if conn.exists("item_sparse_values").await.unwrap() {
            // flattened to 1d array
            let item_sparse_indices = get_item_feats::<i32>("item_sparse_values", num, conn).await;
            let shape = [num, (item_sparse_indices.len() as i64) / num];
            features.insert(
                String::from("item_sparse_indices"),
                make_int_bytes_tensor_proto(&item_sparse_indices, &shape),
            );
        }
        if conn.exists("item_dense_values").await.unwrap() {
            let item_dense_values = get_item_feats("item_dense_values", num, conn).await;
            let shape = [num, (item_dense_values.len() as i64) / num];
            features.insert(
                String::from("item_dense_values"),
                make_float_tensor_proto(&item_dense_values, &shape),
            );
        }
        features.insert(
            String::from("k"),
            make_int_bytes_tensor_proto(&[candidate_num], &[]),
        );
        features
    }

    async fn build_cross_features(
        &self,
        model_name: &str,
        user_id: i32,
        n_items: i32,
        user_consumed: &[i32],
        conn: &mut RedisConnection,
        user_feats: HashMap<String, Feature>,
        user_seq: Vec<i32>,
        candidate_num: i32,
    ) -> HashMap<String, TensorProto> {
        let num = n_items as i64;
        let mut features = HashMap::new();
        features.insert(
            String::from("user_indices"),
            make_int_bytes_tensor_proto(&vec![user_id; n_items as usize], &[num]),
        );
        let item_indices = (0..n_items).collect::<Vec<i32>>();
        features.insert(
            String::from("item_indices"),
            make_int_bytes_tensor_proto(&item_indices, &[num]),
        );

        let (user_sparse_feats, user_dense_feats) = split_user_feats(user_feats, conn).await;
        // flattened to 1d array
        let sparse_indices = combine_features::<i32>(
            SPARSE_REDIS_KEYS,
            user_id,
            n_items as usize,
            conn,
            user_sparse_feats,
        )
        .await;
        if !sparse_indices.is_empty() {
            let shape = [num, (sparse_indices.len() as i64) / num];
            features.insert(
                String::from("sparse_indices"),
                make_int_bytes_tensor_proto(&sparse_indices, &shape),
            );
        }
        let dense_values = combine_features::<f32>(
            DENSE_REDIS_KEYS,
            user_id,
            n_items as usize,
            conn,
            user_dense_feats,
        )
        .await;
        if !dense_values.is_empty() {
            let shape = [num, (dense_values.len() as i64) / num];
            features.insert(
                String::from("dense_values"),
                make_float_tensor_proto(&dense_values, &shape),
            );
        }

        if CROSS_SEQ_MODELS.contains(&model_name) {
            let (seqs, seq_lens) =
                get_seq::<f64>(model_name, user_consumed, n_items, conn, &user_seq).await;
            let seq_lens: Vec<f32> = seq_lens.into_iter().map(|i| i as f32).collect();
            features.insert(
                String::from("user_interacted_seq"),
                make_int_bytes_tensor_proto(&seqs, &[num, (seqs.len() as i64) / num]),
            );
            features.insert(
                String::from("user_interacted_len"),
                make_float_tensor_proto(&seq_lens, &[num]),
            );
        }

        features.insert(
            String::from("k"),
            make_int_bytes_tensor_proto(&[candidate_num], &[]),
        );

        // for (f, v) in &features {
        //    if v.tensor_shape.is_some() {
        //        log::info!("{}, {:?}", f, v.tensor_shape.as_ref().unwrap());
        //    }
        // }
        features
    }
}

async fn combine_features<T>(
    redis_keys: RedisFeatKeys,
    user_id: i32,
    n_items: usize,
    conn: &mut RedisConnection,
    user_feats: Vec<(usize, T)>,
) -> Vec<T>
where
    T: Copy + Clone + Debug + Default + DeserializeOwned,
{
    let has_user_feats: bool = conn.exists(redis_keys.user_index).await.unwrap();
    let has_item_feats: bool = conn.exists(redis_keys.item_index).await.unwrap();
    match (has_user_feats, has_item_feats) {
        (true, true) => get_cross_feats(redis_keys, user_id, n_items, conn, user_feats).await,
        (true, false) => {
            get_user_feats(redis_keys.user_value, user_id, conn, user_feats, n_items).await
        }
        (false, true) => get_item_feats(redis_keys.item_value, n_items as i64, conn).await,
        _ => Vec::default(),
    }
}

async fn get_cross_feats<T>(
    redis_keys: RedisFeatKeys,
    user_id: i32,
    n_items: usize,
    conn: &mut RedisConnection,
    user_feats: Vec<(usize, T)>,
) -> Vec<T>
where
    T: Copy + Clone + Debug + Default + DeserializeOwned,
{
    let user_col_index: Vec<usize> = get_index_from_redis(redis_keys.user_index, conn).await;
    let item_col_index: Vec<usize> = get_index_from_redis(redis_keys.item_index, conn).await;
    let mut user_vals: Vec<T> = get_one_feat_from_redis(redis_keys.user_value, user_id, conn).await;
    if !user_feats.is_empty() {
        // log::info!("cross user_feats: {:?}", user_feats);
        user_feats
            .into_iter()
            .for_each(|(idx, val)| user_vals[idx] = val)
    }
    let item_vals: Vec<Vec<T>> =
        get_all_feat_from_redis(redis_keys.item_value, n_items, conn).await;

    let dim = user_col_index.len() + item_col_index.len();
    let mut features = vec![vec![T::default(); dim]; n_items];
    for item_id in 0..n_items {
        for (i, v) in user_col_index.iter().zip(user_vals.iter()) {
            features[item_id][*i] = *v;
        }
        for (i, v) in item_col_index
            .iter()
            .zip(item_vals[item_id].iter())
        {
            features[item_id][*i] = *v;
        }
    }
    features.into_iter().flatten().collect()
}

async fn get_user_feats<T>(
    value_name: &str,
    user_id: i32,
    conn: &mut RedisConnection,
    user_feats: Vec<(usize, T)>,
    repeat_num: usize,
) -> Vec<T>
where
    T: Copy + Debug + DeserializeOwned,
{
    let mut user_vals: Vec<T> = get_one_feat_from_redis(value_name, user_id, conn).await;
    if !user_feats.is_empty() {
        // log::info!("user_feats: {:?}", user_feats);
        user_feats
            .into_iter()
            .for_each(|(idx, val)| user_vals[idx] = val)
    }

    if repeat_num == 1 {
        user_vals
    } else {
        let total_length = user_vals.len() * repeat_num;
        user_vals
            .iter()
            .cloned()
            .cycle()
            .take(total_length)
            .collect::<Vec<T>>()
    }
}

async fn get_item_feats<T>(value_name: &str, n_items: i64, conn: &mut RedisConnection) -> Vec<T>
where
    T: DeserializeOwned,
{
    let value_strs: Vec<String> = conn.lrange(value_name, 0, -1).await.unwrap();
    value_strs
        .into_iter()
        .take(n_items as usize)
        .flat_map(|i| serde_json::from_str::<Vec<T>>(&i).unwrap())
        .collect::<Vec<_>>()
}

async fn get_index_from_redis(index_name: &str, conn: &mut RedisConnection) -> Vec<usize> {
    let index_str = get_str(conn, index_name, "none", "get")
        .await
        .unwrap();
    serde_json::from_str(&index_str).unwrap()
}

async fn get_one_feat_from_redis<T>(
    value_name: &str,
    user_id: i32,
    conn: &mut RedisConnection,
) -> Vec<T>
where
    T: Debug + DeserializeOwned,
{
    let value_str = get_str(conn, value_name, &user_id.to_string(), "hget")
        .await
        .unwrap();
    serde_json::from_str(&value_str).unwrap()
}

async fn get_all_feat_from_redis<T>(
    value_name: &str,
    n_items: usize,
    conn: &mut RedisConnection,
) -> Vec<Vec<T>>
where
    T: DeserializeOwned,
{
    let value_strs: Vec<String> = conn.lrange(value_name, 0, -1).await.unwrap();
    value_strs
        .into_iter()
        .take(n_items)
        .map(|i| serde_json::from_str(&i).unwrap())
        .collect()
}

async fn split_user_feats(
    user_feats: HashMap<String, Feature>,
    conn: &mut RedisConnection,
) -> (Vec<(usize, i32)>, Vec<(usize, f32)>) {
    let mut user_sparse_feats: Vec<(usize, i32)> = Vec::new();
    let mut user_dense_feats: Vec<(usize, f32)> = Vec::new();
    if !user_feats.is_empty() {
        for (col, val) in user_feats.iter() {
            if let Some(sparse_idx_val) = feat_to_sparse_index(val, col, conn).await {
                user_sparse_feats.push(sparse_idx_val);
            } else if let Some(dense_idx_val) = feat_to_dense_value(val, col, conn).await {
                user_dense_feats.push(dense_idx_val);
            }
        }
    }
    (user_sparse_feats, user_dense_feats)
}

fn feat_to_str(feat: &Feature) -> Option<String> {
    match feat.value {
        Some(FeatValue::StringVal(ref val)) => Some(val.to_string()),
        Some(FeatValue::IntVal(ref val)) => Some(val.to_string()),
        _ => None,
    }
}

async fn feat_to_sparse_index(
    feat: &Feature,
    col: &str,
    conn: &mut RedisConnection,
) -> Option<(usize, i32)> {
    let mut res: Option<(usize, i32)> = None;
    if conn
        .hexists("user_sparse_fields", col)
        .await
        .unwrap()
    {
        if let Some(val) = feat_to_str(feat) {
            let mapping_key = format!("user_sparse_idx_mapping__{col}");
            if conn.hexists(&mapping_key, &val).await.unwrap() {
                let field_index = conn
                    .hget("user_sparse_fields", col)
                    .await
                    .unwrap();
                let sparse_index = conn.hget(&mapping_key, &val).await.unwrap();
                res.replace((field_index, sparse_index));
            } else {
                log::warn!("Unknown value `{val}` in sparse feature `{col}`")
            }
        } else {
            log::warn!("Failed to convert `{:?}` from sparse feature `{col}`", feat);
        }
    }
    res
}

async fn feat_to_dense_value(
    feat: &Feature,
    col: &str,
    conn: &mut RedisConnection,
) -> Option<(usize, f32)> {
    let mut res: Option<(usize, f32)> = None;
    if conn
        .hexists("user_dense_fields", col)
        .await
        .unwrap()
    {
        let field_index = conn.hget("user_dense_fields", col).await.unwrap();
        if let Some(FeatValue::FloatVal(ref dense_val)) = feat.value {
            res.replace((field_index, *dense_val));
        } else if let Some(FeatValue::IntVal(ref dense_val)) = feat.value {
            let float_val = (*dense_val) as f32;
            res.replace((field_index, float_val));
        } else {
            log::warn!("Failed to convert `{:?}` from dense feature `{col}`", feat);
        }
    }
    res
}

async fn get_sparse_seq(
    user_consumed: &[i32],
    n_items: i32,
    conn: &mut RedisConnection,
    user_seq: &[i32],
) -> (Vec<i64>, Vec<i64>) {
    let max_seq_len: usize = conn.get("max_seq_len").await.unwrap();
    let mut seq_ids = Vec::new();
    for i in user_seq {
        let i_str = i.to_string();
        if conn.hexists("item2id", &i_str).await.unwrap() {
            seq_ids.push(conn.hget("item2id", &i_str).await.unwrap())
        } else {
            seq_ids.push(n_items)
        }
    }
    let original_seq = if user_seq.is_empty() {
        user_consumed.to_vec()
    } else {
        seq_ids
    };

    let (sparse_indices, sparse_values) = if original_seq.is_empty() {
        (vec![0, 0], vec![-1])
    } else {
        let seq_len = std::cmp::min(max_seq_len, original_seq.len());
        let indices = vec![vec![0, 0]; seq_len]
            .into_iter()
            .flatten()
            .collect::<Vec<i64>>();
        let src_start = original_seq.len() - seq_len;
        let values = original_seq[src_start..]
            .iter()
            .map(|&i| if i < n_items { i as i64 } else { -1 })
            .collect();
        (indices, values)
    };

    (sparse_indices, sparse_values)
}

async fn get_seq<T: Clone + From<i32>>(
    model_name: &str,
    user_consumed: &[i32],
    n_items: i32,
    conn: &mut RedisConnection,
    user_seq: &[i32],
) -> (Vec<i32>, Vec<T>) {
    let repeat_num = if CROSS_FEAT_MODELS.contains(&model_name) {
        n_items
    } else {
        1
    };

    let max_seq_len: usize = conn.get("max_seq_len").await.unwrap();
    let mut seq_ids = Vec::new();
    for i in user_seq {
        let i_str = i.to_string();
        if conn.hexists("item2id", &i_str).await.unwrap() {
            seq_ids.push(conn.hget("item2id", &i_str).await.unwrap())
        } else {
            seq_ids.push(n_items)
        }
    }
    let original_seq = if seq_ids.is_empty() {
        user_consumed
    } else {
        seq_ids.as_slice()
    };

    let (res_seq, res_seq_len) = if !original_seq.is_empty() {
        let mut seq = vec![n_items as i32; max_seq_len];
        let seq_len = std::cmp::min(max_seq_len, original_seq.len());
        let src_start = original_seq.len() - seq_len;
        seq[0..seq_len].copy_from_slice(&original_seq[src_start..]);
        (seq, seq_len)
    } else {
        (vec![n_items; max_seq_len], max_seq_len)
    };

    let res_seq = if repeat_num == 1 {
        res_seq
    } else {
        vec![res_seq; repeat_num as usize]
            .into_iter()
            .flatten()
            .collect::<Vec<i32>>()
    };
    let res_seq_len = vec![T::from(res_seq_len as i32); repeat_num as usize];
    (res_seq, res_seq_len)
}

fn make_tensor_shape(shape: &[i64]) -> Option<TensorShapeProto> {
    if shape.is_empty() {
        None
    } else {
        let tensor_shape: Vec<Dim> = shape
            .iter()
            .enumerate()
            .map(|(i, &s)| Dim {
                size: s,
                name: format!("dim_{i}"),
            })
            .collect();

        let shape_proto = TensorShapeProto {
            dim: tensor_shape,
            unknown_rank: false,
        };
        Some(shape_proto)
    }
}

// fn make_int_tensor_proto(value: &[i32], shape: &[i64]) -> TensorProto {
//    let tensor_shape = make_tensor_shape(shape);
//    TensorProto {
//        dtype: DataType::DtInt32 as i32,
//        tensor_shape,
//        int_val: value.to_vec(),
//        ..Default::default()
//    }
// }

// fn make_int64_tensor_proto(value: &[i64], shape: &[i64]) -> TensorProto {
//    let tensor_shape = make_tensor_shape(shape);
//    TensorProto {
//        dtype: DataType::DtInt64 as i32,
//        tensor_shape,
//        int64_val: value.to_vec(),
//        ..Default::default()
//    }
// }

fn make_int_bytes_tensor_proto<T: Copy>(value: &[T], shape: &[i64]) -> TensorProto {
    let tensor_shape = make_tensor_shape(shape);
    let type_size = std::mem::size_of::<T>();
    let dtype = match type_size {
        8 => DataType::DtInt64 as i32,
        _ => DataType::DtInt32 as i32,
    };
    TensorProto {
        dtype,
        tensor_shape,
        tensor_content: get_bytes(value, type_size),
        ..Default::default()
    }
}

fn get_bytes<T: Copy>(value: &[T], type_size: usize) -> Vec<u8> {
    let mut value = value.to_vec();
    let length = value.len() * type_size;
    let capacity = value.capacity() * type_size;
    let ptr = value.as_mut_ptr() as *mut u8;
    unsafe {
        std::mem::forget(value);
        Vec::from_raw_parts(ptr, length, capacity)
    }
}

fn make_float_tensor_proto(value: &[f32], shape: &[i64]) -> TensorProto {
    let tensor_shape = make_tensor_shape(shape);
    TensorProto {
        dtype: DataType::DtFloat as i32,
        tensor_shape,
        // float_val: value.to_vec(),
        // assume little-endian
        tensor_content: value
            .iter()
            .flat_map(|&i| i.to_le_bytes())
            .collect::<Vec<u8>>(),
        ..Default::default()
    }
}
