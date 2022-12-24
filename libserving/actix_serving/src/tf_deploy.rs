use std::collections::{HashMap, HashSet};

use actix_web::{error, post, web, Responder};
use log::info;
use redis::AsyncCommands;
use serde_json::{json, Value};

use crate::common::{Param, Prediction, Recommendation};
use crate::features::{build_features, build_last_interaction, get_raw_features, get_seq_feature};
use crate::redis_ops::{check_exists, get_multi_str, get_str, get_vec};

#[post("/tf/recommend")]
pub async fn tf_serving(
    param: web::Json<Param>,
    redis: web::Data<redis::Client>,
) -> actix_web::Result<impl Responder> {
    let Param { user, n_rec } = param.0;
    let mut conn = redis.get_async_connection().await.map_err(|e| {
        error::ErrorInternalServerError(format!("Failed to connect to redis: {}", e))
    })?;
    info!("recommend {n_rec} items for user {user}");

    if let Err(e) = check_exists(&mut conn, "user2id", &user, "hget").await {
        return Err(error::ErrorBadRequest(e));
    }
    let user_id = get_str(&mut conn, "user2id", &user, "hget").await?;
    let model_name = get_str(&mut conn, "model_name", "none", "get").await?;
    let n_items: usize = conn
        .get("n_items")
        .await
        .map_err(error::ErrorInternalServerError)?;
    let user_consumed: Vec<usize> = get_vec(&mut conn, "user_consumed", &user).await?;
    let raw_feats = get_raw_features(&user_id, &mut conn).await?;
    let mut all_data: HashMap<String, Value> = HashMap::new();

    build_features(&mut all_data, &raw_feats, &user_id, n_items)?;
    let max_seq_len = get_seq_feature(&mut conn, &model_name).await?;
    build_last_interaction(&mut all_data, max_seq_len, n_items, &user_consumed)?;
    let scores = request_tf_serving(&all_data, &model_name)
        .await
        .map_err(|e| {
            error::ErrorInternalServerError(format!(
                "Failed to request from tensorflow serving: {}",
                e
            ))
        })?;
    let item_ids = rank_items_by_score(&scores, n_rec, &user_consumed)?;
    let recs = get_multi_str(&mut conn, "id2item", &item_ids).await?;
    Ok(web::Json(Recommendation { rec_list: recs }))
}

async fn request_tf_serving(
    data: &HashMap<String, Value>,
    model_name: &str,
) -> Result<Vec<f32>, reqwest::Error> {
    let host = std::env::var("TF_SERVING_HOST").unwrap_or_else(|_| String::from("127.0.0.1"));
    let url = format!(
        "http://{}:8501/v1/models/{}:predict",
        host,
        model_name.to_lowercase()
    );
    let req = json!({
        "signature_name": "predict",
        "inputs": data,
    });
    let resp = reqwest::Client::new()
        .post(url)
        .json(&req)
        .send()
        .await?
        .json::<Prediction>()
        .await?;
    Ok(resp.outputs)
}

fn rank_items_by_score(
    scores: &[f32],
    n_rec: usize,
    user_consumed: &[usize],
) -> actix_web::Result<Vec<usize>> {
    let user_consumed_set: HashSet<&usize> = HashSet::from_iter(user_consumed);
    let mut rank_items = scores.iter().enumerate().collect::<Vec<(usize, &f32)>>();
    rank_items.sort_unstable_by(|&(_, a), &(_, b)| a.partial_cmp(b).unwrap().reverse());
    let item_ids = rank_items
        .into_iter()
        .filter_map(|(i, _)| {
            if !user_consumed_set.contains(&i) {
                Some(i)
            } else {
                None
            }
        })
        .take(n_rec)
        .collect::<Vec<_>>();
    Ok(item_ids)
}
