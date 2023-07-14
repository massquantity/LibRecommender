use std::collections::HashSet;

use actix_web::{post, web, Responder};
use deadpool_redis::{redis::AsyncCommands, Pool};
use serde_json::json;

use crate::common::{RankedItems, RealtimePayload, Recommendation};
use crate::constants::{SEPARATE_FEAT_MODELS, SEQ_EMBED_MODELS, SPARSE_SEQ_MODELS};
use crate::errors::ServingResult;
use crate::features::{
    build_cross_features, build_separate_features, build_seq_embed_features,
    build_sparse_seq_features, DynFeats, Features,
};
use crate::redis_ops::{get_multi_str, get_str, get_vec};
use crate::tf_deploy::TfAppState;

#[post("/online/recommend")]
pub async fn online_serving(
    param: web::Json<RealtimePayload>,
    state: web::Data<TfAppState>,
    redis_pool: web::Data<Pool>,
) -> ServingResult<impl Responder> {
    let RealtimePayload {
        user,
        n_rec,
        user_feats,
        seq,
    } = param.0;

    let mut conn = redis_pool.get().await?;
    log::info!("recommend {n_rec} items for user {user}");

    if let Some(user_feats) = user_feats.as_ref() {
        log::info!("user features: {:#?}", user_feats);
    }
    if let Some(seq) = seq.as_ref() {
        log::info!("sequence: {:#?}", seq);
    }

    let model_name: &str = &get_str(&mut conn, "model_name", "none", "get").await?;
    let n_items: u32 = conn.get("n_items").await?;
    let user_id = if conn.hexists("user2id", &user).await? {
        get_str(&mut conn, "user2id", &user, "hget").await?
    } else {
        get_str(&mut conn, "n_users", "none", "get").await?
    };

    let user_consumed: Vec<u32> = if conn.hexists("user_consumed", &user_id).await? {
        get_vec(&mut conn, "user_consumed", &user_id).await?
    } else {
        Vec::new()
    };
    let candidate_num = std::cmp::min((n_rec + user_consumed.len()) as u32, n_items);
    let dyn_feats = DynFeats {
        user_feats,
        seq,
        candidate_num,
    };

    let features = if SEQ_EMBED_MODELS.contains(&model_name) {
        build_seq_embed_features(
            model_name,
            &user_id,
            n_items,
            &user_consumed,
            &mut conn,
            Some(&dyn_feats),
        )
        .await?
    } else if SPARSE_SEQ_MODELS.contains(&model_name) {
        build_sparse_seq_features(
            &user_id,
            n_items,
            &user_consumed,
            &mut conn,
            Some(&dyn_feats),
        )
        .await?
    } else if SEPARATE_FEAT_MODELS.contains(&model_name) {
        build_separate_features(&user_id, n_items, &mut conn, Some(&dyn_feats)).await?
    } else {
        build_cross_features(
            model_name,
            &user_id,
            n_items,
            &user_consumed,
            &mut conn,
            Some(&dyn_feats),
        )
        .await?
    };
    let ranked_items = request_tf_serving(features, model_name, state).await?;
    let item_ids = convert_items(&ranked_items, n_rec, &user_consumed);
    let recs = get_multi_str(&mut conn, "id2item", &item_ids).await?;
    Ok(web::Json(Recommendation { rec_list: recs }))
}

async fn request_tf_serving(
    features: Features,
    model_name: &str,
    state: web::Data<TfAppState>,
) -> Result<Vec<u32>, reqwest::Error> {
    let host = std::env::var("TF_SERVING_HOST").unwrap_or_else(|_| String::from("127.0.0.1"));
    let url = format!(
        "http://{}:8501/v1/models/{}:predict",
        host,
        model_name.to_lowercase()
    );
    let req = json!({
        "signature_name": "topk",
        "inputs": features,
    });
    let permit = state.semaphore.acquire().await.unwrap();
    let resp = state.client.post(url).json(&req).send().await?;
    // early called destructor
    drop(permit);
    let items = resp.json::<RankedItems>().await?.outputs;
    Ok(items)
}

fn convert_items(ranked_items: &[u32], n_rec: usize, user_consumed: &[u32]) -> Vec<usize> {
    let user_consumed_set: HashSet<&u32> = HashSet::from_iter(user_consumed);
    ranked_items
        .iter()
        .filter_map(|i| {
            if !user_consumed_set.contains(i) {
                Some(*i as usize)
            } else {
                None
            }
        })
        .take(n_rec)
        .collect::<Vec<_>>()
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::collections::HashMap;

    use actix_web::http::{header::ContentType, StatusCode};
    use actix_web::{dev::Service, middleware::Logger, test, App};
    use pretty_assertions::assert_eq;
    use serde_json::Value;

    use crate::redis_ops::create_redis_pool;
    use crate::tf_deploy::init_tf_state;

    #[actix_web::test]
    async fn test_online_serving() -> Result<(), Box<dyn std::error::Error>> {
        std::env::set_var("RUST_LOG", "debug");
        env_logger::init();
        let logger = Logger::default();
        let redis_pool = create_redis_pool(String::from("localhost"))?;
        let tf_state = init_tf_state();
        let app = test::init_service(
            App::new()
                .wrap(logger)
                .app_data(web::Data::new(redis_pool))
                .app_data(web::Data::new(tf_state))
                .service(online_serving),
        )
        .await;

        let user_feats = HashMap::from([
            (String::from("sex"), Value::from("female")),
            (String::from("age"), Value::from(12)),
        ]);
        let seq = vec![Value::from(1), Value::from("232")];

        let payload_1_rec = RealtimePayload {
            user: String::from("10"),
            n_rec: 1,
            user_feats: Some(user_feats),
            seq: Some(seq),
        };
        let req = test::TestRequest::post()
            .uri("/online/recommend")
            .set_json(payload_1_rec)
            .to_request();
        let resp = test::try_call_service(&app, req).await?;
        assert_eq!(resp.status(), StatusCode::OK);
        let body: Recommendation = test::try_read_body_json(resp).await?;
        assert_eq!(body.rec_list.len(), 1);

        let payload_10_rec = RealtimePayload {
            user: String::from("10"),
            n_rec: 10,
            user_feats: None,
            seq: None,
        };
        let req = test::TestRequest::post()
            .uri("/online/recommend")
            .set_json(payload_10_rec)
            .to_request();
        let resp = test::try_call_service(&app, req).await?;
        assert_eq!(resp.status(), StatusCode::OK);
        let body: Recommendation = test::try_read_body_json(resp).await?;
        assert_eq!(body.rec_list.len(), 10);
        Ok(())
    }

    #[actix_web::test]
    async fn test_invalid_request() {
        let redis_pool = create_redis_pool(String::from("localhost")).unwrap();
        let tf_state = init_tf_state();
        let app = test::init_service(
            App::new()
                .app_data(web::Data::new(redis_pool))
                .app_data(web::Data::new(tf_state))
                .service(online_serving),
        )
        .await;

        let payload = r#"{"user":10,"n_rec":1}"#.as_bytes(); // user not String
        let req = test::TestRequest::post()
            .uri("/online/recommend")
            .insert_header(ContentType::json())
            .set_payload(payload)
            .to_request();
        let resp = app.call(req).await.unwrap();
        assert_eq!(resp.status(), StatusCode::BAD_REQUEST);
    }
}
