use std::collections::HashSet;

use actix_web::{post, web, Responder};
use deadpool_redis::{redis::AsyncCommands, Pool};
use serde_json::json;
use tokio::sync::Semaphore;

use crate::common::{Payload, Prediction, Recommendation};
use crate::errors::{ServingError, ServingResult};
use crate::features::{build_cross_features, Features};
use crate::redis_ops::{check_exists, get_multi_str, get_str, get_vec};

#[derive(Clone)]
pub struct TfAppState {
    pub client: reqwest::Client,
    pub semaphore: std::sync::Arc<Semaphore>,
}

// pub fn init_tf_state(model_type: &str) -> ServingResult<Option<web::Data<TfAppState>>> {
//    match model_type {
//        "tf" => {
//            let client = reqwest::Client::new();
//            let default_limit = num_cpus::get_physical() * 4;
//            let request_limit =
//                std::env::var("REQUEST_LIMIT").map_or(Ok(default_limit), |s| s.parse::<usize>())?;
//            log::debug!("tf serving request limit: {request_limit}");
//            let semaphore = std::sync::Arc::new(Semaphore::new(request_limit));
//            Ok(Some(web::Data::new(TfAppState { client, semaphore })))
//        }
//        _ => Ok(None),
//    }
// }

pub fn init_tf_state() -> TfAppState {
    let client = reqwest::Client::new();
    let default_limit = num_cpus::get_physical() * 4;
    let request_limit = std::env::var("REQUEST_LIMIT")
        .map_or(Ok(default_limit), |s| s.parse::<usize>())
        .expect("Failed to parse env `REQUEST_LIMIT`");
    log::debug!("tf serving request limit: {request_limit}");
    let semaphore = std::sync::Arc::new(Semaphore::new(request_limit));
    TfAppState { client, semaphore }
}

#[post("/tf/recommend")]
pub async fn tf_serving(
    param: web::Json<Payload>,
    state: web::Data<TfAppState>,
    redis_pool: web::Data<Pool>,
) -> ServingResult<impl Responder> {
    let Payload { user, n_rec } = param.0;
    let mut conn = redis_pool.get().await?;
    log::info!("recommend {n_rec} items for user {user}");

    if (check_exists(&mut conn, "user2id", &user, "hget").await).is_err() {
        return Err(ServingError::NotExist("request user"));
    }
    let user_id = get_str(&mut conn, "user2id", &user, "hget").await?;
    let model_name = get_str(&mut conn, "model_name", "none", "get").await?;
    let n_items: u32 = conn.get("n_items").await?;
    let user_consumed: Vec<u32> = get_vec(&mut conn, "user_consumed", &user).await?;

    let features = build_cross_features(
        &model_name,
        &user_id,
        n_items,
        &user_consumed,
        &mut conn,
        None,
    )
    .await?;
    // let feature_json = serde_json::to_value(features)?;
    let scores = request_tf_serving(features, &model_name, state).await?;
    let item_ids = rank_items_by_score(&scores, n_rec, &user_consumed);
    let recs = get_multi_str(&mut conn, "id2item", &item_ids).await?;
    Ok(web::Json(Recommendation { rec_list: recs }))
}

async fn request_tf_serving(
    features: Features,
    model_name: &str,
    state: web::Data<TfAppState>,
) -> Result<Vec<f32>, reqwest::Error> {
    let host = std::env::var("TF_SERVING_HOST").unwrap_or_else(|_| String::from("127.0.0.1"));
    let url = format!(
        "http://{}:8501/v1/models/{}:predict",
        host,
        model_name.to_lowercase()
    );
    let req = json!({
        "signature_name": "predict",
        "inputs": features,
    });
    let permit = state.semaphore.acquire().await.unwrap();
    let resp = state.client.post(url).json(&req).send().await?;
    // early called destructor
    drop(permit);
    let preds = resp.json::<Prediction>().await?.outputs;
    Ok(preds)
}

fn rank_items_by_score(scores: &[f32], n_rec: usize, user_consumed: &[u32]) -> Vec<usize> {
    let user_consumed_set: HashSet<&u32> = HashSet::from_iter(user_consumed);
    let mut rank_items = scores
        .iter()
        .enumerate()
        .collect::<Vec<(usize, &f32)>>();
    rank_items.sort_unstable_by(|&(_, a), &(_, b)| a.partial_cmp(b).unwrap().reverse());
    rank_items
        .into_iter()
        .filter_map(|(i, _)| {
            if !user_consumed_set.contains(&(i as u32)) {
                Some(i)
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
    use crate::redis_ops::create_redis_pool;
    use actix_web::http::{header::ContentType, StatusCode};
    use actix_web::{dev::Service, middleware::Logger, test, App};
    use pretty_assertions::assert_eq;

    // #[test]
    // #[should_panic(expected = "called `Option::unwrap()` on a `None` value")]
    // async fn test_tf_state() {
    //    assert!(init_tf_state("tf").unwrap().is_some());
    //    init_tf_state("ooo").unwrap().unwrap();
    // }

    #[actix_web::test]
    async fn test_tf_serving() -> Result<(), Box<dyn std::error::Error>> {
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
                .service(tf_serving),
        )
        .await;

        let payload_1_rec = Payload {
            user: String::from("10"),
            n_rec: 1,
        };
        let req = test::TestRequest::post()
            .uri("/tf/recommend")
            .set_json(payload_1_rec)
            .to_request();
        let resp = test::try_call_service(&app, req).await?;
        assert_eq!(resp.status(), StatusCode::OK);
        let body: Recommendation = test::try_read_body_json(resp).await?;
        assert_eq!(body.rec_list.len(), 1);

        let payload_10_rec = Payload {
            user: String::from("10"),
            n_rec: 10,
        };
        let req = test::TestRequest::post()
            .uri("/tf/recommend")
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
                .service(tf_serving),
        )
        .await;

        let payload = r#"{"user":10,"n_rec":1}"#.as_bytes(); // user not String
        let req = test::TestRequest::post()
            .uri("/tf/recommend")
            .insert_header(ContentType::json())
            .set_payload(payload)
            .to_request();
        let resp = app.call(req).await.unwrap();
        assert_eq!(resp.status(), StatusCode::BAD_REQUEST);

        let req = test::TestRequest::get() // not post
            .uri("/tf/recommend")
            .to_request();
        let resp = app.call(req).await.unwrap();
        assert_eq!(resp.status(), StatusCode::NOT_FOUND);
    }
}
