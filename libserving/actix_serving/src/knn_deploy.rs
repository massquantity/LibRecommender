use actix_web::{post, web, Responder};
use deadpool_redis::{Connection, Pool};
use fnv::{FnvHashMap, FnvHashSet};

use crate::common::{Payload, Recommendation};
use crate::errors::{ServingError, ServingResult};
use crate::redis_ops::{check_exists, get_multi_str, get_str, get_vec};

#[post("/knn/recommend")]
pub async fn knn_serving(
    param: web::Json<Payload>,
    redis_pool: web::Data<Pool>,
) -> ServingResult<impl Responder> {
    let Payload { user, n_rec } = param.0;
    let mut conn = redis_pool.get().await?;
    log::info!("recommend {n_rec} items for user {user}");

    if (check_exists(&mut conn, "user2id", &user, "hget").await).is_err() {
        return Err(ServingError::NotExist("request user"));
    }
    if (check_exists(&mut conn, "model_name", "none", "get").await).is_err() {
        return Err(ServingError::NotExist("model_name"));
    }
    let user_id = get_str(&mut conn, "user2id", &user, "hget").await?;
    let model_name = get_str(&mut conn, "model_name", "none", "get").await?;
    let consumed_list: Vec<usize> = get_vec(&mut conn, "user_consumed", &user).await?;
    let user_consumed: FnvHashSet<usize> = FnvHashSet::from_iter(consumed_list);

    let recs = match model_name.as_str() {
        "UserCF" => rec_on_user_sims(&user_id, n_rec, &user_consumed, &mut conn).await?,
        "ItemCF" => rec_on_item_sims(n_rec, &user_consumed, &mut conn).await?,
        m => return Err(ServingError::UnknownModel(m.to_string())),
    };
    Ok(web::Json(Recommendation { rec_list: recs }))
}

async fn rec_on_user_sims(
    user_id: &str,
    n_rec: usize,
    user_consumed: &FnvHashSet<usize>,
    conn: &mut Connection,
) -> ServingResult<Vec<String>> {
    let mut id_sim_map: FnvHashMap<usize, f32> = FnvHashMap::default();
    let k_sim_str = get_str(conn, "k_sims", user_id, "hget").await?;
    let k_sim_users: Vec<(usize, f32)> = serde_json::from_str(&k_sim_str)?;
    for (v, sim) in k_sim_users {
        let v_consumed: Vec<usize> = get_vec(conn, "user_consumed", &v.to_string()).await?;
        for i in v_consumed {
            if user_consumed.contains(&i) {
                continue;
            }
            id_sim_map
                .entry(i)
                .and_modify(|s| *s += sim)
                .or_insert(sim);
        }
    }
    let item_ids = sort_by_sims(&id_sim_map, n_rec);
    get_multi_str(conn, "id2item", &item_ids)
        .await
        .map_err(ServingError::RedisError)
}

async fn rec_on_item_sims(
    n_rec: usize,
    user_consumed: &FnvHashSet<usize>,
    conn: &mut Connection,
) -> ServingResult<Vec<String>> {
    let mut id_sim_map: FnvHashMap<usize, f32> = FnvHashMap::default();
    for i in user_consumed {
        let k_sim_str = get_str(conn, "k_sims", &i.to_string(), "hget").await?;
        let k_sim_items: Vec<(usize, f32)> = serde_json::from_str(&k_sim_str)?;
        for (j, sim) in k_sim_items {
            if user_consumed.contains(&j) {
                continue;
            }
            id_sim_map
                .entry(j)
                .and_modify(|s| *s += sim)
                .or_insert(sim);
        }
    }
    let item_ids = sort_by_sims(&id_sim_map, n_rec);
    get_multi_str(conn, "id2item", &item_ids)
        .await
        .map_err(ServingError::RedisError)
}

fn sort_by_sims(map: &FnvHashMap<usize, f32>, n_rec: usize) -> Vec<usize> {
    let mut id_sims = map.iter().collect::<Vec<(_, _)>>();
    id_sims.sort_unstable_by(|&(_, a), &(_, b)| a.partial_cmp(b).unwrap().reverse());
    id_sims
        .into_iter()
        .take(n_rec)
        .map(|(i, _)| *i)
        .collect::<Vec<_>>()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::redis_ops::create_redis_pool;
    use actix_web::http::{header::ContentType, StatusCode};
    use actix_web::{dev::Service, middleware::Logger, test, App};
    use pretty_assertions::assert_eq;

    #[actix_web::test]
    async fn test_knn_serving() -> Result<(), Box<dyn std::error::Error>> {
        std::env::set_var("RUST_LOG", "debug");
        env_logger::init();
        let logger = Logger::default();
        let redis_pool = create_redis_pool(String::from("localhost"))?;
        let app = test::init_service(
            App::new()
                .wrap(logger)
                .app_data(web::Data::new(redis_pool))
                .service(knn_serving),
        )
        .await;

        let payload_1_rec = r#"{"user":"10","n_rec":1}"#.as_bytes();
        let req = test::TestRequest::post()
            .uri("/knn/recommend")
            .insert_header(ContentType::json())
            .set_payload(payload_1_rec)
            .to_request();
        let resp = test::try_call_service(&app, req).await?;
        assert_eq!(resp.status(), StatusCode::OK);
        let body: Recommendation = test::try_read_body_json(resp).await?;
        assert_eq!(body.rec_list.len(), 1);

        let payload_10_rec = r#"{"user":"10","n_rec":10}"#.as_bytes();
        let req = test::TestRequest::post()
            .uri("/knn/recommend")
            .insert_header(ContentType::json())
            .set_payload(payload_10_rec)
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
        let app = test::init_service(
            App::new()
                .app_data(web::Data::new(redis_pool))
                .service(knn_serving),
        )
        .await;

        let payload = r#"{"user":10,"n_rec":1}"#.as_bytes(); // user not String
        let req = test::TestRequest::post()
            .uri("/knn/recommend")
            .insert_header(ContentType::json())
            .set_payload(payload)
            .to_request();
        let resp = app.call(req).await.unwrap();
        assert_eq!(resp.status(), StatusCode::BAD_REQUEST);

        let req = test::TestRequest::get() // not post
            .uri("/knn/recommend")
            .to_request();
        let resp = app.call(req).await.unwrap();
        assert_eq!(resp.status(), StatusCode::NOT_FOUND);
    }
}
