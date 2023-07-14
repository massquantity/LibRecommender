use std::cell::{RefCell, RefMut};
use std::collections::HashSet;
use std::sync::{Mutex, MutexGuard};

use actix_web::{post, web, Responder};
use deadpool_redis::Pool;
use faiss::index::{IndexImpl, SearchResult};
use faiss::{read_index, Index};

use crate::common::{Payload, Recommendation};
use crate::errors::{ServingError, ServingResult};
use crate::faiss::find_index_path;
use crate::redis_ops::{check_exists, get_multi_str, get_str, get_vec};

pub type EmbedAppState = Mutex<RefCell<IndexImpl>>;

// pub fn init_embed_state(model_type: &str) -> ServingResult<Option<web::Data<EmbedAppState>>> {
//    match model_type {
//        "embed" => {
//            let index_path = find_index_path(None)?;
//            read_index(index_path)
//                .map_err(ServingError::FaissError)
//                .map(|index| Some(web::Data::new(Mutex::new(RefCell::new(index)))))
//        }
//        _ => Ok(None),
//    }
// }

pub fn init_emb_state() -> EmbedAppState {
    let index_path = find_index_path(None).expect("Failed to find faiss index.");
    read_index(index_path)
        .map(|index| Mutex::new(RefCell::new(index)))
        .expect("Failed to load faiss index.")
}

#[post("/embed/recommend")]
pub async fn embed_serving(
    param: web::Json<Payload>,
    state: web::Data<EmbedAppState>,
    redis_pool: web::Data<Pool>,
) -> ServingResult<impl Responder> {
    let Payload { user, n_rec } = param.0;
    let mut conn = redis_pool.get().await?;
    log::info!("recommend {n_rec} items for user {user}");

    if (check_exists(&mut conn, "user2id", &user, "hget").await).is_err() {
        return Err(ServingError::NotExist("request user"));
    }
    let user_id = get_str(&mut conn, "user2id", &user, "hget").await?;
    let user_embed: Vec<f32> = get_vec(&mut conn, "user_embed", &user_id).await?;
    let consumed = get_str(&mut conn, "user_consumed", &user, "hget").await?;
    // let item_ids = rec_on_sim_embeds(&user_embed, n_rec, &consumed, state).await?;
    let item_ids = tokio::task::spawn_blocking(move || {
        rec_on_sim_embeds(&user_embed, n_rec, &consumed, state)
    })
    .await??;
    let recs = get_multi_str(&mut conn, "id2item", &item_ids).await?;
    Ok(web::Json(Recommendation { rec_list: recs }))
}

fn rec_on_sim_embeds(
    user_embed: &[f32],
    n_rec: usize,
    consumed_str: &str,
    state: web::Data<EmbedAppState>,
) -> ServingResult<Vec<usize>> {
    let user_consumed: Vec<usize> = serde_json::from_str(consumed_str)?;
    let user_consumed_set: HashSet<usize> = HashSet::from_iter(user_consumed);
    let candidate_num = n_rec + user_consumed_set.len();
    let faiss_index: MutexGuard<RefCell<IndexImpl>> = state.lock().unwrap();
    let mut borrowed_index: RefMut<IndexImpl> = faiss_index.borrow_mut();
    if user_embed.len() != borrowed_index.d() as usize {
        return Err(ServingError::Other(
            "`user_embed` dimension != `item_embed` dimension, \
            did u load the wrong faiss index?",
        ));
    }
    let SearchResult {
        distances: _,
        labels: item_idxs,
    } = borrowed_index
        .search(user_embed, candidate_num)
        .map_err(ServingError::FaissError)?;
    // early called destructor
    drop(borrowed_index);
    drop(faiss_index);

    let item_ids = item_idxs
        .into_iter()
        .filter_map(|idx| {
            let i = idx.to_native() as usize;
            if !user_consumed_set.contains(&i) {
                Some(i)
            } else {
                None
            }
        })
        .take(n_rec)
        .collect::<Vec<usize>>();
    Ok(item_ids)
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
    // async fn test_embed_state() {
    //    assert!(init_embed_state("embed").unwrap().is_some());
    // init_embed_state("ooo").unwrap().unwrap();
    // }

    #[actix_web::test]
    async fn test_embed_serving() -> Result<(), Box<dyn std::error::Error>> {
        std::env::set_var("RUST_LOG", "debug");
        env_logger::init();
        let logger = Logger::default();
        let redis_pool = create_redis_pool(String::from("localhost"))?;
        let embed_state = init_emb_state();
        let app = test::init_service(
            App::new()
                .wrap(logger)
                .app_data(web::Data::new(redis_pool))
                .app_data(web::Data::new(embed_state))
                .service(embed_serving),
        )
        .await;

        let payload_1_rec = r#"{"user":"10","n_rec":1}"#.as_bytes();
        let req = test::TestRequest::post()
            .uri("/embed/recommend")
            .insert_header(ContentType::json())
            .set_payload(payload_1_rec)
            .to_request();
        let resp = test::try_call_service(&app, req).await?;
        assert_eq!(resp.status(), StatusCode::OK);
        let body: Recommendation = test::try_read_body_json(resp).await?;
        assert_eq!(body.rec_list.len(), 1);

        let payload_10_rec = r#"{"user":"10","n_rec":10}"#.as_bytes();
        let req = test::TestRequest::post()
            .uri("/embed/recommend")
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
        let embed_state = init_emb_state();
        let app = test::init_service(
            App::new()
                .app_data(web::Data::new(redis_pool))
                .app_data(web::Data::new(embed_state))
                .service(embed_serving),
        )
        .await;

        let payload = r#"{"user":10,"n_rec":1}"#.as_bytes(); // user not String
        let req = test::TestRequest::post()
            .uri("/embed/recommend")
            .insert_header(ContentType::json())
            .set_payload(payload)
            .to_request();
        let resp = app.call(req).await.unwrap();
        assert_eq!(resp.status(), StatusCode::BAD_REQUEST);

        let req = test::TestRequest::get() // not post
            .uri("/embed/recommend")
            .to_request();
        let resp = app.call(req).await.unwrap();
        assert_eq!(resp.status(), StatusCode::NOT_FOUND);
    }
}
