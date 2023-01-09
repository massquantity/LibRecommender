use std::cell::{RefCell, RefMut};
use std::collections::HashSet;
use std::io::ErrorKind;
use std::sync::{Mutex, MutexGuard};

use actix_web::{error, post, web, Responder};
use deadpool_redis::Pool;
use faiss::index::{IndexImpl, SearchResult};
use faiss::{index_factory, Index, MetricType};
use log::info;

use crate::common::{Param, Recommendation};
use crate::faiss::load_faiss_index;
use crate::redis_ops::{check_exists, get_multi_str, get_str, get_vec};

pub struct EmbedAppState {
    faiss_index: Mutex<RefCell<IndexImpl>>,
}

pub fn init_embed_state(model_type: &str) -> std::io::Result<Option<web::Data<EmbedAppState>>> {
    let faiss_index = if model_type == "embed" {
        load_faiss_index()
    } else {
        index_factory(16, "Flat", MetricType::InnerProduct)
            .map_err(|e| std::io::Error::new(ErrorKind::Other, e.to_string()))?
    };
    if model_type == "embed" {
        Ok(Some(web::Data::new(EmbedAppState {
            faiss_index: Mutex::new(RefCell::new(faiss_index)),
        })))
    } else {
        Ok(None)
    }
}

#[post("/embed/recommend")]
pub async fn embed_serving(
    param: web::Json<Param>,
    state: web::Data<EmbedAppState>,
    redis_pool: web::Data<Pool>,
) -> actix_web::Result<impl Responder> {
    let Param { user, n_rec } = param.0;
    let mut conn = redis_pool.get().await.map_err(|e| {
        error::ErrorInternalServerError(format!("Failed to get redis pool connection: {}", e))
    })?;
    info!("recommend {n_rec} items for user {user}");

    if let Err(e) = check_exists(&mut conn, "user2id", &user, "hget").await {
        return Err(error::ErrorBadRequest(e));
    }
    let user_id = get_str(&mut conn, "user2id", &user, "hget").await?;
    let user_embed: Vec<f32> = get_vec(&mut conn, "user_embed", &user_id).await?;
    let consumed = get_str(&mut conn, "user_consumed", &user, "hget").await?;
    let item_ids = rec_on_sim_embeds(&user_embed, n_rec, &consumed, state).await?;
    let recs = get_multi_str(&mut conn, "id2item", &item_ids).await?;
    Ok(web::Json(Recommendation { rec_list: recs }))
}

async fn rec_on_sim_embeds(
    user_embed: &[f32],
    n_rec: usize,
    consumed_str: &str,
    state: web::Data<EmbedAppState>,
) -> Result<Vec<usize>, actix_web::Error> {
    let user_consumed: Vec<usize> = serde_json::from_str(consumed_str)?;
    let user_consumed_set: HashSet<usize> = HashSet::from_iter(user_consumed);
    let candidate_num = n_rec + user_consumed_set.len();
    let faiss_index: MutexGuard<RefCell<IndexImpl>> = state.faiss_index.lock().unwrap();
    let mut borrowed_index: RefMut<IndexImpl> = (*faiss_index).borrow_mut();
    if user_embed.len() != borrowed_index.d() as usize {
        return Err(error::ErrorInternalServerError(
            "`user_embed` dimension != `item_embed` dimension, \
            did u load the wrong faiss index?",
        ));
    }
    let SearchResult {
        distances: _,
        labels: item_idxs,
    } = borrowed_index
        .search(user_embed, candidate_num)
        .map_err(error::ErrorInternalServerError)?;
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
