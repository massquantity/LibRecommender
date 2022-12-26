use std::collections::{HashMap, HashSet};

use actix_web::{error, post, web, Responder};
use log::info;

use crate::common::{Param, Recommendation};
use crate::redis_ops::{check_exists, get_multi_str, get_str, get_vec};

#[post("/knn/recommend")]
pub async fn knn_serving(
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
    if let Err(e) = check_exists(&mut conn, "model_name", "none", "get").await {
        return Err(error::ErrorBadRequest(e));
    }
    let user_id = get_str(&mut conn, "user2id", &user, "hget").await?;
    let model_name = get_str(&mut conn, "model_name", "none", "get").await?;
    let consumed_list: Vec<usize> = get_vec(&mut conn, "user_consumed", &user).await?;
    let user_consumed: HashSet<usize> = HashSet::from_iter(consumed_list);

    let recs = match model_name.as_str() {
        "UserCF" => rec_on_user_sims(&user_id, n_rec, &user_consumed, &mut conn).await?,
        "ItemCF" => rec_on_item_sims(n_rec, &user_consumed, &mut conn).await?,
        _ => {
            return Err(format!("Unknown knn model: {}", model_name))
                .map_err(error::ErrorInternalServerError)
        }
    };
    Ok(web::Json(Recommendation { rec_list: recs }))
}

async fn rec_on_user_sims(
    user_id: &str,
    n_rec: usize,
    user_consumed: &HashSet<usize>,
    conn: &mut redis::aio::Connection,
) -> Result<Vec<String>, actix_web::Error> {
    let mut id_sim_map: HashMap<usize, f32> = HashMap::new();
    let k_sim_users: Vec<(usize, f32)> =
        serde_json::from_str(&get_str(conn, "k_sims", user_id, "hget").await?)
            .map_err(error::ErrorInternalServerError)?;

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
    get_multi_str(conn, "id2item", &item_ids).await
}

async fn rec_on_item_sims(
    n_rec: usize,
    user_consumed: &HashSet<usize>,
    conn: &mut redis::aio::Connection,
) -> Result<Vec<String>, actix_web::Error> {
    let mut id_sim_map: HashMap<usize, f32> = HashMap::new();
    for i in user_consumed {
        let k_sim_items: Vec<(usize, f32)> =
            serde_json::from_str(&get_str(conn, "k_sims", &i.to_string(), "hget").await?)
                .map_err(error::ErrorInternalServerError)?;
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
    get_multi_str(conn, "id2item", &item_ids).await
}

fn sort_by_sims(map: &HashMap<usize, f32>, n_rec: usize) -> Vec<usize> {
    let mut id_sims = map.iter().collect::<Vec<(_, _)>>();
    id_sims.sort_unstable_by(|&(_, a), &(_, b)| a.partial_cmp(b).unwrap().reverse());
    id_sims
        .into_iter()
        .take(n_rec)
        .map(|(i, _)| *i)
        .collect::<Vec<_>>()
}
