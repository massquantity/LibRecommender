use std::collections::{HashMap, HashSet};

use actix_web::{post, web, Responder};
use deadpool_redis::{Connection, Pool};

use crate::common::{Param, Recommendation};
use crate::errors::{ServingError, ServingResult};
use crate::redis_ops::{check_exists, get_multi_str, get_str, get_vec};

#[post("/knn/recommend")]
pub async fn knn_serving(
    param: web::Json<Param>,
    redis_pool: web::Data<Pool>,
) -> ServingResult<impl Responder> {
    let Param { user, n_rec } = param.0;
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
    let user_consumed: HashSet<usize> = HashSet::from_iter(consumed_list);

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
    user_consumed: &HashSet<usize>,
    conn: &mut Connection,
) -> ServingResult<Vec<String>> {
    let mut id_sim_map: HashMap<usize, f32> = HashMap::new();
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
    user_consumed: &HashSet<usize>,
    conn: &mut Connection,
) -> ServingResult<Vec<String>> {
    let mut id_sim_map: HashMap<usize, f32> = HashMap::new();
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

fn sort_by_sims(map: &HashMap<usize, f32>, n_rec: usize) -> Vec<usize> {
    let mut id_sims = map.iter().collect::<Vec<(_, _)>>();
    id_sims.sort_unstable_by(|&(_, a), &(_, b)| a.partial_cmp(b).unwrap().reverse());
    id_sims
        .into_iter()
        .take(n_rec)
        .map(|(i, _)| *i)
        .collect::<Vec<_>>()
}
