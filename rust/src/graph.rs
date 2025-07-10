use std::cmp::Ordering;
use std::sync::Arc;

use dashmap::DashMap;
use fxhash::FxHashMap;
use pyo3::PyResult;
use rayon::prelude::*;

use crate::sparse::{get_row, CsrMatrix};

pub(crate) fn compute_n_items_by_user(
    interactions: &CsrMatrix<i32, f32>,
    num: usize,
) -> Vec<usize> {
    let mut n_items_by_user = Vec::new();
    for i in 0..num {
        let start = interactions.indptr[i];
        let end = interactions.indptr[i + 1];
        n_items_by_user.push(end - start);
    }
    n_items_by_user
}

fn get_intersect_items(u_items: &[usize], v_items: &[usize]) -> Vec<usize> {
    let mut i = 0;
    let mut j = 0;
    let mut common_items = Vec::new();
    while i < u_items.len() && j < v_items.len() {
        match u_items[i].cmp(&v_items[j]) {
            Ordering::Less => i += 1,
            Ordering::Greater => j += 1,
            Ordering::Equal => {
                common_items.push(u_items[i]);
                i += 1;
                j += 1;
            }
        }
    }
    common_items
}

/// pre compute common items among active users.
// fn pre_compute_common_items(
//    user_interactions: &CsrMatrix<i32, f32>,
//    n_items_by_user: &[usize],
//    pre_compute_ratio: f32,
// ) -> FxHashMap<usize, Vec<usize>> {
//    let start = Instant::now();
//    let cutoff = (pre_compute_ratio * n_items_by_user.len() as f32).ceil();
//    let mut sorted_users = n_items_by_user
//        .iter()
//        .enumerate()
//        .collect::<Vec<(usize, &usize)>>();
//    sorted_users.sort_unstable_by(|(_, &a), (_, &b)| a.partial_cmp(&b).unwrap().reverse());
//    let mut users: Vec<usize> = sorted_users
//        .iter()
//        .take(cutoff as usize)
//        .map(|&(u, _)| u)
//        .collect();
//    users.sort_unstable();

//    let n_users = n_items_by_user.len();
//    let pre_compute: Vec<(usize, Vec<usize>)> = (0..users.len())
//        .into_par_iter()
//        .flat_map(|i| {
//            let mut res = Vec::new();
//            let u = users[i];
//            for &v in &users[(i + 1)..users.len()] {
//                let common_items = get_intersect_items(
//                    &get_row_vec(user_interactions, u),
//                    &get_row_vec(user_interactions, v),
//                );
//                let key = u * n_users + v;
//               res.push((key, common_items));
//            }
//            res
//        })
//        .collect();

//    let duration = start.elapsed();
//    pre_compute_statistics(duration, &pre_compute, user_interactions, n_users, cutoff);
//    FxHashMap::from_iter(pre_compute)
// }

// fn pre_compute_statistics(
//    duration: Duration,
//    pre_compute: &[(usize, Vec<usize>)],
//    user_interactions: &CsrMatrix<i32, f32>,
//    n_users: usize,
//    cutoff: f32,
// ) {
//    let cache_item_num: usize = pre_compute.iter().map(|(_, v)| v.len()).sum();
//    let total_item_num: usize = (0..n_users)
//        .map(|u| get_row_vec(user_interactions, u).len())
//        .map(|u| u * u)
//        .sum();

//    println!(
//        "swing pre compute cutoff n_users: {}, cache num: {}, cache ratio: {:.2}%, time: {:.4?}",
//        cutoff,
//        cache_item_num,
//        cache_item_num as f32 / total_item_num as f32 * 100.0,
//        duration,
//    );
// }

fn get_row_vec(interactions: &CsrMatrix<i32, f32>, n: usize) -> Vec<usize> {
    if let Some(row) = get_row(interactions, n, false) {
        row.map(|(i, _)| i as usize).collect()
    } else {
        Vec::new()
    }
}

fn init_item_scores(
    target_item: i32,
    n_items: usize,
    prev_scores: &FxHashMap<i32, Vec<(i32, f32)>>,
) -> Vec<f32> {
    let mut item_scores = vec![0.0; n_items];
    if !prev_scores.is_empty() && prev_scores.contains_key(&target_item) {
        for &(i, s) in &prev_scores[&target_item] {
            item_scores[i as usize] = s;
        }
    }
    item_scores
}

fn extract_valid_scores(scores: Vec<f32>) -> Vec<(i32, f32)> {
    let mut non_zero_scores: Vec<(i32, f32)> = scores
        .into_iter()
        .enumerate()
        .filter_map(|(i, score)| {
            if score != 0.0 {
                Some((i as i32, score))
            } else {
                None
            }
        })
        .collect();
    if non_zero_scores.len() > 1 {
        non_zero_scores.sort_unstable_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap().reverse());
    }
    non_zero_scores
}

fn compute_single_swing(
    target_item: usize,
    user_interactions: &CsrMatrix<i32, f32>,
    item_interactions: &CsrMatrix<i32, f32>,
    prev_scores: &FxHashMap<i32, Vec<(i32, f32)>>,
    user_weights: &[f32],
    alpha: f32,
    n_users: usize,
    n_items: usize,
    cached_common_items: Arc<DashMap<usize, Vec<usize>>>,
    max_cache_num: usize,
) -> (i32, Vec<(i32, f32)>) {
    let target_i32 = target_item as i32;
    let users = get_row_vec(item_interactions, target_item);
    if users.is_empty() {
        let scores = prev_scores
            .get(&target_i32)
            .cloned()
            .unwrap_or_default();
        return (target_i32, scores);
    }

    let mut item_scores = init_item_scores(target_i32, n_items, prev_scores);
    for (j, &u) in users.iter().enumerate() {
        for &v in &users[(j + 1)..users.len()] {
            let key = u * n_users + v;
            let common_items = match cached_common_items.get(&key) {
                Some(items) => items.to_owned(),
                None => {
                    let items = get_intersect_items(
                        &get_row_vec(user_interactions, u),
                        &get_row_vec(user_interactions, v),
                    );
                    if cached_common_items.len() < max_cache_num {
                        cached_common_items.insert(key, items.clone());
                    }
                    items
                }
            };
            // exclude self item according to the paper
            let k = (common_items.len() - 1) as f32;
            let score = user_weights[u] * user_weights[v] * (alpha + k).recip();
            for ci in common_items {
                if ci != target_item {
                    item_scores[ci] += score;
                }
            }
        }
    }

    (target_i32, extract_valid_scores(item_scores))
}

pub(crate) fn compute_swing_scores(
    user_interactions: &CsrMatrix<i32, f32>,
    item_interactions: &CsrMatrix<i32, f32>,
    prev_mapping: &FxHashMap<i32, Vec<(i32, f32)>>,
    n_users: usize,
    n_items: usize,
    alpha: f32,
    max_cache_num: usize,
) -> PyResult<FxHashMap<i32, Vec<(i32, f32)>>> {
    let n_items_by_user = compute_n_items_by_user(user_interactions, n_users);
    let user_weights: Vec<f32> = n_items_by_user
        .iter()
        .map(|&i| (i as f32).sqrt().recip())
        .collect();
    let cached_common_items = Arc::new(DashMap::new());
    let swing_scores: Vec<(i32, Vec<(i32, f32)>)> = (0..n_items)
        .into_par_iter()
        .map(|i| {
            compute_single_swing(
                i,
                user_interactions,
                item_interactions,
                prev_mapping,
                &user_weights,
                alpha,
                n_users,
                n_items,
                Arc::clone(&cached_common_items),
                max_cache_num,
            )
        })
        .filter(|(_, s)| !s.is_empty())
        .collect();
    Ok(FxHashMap::from_iter(swing_scores))
}
