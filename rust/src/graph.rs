use std::cmp::Ordering;
use std::time::Instant;

use fxhash::FxHashMap;
use pyo3::PyResult;

use crate::sparse::{get_row, CsrMatrix};

pub(crate) fn compute_n_items_by_user(
    interactions: &CsrMatrix<i32, f32>,
    num: usize,
) -> Vec<usize> {
    let mut n_items_by_user = Vec::new();
    for i in 0..num {
        if i >= interactions.n_rows() {
            n_items_by_user.push(0)
        }
        let start = interactions.indptr[i];
        let end = interactions.indptr[i + 1];
        if start == end {
            n_items_by_user.push(0)
        } else {
            n_items_by_user.push(end - start)
        }
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

fn get_row_vec(interactions: &CsrMatrix<i32, f32>, n: usize) -> Vec<usize> {
    if let Some(row) = get_row(interactions, n) {
        row.map(|(i, _)| i as usize).collect()
    } else {
        Vec::new()
    }
}

pub(crate) fn compute_swing_score(
    user_interactions: &CsrMatrix<i32, f32>,
    item_interactions: &CsrMatrix<i32, f32>,
    n_items_by_user: &[usize],
    alpha: f32,
    n_items: usize,
    swing_score_mapping: &mut FxHashMap<i32, Vec<(i32, f32)>>,
) -> PyResult<()> {
    let mut start = Instant::now();
    let mut map: FxHashMap<(usize, usize), Vec<usize>> = FxHashMap::default();
    for i in 0..n_items {
        if i % 20 == 0 {
            let duration = start.elapsed();
            println!("{} items completed in {:.4?}", i, duration);
            start = Instant::now();
        }

        let mut item_scores = vec![0.0; n_items];
        let users = get_row_vec(item_interactions, i);
        if users.is_empty() {
            continue;
        }
        let user_num = users.len();
        for j in 0..user_num {
            let u = users[j];
            let w_u = (n_items_by_user[u] as f32).sqrt().recip(); // compute sqrt first
            for &v in &users[(j + 1)..user_num] {
                if u == v {
                    continue;
                }
                let w_v = (n_items_by_user[v] as f32).sqrt().recip();
                let common_items = match map.get(&(u, v)) {
                    Some(m) => m.to_owned(),
                    None => {
                        let aa = get_intersect_items(
                            &get_row_vec(user_interactions, u),
                            &get_row_vec(user_interactions, v),
                        );
                        map.insert((u, v), aa.clone());
                        aa
                    }
                };
                //let (common_items, k) = get_intersect_items(
                //    &get_row_vec(user_interactions, u),
                //    &get_row_vec(user_interactions, v),
                //);
                let score = w_u * w_v * (alpha + common_items.len() as f32).recip();
                for j in common_items {
                    if i != j {
                        item_scores[j] += score;
                    }
                }
            }
        }

        let mut non_zero_scores: Vec<(i32, f32)> = item_scores
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
        if !non_zero_scores.is_empty() {
            non_zero_scores.sort_unstable_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap().reverse());
            swing_score_mapping.insert(i32::try_from(i)?, non_zero_scores);
        }
    }

    Ok(())
}
