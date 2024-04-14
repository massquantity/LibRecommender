use std::cmp::Ordering;
use std::collections::BinaryHeap;

use fxhash::FxHashMap;
use pyo3::exceptions::PyValueError;
use pyo3::PyResult;
use rand::prelude::SliceRandom;

use crate::ordering::SimOrd;

pub(crate) fn get_intersect_neighbors(
    elem_sims: &[(i32, f32)],
    elem_labels: &[(i32, f32)],
    k_sim: usize,
) -> (Vec<f32>, Vec<f32>) {
    let mut i = 0;
    let mut j = 0;
    let mut max_heap: BinaryHeap<SimOrd> = BinaryHeap::new();
    while i < elem_sims.len() && j < elem_labels.len() {
        let elem1 = elem_sims[i].0;
        let elem2 = elem_labels[j].0;
        match elem1.cmp(&elem2) {
            Ordering::Less => i += 1,
            Ordering::Greater => j += 1,
            Ordering::Equal => {
                max_heap.push(SimOrd(elem_sims[i].1, elem_labels[j].1));
                i += 1;
                j += 1;
            }
        }
    }

    let mut k_neighbor_sims = Vec::new();
    let mut k_neighbor_labels = Vec::new();
    for _ in 0..k_sim {
        match max_heap.pop() {
            Some(SimOrd(sim, label)) => {
                k_neighbor_sims.push(sim);
                k_neighbor_labels.push(label);
            }
            None => break,
        }
    }
    (k_neighbor_sims, k_neighbor_labels)
}

pub(crate) fn compute_pred(
    task: &str,
    k_neighbor_sims: &[f32],
    k_neighbor_labels: &[f32],
) -> PyResult<f32> {
    let pred = match task {
        "rating" => {
            let sum_sims: f32 = k_neighbor_sims.iter().sum();
            k_neighbor_sims
                .iter()
                .zip(k_neighbor_labels.iter())
                .map(|(&sim, &label)| label * sim / sum_sims)
                .sum()
        }
        "ranking" => {
            let sum_sims: f32 = k_neighbor_sims.iter().sum();
            sum_sims / k_neighbor_sims.len() as f32
        }
        _ => {
            let err_msg = format!("Unknown task type: \"{task}\"");
            return Err(PyValueError::new_err(err_msg));
        }
    };
    Ok(pred)
}

pub(crate) fn get_rec_items(
    item_sim_scores: FxHashMap<i32, f32>,
    n_rec: usize,
    random_rec: bool,
) -> Vec<i32> {
    if random_rec && item_sim_scores.len() > n_rec {
        let mut rng = &mut rand::thread_rng();
        item_sim_scores
            .keys()
            .copied()
            .collect::<Vec<i32>>()
            .choose_multiple(&mut rng, n_rec)
            .cloned()
            .collect::<Vec<_>>()
    } else {
        let mut item_preds: Vec<(i32, f32)> = item_sim_scores.into_iter().collect();
        item_preds.sort_unstable_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap().reverse());
        item_preds
            .into_iter()
            .take(n_rec)
            .map(|(i, _)| i)
            .collect::<Vec<_>>()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_get_intersect_neighbors() {
        let elem_sims = vec![(1, 0.1), (2, 0.6), (3, 0.3), (4, 0.8), (5, 0.5)];
        let elem_labels = vec![(1, 2.0), (2, 4.0), (3, 1.0), (4, 3.0), (5, 5.0)];
        let (k_neighbor_sims, k_neighbor_labels) =
            get_intersect_neighbors(&elem_sims, &elem_labels, 2);
        assert_eq!(k_neighbor_sims, vec![0.8, 0.6]);
        assert_eq!(k_neighbor_labels, vec![3.0, 4.0]);
    }

    #[test]
    fn test_compute_pred() -> PyResult<()> {
        let k_neighbor_sims = vec![0.1, 0.2, 0.3];
        let k_neighbor_labels = vec![2.0, 4.0, 1.0];
        let pred = compute_pred("rating", &k_neighbor_sims, &k_neighbor_labels);
        assert!(pred.is_ok());
        assert!((pred? - 2.166_666_7).abs() < 1e-4);

        let pred = compute_pred("ranking", &k_neighbor_sims, &k_neighbor_labels);
        assert!(pred.is_ok());
        assert_eq!(pred?, 0.2);

        let pred = compute_pred("unknown", &k_neighbor_sims, &k_neighbor_labels);
        assert!(pred.is_err());
        pyo3::prepare_freethreaded_python();
        assert_eq!(
            pred.unwrap_err().to_string(),
            "ValueError: Unknown task type: \"unknown\""
        );
        Ok(())
    }

    #[test]
    fn test_compute_rec_items() -> PyResult<()> {
        let mut item_sim_scores = FxHashMap::default();
        item_sim_scores.insert(1, 1.0);
        item_sim_scores.insert(2, 2.0);
        item_sim_scores.insert(3, 3.0);
        item_sim_scores.insert(4, 4.0);
        item_sim_scores.insert(5, 5.0);

        (0..10).for_each({
            |_| {
                let scores = item_sim_scores.clone();
                let rec_items = get_rec_items(scores, 3, true);
                rec_items
                    .iter()
                    .all(|i| item_sim_scores.contains_key(i));
            }
        });
        let rec_items = get_rec_items(item_sim_scores, 3, false);
        assert_eq!(rec_items, vec![5, 4, 3]);
        Ok(())
    }
}
