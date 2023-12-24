use fxhash::FxHashMap;
use pyo3::exceptions::PyValueError;
use pyo3::PyResult;
use rand::prelude::SliceRandom;

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

pub(crate) fn compute_rec_items(
    item_sim_scores: &FxHashMap<i32, (f32, f32)>,
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
        let mut item_preds: Vec<(i32, f32)> = item_sim_scores
            .iter()
            .map(|(&i, &(sim, score))| (i, score / sim))
            .collect();
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
        item_sim_scores.insert(1, (1.0, 1.0));
        item_sim_scores.insert(2, (2.0, 4.0));
        item_sim_scores.insert(3, (3.0, 9.0));
        item_sim_scores.insert(4, (4.0, 16.0));
        item_sim_scores.insert(5, (5.0, 25.0));

        (0..10).for_each({
            |_| {
                let rec_items = compute_rec_items(&item_sim_scores, 3, true);
                rec_items
                    .iter()
                    .all(|i| item_sim_scores.contains_key(i));
            }
        });
        let rec_items = compute_rec_items(&item_sim_scores, 3, false);
        assert_eq!(rec_items, vec![5, 4, 3]);
        Ok(())
    }
}
