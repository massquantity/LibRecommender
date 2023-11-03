use std::collections::{BinaryHeap, HashSet};
use std::time::Instant;

use fxhash::FxHashMap;
use pyo3::prelude::*;
use pyo3::types::*;
use rand::seq::SliceRandom;

use crate::similarities::{invert_cosine, sort_by_sims, LabelSim};

// todo: user_consumed, sparse struct, PyUserCF
#[pyclass]
pub struct UserCF {
    task: String,
    k_sim: usize,
    n_users: usize,
    n_items: usize,
    min_common: usize,
    sum_squares: Vec<f32>,
    prods: FxHashMap<(i32, i32), f32>,
    counts: FxHashMap<(i32, i32), usize>,
    sim_mapping: FxHashMap<i32, (Vec<i32>, Vec<f32>)>,
}

#[pymethods]
impl UserCF {
    #[new]
    fn new(
        task: &str,
        k_sim: usize,
        n_users: usize,
        n_items: usize,
        min_common: usize,
        sum_squares: Vec<f32>,
    ) -> Self {
        UserCF {
            task: task.to_string(),
            k_sim,
            n_users,
            n_items,
            min_common,
            sum_squares,
            prods: FxHashMap::default(),
            counts: FxHashMap::default(),
            sim_mapping: FxHashMap::default(),
        }
    }

    /// invert index: sparse (indices, indptr, data) of `item` interaction
    fn compute_similarities(&mut self, indices: Vec<i32>, indptr: Vec<usize>, data: Vec<f32>) {
        let cosine_sims = invert_cosine(
            &indices,
            &indptr,
            &data,
            &self.sum_squares,
            &mut self.prods,
            &mut self.counts,
            self.n_items,
            self.min_common,
        );
        sort_by_sims(cosine_sims, &mut self.sim_mapping)
    }

    fn num_sim_elements(&self) -> PyResult<usize> {
        Ok(self
            .sim_mapping
            .iter()
            .map(|(_, i)| i.0.len())
            .sum())
    }

    /// sparse (indices, indptr, data) of `item` interaction
    fn predict(
        &self,
        users: &PyList,
        items: &PyList,
        indices: Vec<i32>,
        indptr: Vec<usize>,
        data: Vec<f32>,
        default_pred: f32,
    ) -> PyResult<Vec<f32>> {
        // let start = Instant::now();
        let mut preds = Vec::new();
        let users = users.extract::<Vec<i32>>()?;
        let items = items.extract::<Vec<usize>>()?;
        for (u, i) in users.iter().zip(items.iter()) {
            if *u as usize == self.n_users || *i == self.n_items {
                preds.push(default_pred);
                continue;
            }
            if let Some((sim_users, sim_values)) = self.sim_mapping.get(u) {
                let i_start = indptr[*i];
                let i_end = indptr[*i + 1];
                let mut max_heap: BinaryHeap<LabelSim> = BinaryHeap::new();
                // todo: sim_users[..self.k_sim]
                for (&su, &sv) in sim_users.iter().zip(sim_values.iter()) {
                    for (&iu, &iv) in indices[i_start..i_end]
                        .iter()
                        .zip(data[i_start..i_end].iter())
                    {
                        if su == iu {
                            max_heap.push(LabelSim(sv, iv))
                        }
                    }
                }
                if max_heap.is_empty() {
                    preds.push(default_pred);
                    continue;
                }

                let mut k_neighbor_sims = Vec::new();
                let mut k_neighbor_labels = Vec::new();
                for _ in 0..self.k_sim {
                    if let Some(LabelSim(sim, label)) = max_heap.pop() {
                        k_neighbor_sims.push(sim);
                        k_neighbor_labels.push(label);
                    } else {
                        break;
                    }
                }

                if self.task == "rating" {
                    let sum_sims: f32 = k_neighbor_sims.iter().sum();
                    let pred: f32 = k_neighbor_sims
                        .iter()
                        .zip(k_neighbor_labels.iter())
                        .map(|(sim, label)| label * sim / sum_sims)
                        .sum();
                    preds.push(pred);
                } else {
                    let sum_sims: f32 = k_neighbor_sims.iter().sum();
                    let pred = sum_sims / k_neighbor_sims.len() as f32;
                    preds.push(pred);
                }
            } else {
                preds.push(default_pred)
            }
        }
        // let duration = start.elapsed();
        // println!("predict elapsed: {:?}", duration);
        Ok(preds)
    }

    /// sparse (indices, indptr, data) of `user` interaction
    fn recommend(
        &self,
        users: Vec<i32>,
        n_rec: usize,
        user_consumed: FxHashMap<i32, Vec<i32>>, // &PyDict
        filter_consumed: bool,
        random_rec: bool,
        indices: Vec<i32>,
        indptr: Vec<usize>,
        data: Vec<f32>,
        popular_items: Vec<i32>,
    ) -> Vec<Vec<i32>> {
        // return vec![vec![1; n_rec]; users.len()];
        let mut recs = Vec::new();
        for u in users {
            if let Some((sim_users, sim_values)) = self.sim_mapping.get(&u) {
                let consumed: HashSet<&i32> = if let Some(consumed) = user_consumed.get(&u) {
                    HashSet::from_iter(consumed)
                } else {
                    HashSet::new()
                };
                let mut item_sims: FxHashMap<i32, f32> = FxHashMap::default();
                let mut item_scores: FxHashMap<i32, f32> = FxHashMap::default();
                // todo: sim_users[..self.k_sim]
                for (&v, &u_v_sim) in sim_users.iter().zip(sim_values.iter()) {
                    let i_start = indptr[v as usize];
                    let i_end = indptr[v as usize + 1];
                    for (&i, &v_i_score) in indices[i_start..i_end]
                        .iter()
                        .zip(data[i_start..i_end].iter())
                    {
                        if filter_consumed && !consumed.is_empty() && consumed.contains(&i) {
                            continue;
                        }
                        item_sims
                            .entry(i)
                            .and_modify(|s| *s += u_v_sim)
                            .or_insert(u_v_sim);
                        item_scores
                            .entry(i)
                            .and_modify(|s| *s += u_v_sim * v_i_score)
                            .or_insert(v_i_score);
                    }
                }

                if item_sims.is_empty() {
                    recs.push(popular_items.to_vec());
                    continue;
                }

                let items = if random_rec && item_sims.len() > n_rec {
                    let mut rng = &mut rand::thread_rng();
                    item_sims
                        .keys()
                        .copied()
                        .collect::<Vec<i32>>()
                        .choose_multiple(&mut rng, n_rec)
                        .cloned()
                        .collect::<Vec<_>>()
                } else {
                    let mut item_preds: Vec<(i32, f32)> = item_sims
                        .into_iter()
                        .map(|(i, sim)| {
                            let score = item_scores.get(&i).unwrap() / sim;
                            (i, score)
                        })
                        .collect();
                    item_preds
                        .sort_unstable_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap().reverse());
                    item_preds
                        .into_iter()
                        .take(n_rec)
                        .map(|(i, _)| i)
                        .collect::<Vec<_>>()
                };
                recs.push(items);
            } else {
                recs.push(popular_items.to_vec());
            }
        }
        recs
    }
}
