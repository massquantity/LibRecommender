use std::cmp::Ordering;
use std::collections::{BinaryHeap, HashSet};

use fxhash::FxHashMap;
use pyo3::prelude::*;
use pyo3::types::*;
use rand::seq::SliceRandom;

use crate::similarities::{invert_cosine, sort_by_sims, SimOrd};
use crate::sparse::SparseMatrix;

// todo: PyUserCF
#[pyclass]
pub struct UserCF {
    task: String,
    k_sim: usize,
    n_users: usize,
    n_items: usize,
    min_common: usize,
    sum_squares: Vec<f32>,
    cum_values: FxHashMap<i32, (i32, i32, f32, usize)>,
    sim_mapping: FxHashMap<i32, (Vec<i32>, Vec<f32>)>,
    user_interactions: SparseMatrix,
    item_interactions: SparseMatrix,
    user_consumed: FxHashMap<i32, Vec<i32>>,
    default_pred: f32,
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
        user_sparse_indices: &PyList,
        user_sparse_indptr: &PyList,
        user_sparse_data: &PyList,
        item_sparse_indices: &PyList,
        item_sparse_indptr: &PyList,
        item_sparse_data: &PyList,
        user_consumed: &PyDict,
        default_pred: f32,
    ) -> PyResult<Self> {
        let user_interactions = SparseMatrix {
            indices: user_sparse_indices.extract::<Vec<i32>>()?,
            indptr: user_sparse_indptr.extract::<Vec<usize>>()?,
            data: user_sparse_data.extract::<Vec<f32>>()?,
        };
        let item_interactions = SparseMatrix {
            indices: item_sparse_indices.extract::<Vec<i32>>()?,
            indptr: item_sparse_indptr.extract::<Vec<usize>>()?,
            data: item_sparse_data.extract::<Vec<f32>>()?,
        };
        let user_consumed = user_consumed.extract::<FxHashMap<i32, Vec<i32>>>()?;
        Ok(UserCF {
            task: task.to_string(),
            k_sim,
            n_users,
            n_items,
            min_common,
            sum_squares,
            cum_values: FxHashMap::default(),
            sim_mapping: FxHashMap::default(),
            user_interactions,
            item_interactions,
            user_consumed,
            default_pred,
        })
    }

    /// invert index: sparse (indices, indptr, data) of `item` interaction
    fn compute_similarities(&mut self) -> PyResult<()> {
        let cosine_sims = invert_cosine(
            &self.item_interactions.indices,
            &self.item_interactions.indptr,
            &self.item_interactions.data,
            &self.sum_squares,
            &mut self.cum_values,
            self.n_users,
            self.n_items,
            self.min_common,
        )?;
        sort_by_sims(self.n_users, cosine_sims, &mut self.sim_mapping)?;
        Ok(())
    }

    fn num_sim_elements(&self) -> PyResult<usize> {
        let n_elements = self
            .sim_mapping
            .iter()
            .map(|(_, i)| i.0.len())
            .sum();
        Ok(n_elements)
    }

    /// sparse (indices, indptr, data) of `item` interaction
    fn predict(&self, users: &PyList, items: &PyList) -> PyResult<Vec<f32>> {
        let mut preds = Vec::new();
        let users = users.extract::<Vec<i32>>()?;
        let items = items.extract::<Vec<usize>>()?;
        for (&u, &i) in users.iter().zip(items.iter()) {
            if usize::try_from(u)? == self.n_users || i == self.n_items {
                preds.push(self.default_pred);
                continue;
            }
            if let Some((sim_users, sim_values)) = self.sim_mapping.get(&u) {
                let i_start = self.item_interactions.indptr[i];
                let i_end = self.item_interactions.indptr[i + 1];
                let mut max_heap: BinaryHeap<SimOrd> = BinaryHeap::new();
                let mut u_sims: Vec<(&i32, &f32)> = sim_users[..self.k_sim]
                    .iter()
                    .zip(sim_values[..self.k_sim].iter())
                    .collect();
                u_sims.sort_unstable_by_key(|(u, _)| *u);
                let u_labels: Vec<(&i32, &f32)> = self.item_interactions.indices[i_start..i_end]
                    .iter()
                    .zip(self.item_interactions.data[i_start..i_end].iter())
                    .collect();
                let mut i = 0;
                let mut j = 0;
                while i < u_sims.len() && j < u_labels.len() {
                    match u_sims[i].0.cmp(u_labels[j].0) {
                        Ordering::Less => i += 1,
                        Ordering::Greater => j += 1,
                        Ordering::Equal => {
                            max_heap.push(SimOrd(*u_sims[i].1, *u_labels[j].1));
                            i += 1;
                            j += 1;
                        }
                    }
                }
                if max_heap.is_empty() {
                    preds.push(self.default_pred);
                    continue;
                }

                // let mut tmp: Vec<i32> = Vec::new();
                // tmp.extend(sim_users);
                // tmp.extend(self.item_interactions.indices[i_start..i_end].iter());
                // tmp.sort_unstable();
                // tmp.dedup();

                let mut k_neighbor_sims = Vec::new();
                let mut k_neighbor_labels = Vec::new();
                for _ in 0..self.k_sim {
                    if let Some(SimOrd(sim, label)) = max_heap.pop() {
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
                        .map(|(&sim, &label)| label * sim / sum_sims)
                        .sum();
                    preds.push(pred);
                } else {
                    let sum_sims: f32 = k_neighbor_sims.iter().sum();
                    let pred = sum_sims / k_neighbor_sims.len() as f32;
                    preds.push(pred);
                }
            } else {
                preds.push(self.default_pred)
            }
        }
        Ok(preds)
    }

    /// sparse (indices, indptr, data) of `user` interaction
    fn recommend(
        &self,
        py: Python<'_>,
        users: &PyList,
        n_rec: usize,
        filter_consumed: bool,
        random_rec: bool,
    ) -> PyResult<(Vec<Py<PyList>>, Py<PyList>)> {
        let mut recs = Vec::new();
        let mut no_rec_indices = Vec::new();
        for (k, u) in users.iter().enumerate() {
            let u = u.extract::<i32>()?;
            if let Some((sim_users, sim_values)) = self.sim_mapping.get(&u) {
                let consumed: HashSet<&i32> = if let Some(consumed) = self.user_consumed.get(&u) {
                    HashSet::from_iter(consumed)
                } else {
                    HashSet::new()
                };
                let mut item_sim_scores: FxHashMap<i32, (f32, f32)> = FxHashMap::default();
                for (&v, &u_v_sim) in sim_users[..self.k_sim]
                    .iter()
                    .zip(sim_values[..self.k_sim].iter())
                {
                    let v = usize::try_from(v)?;
                    let i_start = self.user_interactions.indptr[v];
                    let i_end = self.user_interactions.indptr[v + 1];
                    for (&i, &v_i_score) in self.user_interactions.indices[i_start..i_end]
                        .iter()
                        .zip(self.user_interactions.data[i_start..i_end].iter())
                    {
                        if filter_consumed && !consumed.is_empty() && consumed.contains(&i) {
                            continue;
                        }
                        item_sim_scores
                            .entry(i)
                            .and_modify(|(sim, score)| {
                                *sim += u_v_sim;
                                *score += u_v_sim * v_i_score;
                            })
                            .or_insert((u_v_sim, v_i_score));
                    }
                }

                if item_sim_scores.is_empty() {
                    recs.push(PyList::empty(py).into());
                    no_rec_indices.push(k);
                    continue;
                }

                let items = if random_rec && item_sim_scores.len() > n_rec {
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
                        .into_iter()
                        .map(|(i, (sim, score))| (i, score / sim))
                        .collect();
                    item_preds
                        .sort_unstable_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap().reverse());
                    item_preds
                        .into_iter()
                        .take(n_rec)
                        .map(|(i, _)| i)
                        .collect::<Vec<_>>()
                };
                recs.push(PyList::new(py, items).into());
            } else {
                recs.push(PyList::empty(py).into());
                no_rec_indices.push(k);
            }
        }

        let no_rec_indices = PyList::new(py, no_rec_indices).into_py(py);
        Ok((recs, no_rec_indices))
    }
}
