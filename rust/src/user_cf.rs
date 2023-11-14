use std::cmp::Ordering;
use std::collections::{BinaryHeap, HashSet};

use fxhash::FxHashMap;
use pyo3::prelude::*;
use pyo3::types::*;
use rand::seq::SliceRandom;

use crate::incremental::{update_by_sims, update_cosine, update_sum_squares};
use crate::similarities::{compute_sum_squares, invert_cosine, sort_by_sims, SimOrd};
use crate::sparse::{construct_csr_matrix, CsrMatrix};

#[pyclass(module = "recfarm", name = "UserCF")]
pub struct PyUserCF {
    task: String,
    k_sim: usize,
    n_users: usize,
    n_items: usize,
    min_common: usize,
    sum_squares: Vec<f32>,
    cum_values: FxHashMap<i32, (i32, i32, f32, usize)>,
    sim_mapping: FxHashMap<i32, (Vec<i32>, Vec<f32>)>,
    user_interactions: CsrMatrix,
    item_interactions: CsrMatrix,
    user_consumed: FxHashMap<i32, Vec<i32>>,
    default_pred: f32,
}

#[pymethods]
impl PyUserCF {
    #[new]
    fn new(
        task: &str,
        k_sim: usize,
        n_users: usize,
        n_items: usize,
        min_common: usize,
        user_sparse_indices: &PyList,
        user_sparse_indptr: &PyList,
        user_sparse_data: &PyList,
        item_sparse_indices: &PyList,
        item_sparse_indptr: &PyList,
        item_sparse_data: &PyList,
        user_consumed: &PyDict,
        default_pred: f32,
    ) -> PyResult<Self> {
        let user_interactions =
            construct_csr_matrix(user_sparse_indices, user_sparse_indptr, user_sparse_data)?;
        let item_interactions =
            construct_csr_matrix(item_sparse_indices, item_sparse_indptr, item_sparse_data)?;
        let user_consumed = user_consumed.extract::<FxHashMap<i32, Vec<i32>>>()?;
        Ok(Self {
            task: task.to_string(),
            k_sim,
            n_users,
            n_items,
            min_common,
            sum_squares: Vec::new(),
            cum_values: FxHashMap::default(),
            sim_mapping: FxHashMap::default(),
            user_interactions,
            item_interactions,
            user_consumed,
            default_pred,
        })
    }

    /// invert index: sparse matrix of `item` interaction
    fn compute_similarities(&mut self /* *args, **kwargs */) -> PyResult<()> {
        self.sum_squares = compute_sum_squares(&self.user_interactions, self.n_users);
        let cosine_sims = invert_cosine(
            &self.item_interactions,
            &self.sum_squares,
            &mut self.cum_values,
            self.n_users,
            self.n_items,
            self.min_common,
        )?;
        sort_by_sims(self.n_users, &cosine_sims, &mut self.sim_mapping)?;
        Ok(())
    }

    /// update on new sparse interactions
    fn update_similarities(
        &mut self,
        user_sparse_indices: &PyList,
        user_sparse_indptr: &PyList,
        user_sparse_data: &PyList,
        item_sparse_indices: &PyList,
        item_sparse_indptr: &PyList,
        item_sparse_data: &PyList,
    ) -> PyResult<()> {
        let new_user_interactions =
            construct_csr_matrix(user_sparse_indices, user_sparse_indptr, user_sparse_data)?;
        let new_item_interactions =
            construct_csr_matrix(item_sparse_indices, item_sparse_indptr, item_sparse_data)?;
        update_sum_squares(&mut self.sum_squares, &new_user_interactions, self.n_users);
        let cosine_sims = update_cosine(
            &new_item_interactions,
            &self.sum_squares,
            &mut self.cum_values,
            self.n_users,
            self.min_common,
        )?;
        update_by_sims(self.n_users, &cosine_sims, &mut self.sim_mapping)?;
        Ok(())
    }

    fn update_parameters(&mut self, n_users: usize, n_items: usize) {
        self.n_users = n_users;
        self.n_items = n_items;
    }

    fn num_sim_elements(&self) -> PyResult<usize> {
        let n_elements = self
            .sim_mapping
            .iter()
            .map(|(_, i)| i.0.len())
            .sum();
        Ok(n_elements)
    }

    /// sparse matrix of `item` interaction
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
                let mut max_heap: BinaryHeap<SimOrd> = BinaryHeap::new();
                let sim_num = std::cmp::min(self.k_sim, sim_users.len());
                let mut u_sims: Vec<(&i32, &f32)> = sim_users[..sim_num]
                    .iter()
                    .zip(sim_values[..sim_num].iter())
                    .collect();
                u_sims.sort_unstable_by_key(|(u, _)| *u);

                if let Some(u_labels) = self.item_interactions.get_row(i) {
                    let u_labels: Vec<(&i32, &f32)> = u_labels.collect();
                    let mut i = 0;
                    let mut j = 0;
                    while i < u_sims.len() && j < u_labels.len() {
                        let u1 = u_sims[i].0;
                        let u2 = u_labels[j].0;
                        match u1.cmp(u2) {
                            Ordering::Less => i += 1,
                            Ordering::Greater => j += 1,
                            Ordering::Equal => {
                                max_heap.push(SimOrd(*u_sims[i].1, *u_labels[j].1));
                                i += 1;
                                j += 1;
                            }
                        }
                    }
                }
                if max_heap.is_empty() {
                    preds.push(self.default_pred);
                    continue;
                }

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

    /// sparse matrix of `user` interaction
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
                let sim_num = std::cmp::min(self.k_sim, sim_users.len());
                for (&v, &u_v_sim) in sim_users[..sim_num]
                    .iter()
                    .zip(sim_values[..sim_num].iter())
                {
                    let v = usize::try_from(v)?;
                    if let Some(row) = self.user_interactions.get_row(v) {
                        for (&i, &v_i_score) in row {
                            if filter_consumed && !consumed.is_empty() && consumed.contains(&i) {
                                continue;
                            }
                            item_sim_scores
                                .entry(i)
                                .and_modify(|(sim, score)| {
                                    *sim += u_v_sim;
                                    *score += u_v_sim * v_i_score;
                                })
                                .or_insert((u_v_sim, u_v_sim * v_i_score));
                        }
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

    fn merge_interactions(
        &mut self,
        n_users: usize,
        n_items: usize,
        user_sparse_indices: &PyList,
        user_sparse_indptr: &PyList,
        user_sparse_data: &PyList,
        item_sparse_indices: &PyList,
        item_sparse_indptr: &PyList,
        item_sparse_data: &PyList,
    ) -> PyResult<()> {
        let new_user_interactions =
            construct_csr_matrix(user_sparse_indices, user_sparse_indptr, user_sparse_data)?;
        let new_item_interactions =
            construct_csr_matrix(item_sparse_indices, item_sparse_indptr, item_sparse_data)?;
        self.user_interactions = CsrMatrix::add(
            &self.user_interactions,
            &new_user_interactions,
            Some(n_users),
        );
        self.item_interactions = CsrMatrix::add(
            &self.item_interactions,
            &new_item_interactions,
            Some(n_items),
        );
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn get_user_cf() -> Result<PyUserCF, Box<dyn std::error::Error>> {
        let task = "ranking";
        let k_sim = 10;
        let n_users = 5;
        let n_items = 4;
        let min_common = 1;
        let default_pred = 0.0;
        let user_cf = Python::with_gil(|py| -> PyResult<PyUserCF> {
            // user_interactions:
            // [
            //     [1, 1, 0, 0],
            //     [2, 1, 0, 0],
            //     [0, 1, 1, 0],
            //     [2, 1, 1, 0],
            //     [0, 1, 2, 0],
            // ]
            let user_sparse_indices = PyList::new(py, vec![0, 1, 0, 1, 1, 2, 0, 1, 2, 1, 2]);
            let user_sparse_indptr = PyList::new(py, vec![0, 2, 4, 6, 9, 11]);
            let user_sparse_data = PyList::new(py, vec![1, 1, 2, 1, 1, 1, 2, 1, 1, 1, 2]);
            let item_sparse_indices = PyList::new(py, vec![0, 1, 3, 0, 1, 2, 3, 4, 2, 3, 4]);
            let item_sparse_indptr = PyList::new(py, vec![0, 3, 8, 11, 11]);
            let item_sparse_data = PyList::new(py, vec![1, 2, 2, 1, 1, 1, 1, 1, 1, 1, 2]);
            let user_consumed = [
                (0, vec![0, 1]),
                (1, vec![0, 1]),
                (2, vec![1, 2]),
                (3, vec![0, 1, 2]),
                (4, vec![1, 2]),
            ]
            .into_py_dict(py);

            let mut user_cf = PyUserCF::new(
                task,
                k_sim,
                n_users,
                n_items,
                min_common,
                user_sparse_indices,
                user_sparse_indptr,
                user_sparse_data,
                item_sparse_indices,
                item_sparse_indptr,
                item_sparse_data,
                user_consumed,
                default_pred,
            )?;
            user_cf.compute_similarities()?;
            Ok(user_cf)
        })?;
        Ok(user_cf)
    }

    #[test]
    fn test_user_cf_training() -> Result<(), Box<dyn std::error::Error>> {
        let get_nbs = |model: &PyUserCF, u: i32| model.sim_mapping.get(&u).cloned().unwrap().0;
        pyo3::prepare_freethreaded_python();
        let user_cf = get_user_cf()?;
        assert_eq!(get_nbs(&user_cf, 0), vec![1, 3, 2, 4]);
        assert_eq!(get_nbs(&user_cf, 1), vec![0, 3, 2, 4]);
        assert_eq!(get_nbs(&user_cf, 2), vec![4, 3, 0, 1]);
        assert_eq!(get_nbs(&user_cf, 3), vec![1, 0, 2, 4]);
        assert_eq!(get_nbs(&user_cf, 4), vec![2, 3, 0, 1]);
        Ok(())
    }

    #[test]
    fn test_user_cf_incremental_training() -> Result<(), Box<dyn std::error::Error>> {
        let new_n_users = 6;
        let new_n_items = 5;
        let get_nbs = |model: &PyUserCF, u: i32| model.sim_mapping.get(&u).cloned().unwrap().0;
        pyo3::prepare_freethreaded_python();
        let mut user_cf = get_user_cf()?;
        user_cf.update_parameters(new_n_users, new_n_items);
        Python::with_gil(|py| -> PyResult<()> {
            // user_interactions:
            // [
            //     [0, 0, 0, 0, 0],
            //     [3, 0, 0, 0, 0],
            //     [5, 0, 0, 0, 0],
            //     [0, 0, 0, 0, 0],
            //     [0, 0, 0, 0, 0],
            //     [2, 2, 1, 2, 0]
            // ]
            let user_sparse_indices = PyList::new(py, vec![0, 0, 0, 1, 2, 3]);
            let user_sparse_indptr = PyList::new(py, vec![0, 0, 1, 2, 2, 2, 6]);
            let user_sparse_data = PyList::new(py, vec![3.0, 5.0, 2.0, 2.0, 1.0, 2.0]);
            let item_sparse_indices = PyList::new(py, vec![1, 2, 5, 5, 5, 5]);
            let item_sparse_indptr = PyList::new(py, vec![0, 3, 4, 5, 6, 6]);
            let item_sparse_data = PyList::new(py, vec![3.0, 5.0, 2.0, 2.0, 1.0, 2.0]);
            let _user_consumed = [
                (0, vec![0, 1]),
                (1, vec![0, 1]),
                (2, vec![1, 2]),
                (3, vec![0, 1, 2]),
                (4, vec![1, 2]),
                (5, vec![0, 1, 2, 3]),
            ]
            .into_py_dict(py);

            user_cf.merge_interactions(
                new_n_users,
                new_n_items,
                user_sparse_indices,
                user_sparse_indptr,
                user_sparse_data,
                item_sparse_indices,
                item_sparse_indptr,
                item_sparse_data,
            )?;
            user_cf.update_similarities(
                user_sparse_indices,
                user_sparse_indptr,
                user_sparse_data,
                item_sparse_indices,
                item_sparse_indptr,
                item_sparse_data,
            )?;
            Ok(())
        })?;
        assert_eq!(get_nbs(&user_cf, 0), vec![1, 3, 2, 4]);
        assert_eq!(get_nbs(&user_cf, 1), vec![2, 0, 3, 5, 4]);
        assert_eq!(get_nbs(&user_cf, 2), vec![1, 4, 3, 5, 0]);
        assert_eq!(get_nbs(&user_cf, 3), vec![1, 0, 2, 4]);
        assert_eq!(get_nbs(&user_cf, 4), vec![2, 3, 0, 1]);
        assert_eq!(get_nbs(&user_cf, 5), vec![2, 1]);

        Python::with_gil(|py| -> PyResult<()> {
            // user_interactions:
            // [
            //     [0, 0, 0, 3, 2],
            //     [0, 0, 0, 0, 0],
            //     [0, 0, 0, 0, 0],
            //     [0, 1, 0, 4, 3],
            // ]
            let user_sparse_indices = PyList::new(py, vec![3, 4, 1, 3, 4]);
            let user_sparse_indptr = PyList::new(py, vec![0, 2, 2, 2, 5]);
            let user_sparse_data = PyList::new(py, vec![3.0, 2.0, 1.0, 4.0, 3.0]);
            let item_sparse_indices = PyList::new(py, vec![3, 0, 3, 0, 3]);
            let item_sparse_indptr = PyList::new(py, vec![0, 0, 1, 1, 3, 5]);
            let item_sparse_data = PyList::new(py, vec![1.0, 3.0, 4.0, 2.0, 3.0]);
            user_cf.merge_interactions(
                new_n_users,
                new_n_items,
                user_sparse_indices,
                user_sparse_indptr,
                user_sparse_data,
                item_sparse_indices,
                item_sparse_indptr,
                item_sparse_data,
            )?;
            user_cf.update_similarities(
                user_sparse_indices,
                user_sparse_indptr,
                user_sparse_data,
                item_sparse_indices,
                item_sparse_indptr,
                item_sparse_data,
            )?;
            Ok(())
        })?;
        assert_eq!(get_nbs(&user_cf, 0), vec![3, 1, 2, 4]);
        assert_eq!(get_nbs(&user_cf, 1), vec![2, 0, 3, 5, 4]);
        assert_eq!(get_nbs(&user_cf, 2), vec![1, 4, 3, 5, 0]);
        assert_eq!(get_nbs(&user_cf, 3), vec![0, 1, 2, 4]);
        assert_eq!(get_nbs(&user_cf, 4), vec![2, 3, 0, 1]);
        assert_eq!(get_nbs(&user_cf, 5), vec![2, 1]);
        Ok(())
    }
}
