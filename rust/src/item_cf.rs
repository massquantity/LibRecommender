use fxhash::{FxHashMap, FxHashSet};
use pyo3::prelude::*;
use pyo3::types::*;
use serde::{Deserialize, Serialize};

use crate::inference::{compute_pred, get_intersect_neighbors, get_rec_items};
use crate::similarities::{compute_sum_squares, forward_cosine, invert_cosine, sort_by_sims};
use crate::sparse::{get_row, CsrMatrix};
use crate::utils::CumValues;

#[pyclass(module = "recfarm", name = "ItemCF")]
#[derive(Serialize, Deserialize)]
pub struct PyItemCF {
    task: String,
    k_sim: usize,
    n_users: usize,
    n_items: usize,
    min_common: usize,
    sum_squares: Vec<f32>,
    cum_values: FxHashMap<i32, CumValues>,
    sim_mapping: FxHashMap<i32, (Vec<i32>, Vec<f32>)>,
    user_interactions: CsrMatrix<i32, f32>,
    item_interactions: CsrMatrix<i32, f32>,
    user_consumed: FxHashMap<i32, Vec<i32>>,
    default_pred: f32,
}

#[pymethods]
impl PyItemCF {
    #[setter]
    fn set_n_users(&mut self, n_users: usize) {
        self.n_users = n_users;
    }

    #[setter]
    fn set_n_items(&mut self, n_items: usize) {
        self.n_items = n_items;
    }

    #[setter]
    fn set_user_consumed(&mut self, user_consumed: &PyDict) -> PyResult<()> {
        self.user_consumed = user_consumed.extract::<FxHashMap<i32, Vec<i32>>>()?;
        Ok(())
    }

    #[new]
    fn new(
        task: &str,
        k_sim: usize,
        n_users: usize,
        n_items: usize,
        min_common: usize,
        user_interactions: &PyAny,
        item_interactions: &PyAny,
        user_consumed: &PyDict,
        default_pred: f32,
    ) -> PyResult<Self> {
        let user_consumed: FxHashMap<i32, Vec<i32>> = user_consumed.extract()?;
        let user_interactions: CsrMatrix<i32, f32> = user_interactions.extract()?;
        let item_interactions: CsrMatrix<i32, f32> = item_interactions.extract()?;
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

    /// forward index: sparse matrix of `item` interactions
    /// invert index: sparse matrix of `user` interactions
    fn compute_similarities(&mut self, invert: bool, num_threads: usize) -> PyResult<()> {
        self.sum_squares = compute_sum_squares(&self.item_interactions, self.n_items);
        let cosine_sims = if invert {
            invert_cosine(
                &self.user_interactions,
                &self.sum_squares,
                &mut self.cum_values,
                self.n_items,
                self.n_users,
                self.min_common,
            )?
        } else {
            std::env::set_var("RAYON_NUM_THREADS", format!("{num_threads}"));
            // rayon::ThreadPoolBuilder::new().num_threads(num_threads).build_global().unwrap();
            forward_cosine(
                &self.item_interactions,
                &self.sum_squares,
                &mut self.cum_values,
                self.n_items,
                self.min_common,
            )?
        };
        sort_by_sims(self.n_items, &cosine_sims, &mut self.sim_mapping)?;
        Ok(())
    }

    fn num_sim_elements(&self) -> PyResult<usize> {
        if self.sim_mapping.is_empty() {
            return Err(pyo3::exceptions::PyValueError::new_err(
                "call `compute_similarities` first",
            ));
        }
        let n_elements = self
            .sim_mapping
            .iter()
            .map(|(_, i)| i.0.len())
            .sum();
        Ok(n_elements)
    }

    /// sparse matrix of `user` interactions
    fn predict(&self, users: &PyList, items: &PyList) -> PyResult<Vec<f32>> {
        let mut preds = Vec::new();
        let users: Vec<usize> = users.extract()?;
        let items: Vec<i32> = items.extract()?;
        for (&u, &i) in users.iter().zip(items.iter()) {
            if u == self.n_users || usize::try_from(i)? == self.n_items {
                preds.push(self.default_pred);
                continue;
            }
            let pred = match (
                self.sim_mapping.get(&i),
                get_row(&self.user_interactions, u),
            ) {
                (Some((sim_items, sim_values)), Some(item_labels)) => {
                    let sim_num = std::cmp::min(self.k_sim, sim_items.len());
                    let mut item_sims: Vec<(i32, f32)> = sim_items[..sim_num]
                        .iter()
                        .zip(sim_values[..sim_num].iter())
                        .map(|(i, s)| (*i, *s))
                        .collect();
                    item_sims.sort_unstable_by_key(|&(i, _)| i);
                    let item_labels: Vec<(i32, f32)> = item_labels.collect();
                    let (k_nb_sims, k_nb_labels) =
                        get_intersect_neighbors(&item_sims, &item_labels, self.k_sim);
                    if k_nb_sims.is_empty() {
                        self.default_pred
                    } else {
                        compute_pred(&self.task, &k_nb_sims, &k_nb_labels)?
                    }
                }
                _ => self.default_pred,
            };
            preds.push(pred);
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
            let u: i32 = u.extract()?;
            let consumed = self
                .user_consumed
                .get(&u)
                .map_or(FxHashSet::default(), FxHashSet::from_iter);

            match get_row(&self.user_interactions, usize::try_from(u)?) {
                Some(row) => {
                    let mut item_scores: FxHashMap<i32, f32> = FxHashMap::default();
                    for (i, i_label) in row {
                        if let Some((sim_items, sim_values)) = self.sim_mapping.get(&i) {
                            let sim_num = std::cmp::min(self.k_sim, sim_items.len());
                            for (&j, &i_j_sim) in sim_items[..sim_num]
                                .iter()
                                .zip(sim_values[..sim_num].iter())
                            {
                                if filter_consumed && consumed.contains(&j) {
                                    continue;
                                }
                                item_scores
                                    .entry(j)
                                    .and_modify(|score| *score += i_j_sim * i_label)
                                    .or_insert(i_j_sim * i_label);
                            }
                        }
                    }
                    if item_scores.is_empty() {
                        recs.push(PyList::empty(py).into());
                        no_rec_indices.push(k);
                    } else {
                        let items = get_rec_items(item_scores, n_rec, random_rec);
                        recs.push(PyList::new(py, items).into());
                    }
                }
                None => {
                    recs.push(PyList::empty(py).into());
                    no_rec_indices.push(k);
                }
            }
        }

        let no_rec_indices = PyList::new(py, no_rec_indices).into_py(py);
        Ok((recs, no_rec_indices))
    }
}
