use fxhash::{FxHashMap, FxHashSet};
use pyo3::prelude::*;
use pyo3::types::*;
use serde::{Deserialize, Serialize};

use crate::graph::compute_swing_scores;
use crate::inference::{compute_pred, get_intersect_neighbors, get_rec_items};
use crate::sparse::{get_row, CsrMatrix};

#[pyclass(module = "recfarm", name = "Swing")]
#[derive(Serialize, Deserialize)]
pub struct PySwing {
    task: String,
    top_k: usize,
    alpha: f32,
    pre_compute_ratio: f32,
    n_users: usize,
    n_items: usize,
    cum_swings: FxHashMap<i32, f32>,
    swing_score_mapping: FxHashMap<i32, Vec<(i32, f32)>>,
    user_interactions: CsrMatrix<i32, f32>,
    item_interactions: CsrMatrix<i32, f32>,
    user_consumed: FxHashMap<i32, Vec<i32>>,
    default_pred: f32,
}

#[pymethods]
impl PySwing {
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
        top_k: usize,
        alpha: f32,
        pre_compute_ratio: f32,
        n_users: usize,
        n_items: usize,
        user_interactions: &PyAny,
        item_interactions: &PyAny,
        user_consumed: &PyDict,
        default_pred: f32,
    ) -> PyResult<Self> {
        let user_consumed: FxHashMap<i32, Vec<i32>> = user_consumed.extract()?;
        let user_interactions: CsrMatrix<i32, f32> = user_interactions.extract()?;
        let item_interactions: CsrMatrix<i32, f32> = item_interactions.extract()?;
        Ok(Self {
            task: task.to_owned(),
            top_k,
            alpha,
            pre_compute_ratio,
            n_users,
            n_items,
            cum_swings: FxHashMap::default(),
            swing_score_mapping: FxHashMap::default(),
            user_interactions,
            item_interactions,
            user_consumed,
            default_pred,
        })
    }

    fn compute_swing(&mut self, num_threads: usize) -> PyResult<()> {
        std::env::set_var("RAYON_NUM_THREADS", format!("{num_threads}"));
        self.swing_score_mapping = compute_swing_scores(
            &self.user_interactions,
            &self.item_interactions,
            self.n_users,
            self.n_items,
            self.alpha,
            self.pre_compute_ratio,
        )?;
        Ok(())
    }

    fn num_swing_elements(&self) -> PyResult<usize> {
        if self.swing_score_mapping.is_empty() {
            return Err(pyo3::exceptions::PyRuntimeError::new_err(
                "call `compute_swing` method before calling `num_swing_elments`",
            ));
        }
        let n_elements = self
            .swing_score_mapping
            .iter()
            .map(|(_, i)| i.len())
            .sum();
        Ok(n_elements)
    }

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
                self.swing_score_mapping.get(&i),
                get_row(&self.user_interactions, u),
            ) {
                (Some(item_swings), Some(item_labels)) => {
                    let num = std::cmp::min(self.top_k, item_swings.len());
                    let mut item_swing_scores = vec![(0, 0.0); num];
                    item_swing_scores.clone_from_slice(&item_swings[..num]);
                    item_swing_scores.sort_unstable_by_key(|&(i, _)| i);
                    let item_labels: Vec<(i32, f32)> = item_labels.collect();
                    let (k_nb_swings, k_nb_labels) =
                        get_intersect_neighbors(&item_swing_scores, &item_labels, self.top_k);
                    if k_nb_swings.is_empty() {
                        self.default_pred
                    } else {
                        compute_pred(&self.task, &k_nb_swings, &k_nb_labels)?
                    }
                }
                _ => self.default_pred,
            };
            preds.push(pred);
        }
        Ok(preds)
    }

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
                        if let Some(item_swings) = self.swing_score_mapping.get(&i) {
                            let num = std::cmp::min(self.top_k, item_swings.len());
                            for &(j, i_j_swing_score) in &item_swings[..num] {
                                if filter_consumed && consumed.contains(&j) {
                                    continue;
                                }
                                item_scores
                                    .entry(j)
                                    .and_modify(|score| *score += i_j_swing_score * i_label)
                                    .or_insert(i_j_swing_score * i_label);
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
