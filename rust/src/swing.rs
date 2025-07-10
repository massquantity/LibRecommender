use fxhash::{FxHashMap, FxHashSet};
use pyo3::prelude::*;
use pyo3::types::*;
use serde::{Deserialize, Serialize};

use crate::graph::compute_swing_scores;
use crate::inference::{compute_pred, get_intersect_neighbors, get_rec_items};
use crate::serialization::{load_model, save_model};
use crate::sparse::{get_row, CsrMatrix};

#[pyclass(module = "recfarm", name = "Swing")]
#[derive(Serialize, Deserialize)]
pub struct PySwing {
    top_k: usize,
    alpha: f32,
    max_cache_num: usize,
    n_users: usize,
    n_items: usize,
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
    fn set_user_consumed(&mut self, user_consumed: &Bound<'_, PyDict>) -> PyResult<()> {
        self.user_consumed = user_consumed.extract::<FxHashMap<i32, Vec<i32>>>()?;
        Ok(())
    }

    #[new]
    fn new(
        top_k: usize,
        alpha: f32,
        max_cache_num: usize,
        n_users: usize,
        n_items: usize,
        user_interactions: &Bound<'_, PyAny>,
        item_interactions: &Bound<'_, PyAny>,
        user_consumed: &Bound<'_, PyDict>,
        default_pred: f32,
    ) -> PyResult<Self> {
        let user_consumed: FxHashMap<i32, Vec<i32>> = user_consumed.extract()?;
        let user_interactions: CsrMatrix<i32, f32> = user_interactions.extract()?;
        let item_interactions: CsrMatrix<i32, f32> = item_interactions.extract()?;
        Ok(Self {
            top_k,
            alpha,
            max_cache_num,
            n_users,
            n_items,
            swing_score_mapping: FxHashMap::default(),
            user_interactions,
            item_interactions,
            user_consumed,
            default_pred,
        })
    }

    fn compute_swing(&mut self, num_threads: usize) -> PyResult<()> {
        std::env::set_var("RAYON_NUM_THREADS", format!("{num_threads}"));
        self.swing_score_mapping.clear();
        self.swing_score_mapping = compute_swing_scores(
            &self.user_interactions,
            &self.item_interactions,
            &self.swing_score_mapping,
            self.n_users,
            self.n_items,
            self.alpha,
            self.max_cache_num,
        )?;
        Ok(())
    }

    /// update on new sparse interactions
    fn update_swing(
        &mut self,
        num_threads: usize,
        user_interactions: &Bound<'_, PyAny>,
        item_interactions: &Bound<'_, PyAny>,
    ) -> PyResult<()> {
        std::env::set_var("RAYON_NUM_THREADS", format!("{num_threads}"));
        let new_user_interactions: CsrMatrix<i32, f32> = user_interactions.extract()?;
        let new_item_interactions: CsrMatrix<i32, f32> = item_interactions.extract()?;
        self.swing_score_mapping = compute_swing_scores(
            &new_user_interactions,
            &new_item_interactions,
            &self.swing_score_mapping,
            self.n_users,
            self.n_items,
            self.alpha,
            self.max_cache_num,
        )?;

        // merge interactions for inference on new users/items
        self.user_interactions = CsrMatrix::merge(
            &self.user_interactions,
            &new_user_interactions,
            Some(self.n_users),
        );
        self.item_interactions = CsrMatrix::merge(
            &self.item_interactions,
            &new_item_interactions,
            Some(self.n_items),
        );
        Ok(())
    }

    // fn get_item_interactions(&self, user: usize) -> PyResult<Vec<i32>> {
    //     let start = self.user_interactions.indptr[user];
    //     let end = self.user_interactions.indptr[user + 1];
    //     let item_interactions = (start..end)
    //         .map(|i| self.user_interactions.indices[i])
    //         .collect();
    //     Ok(item_interactions)
    // }
    //
    // fn get_swing_scores(&self, item: i32) -> PyResult<Vec<(i32, f32)>> {
    //     let scores = match self.swing_score_mapping.get(&item).cloned() {
    //         Some(ss) => ss,
    //         None => Vec::new(),
    //     };
    //     Ok(scores)
    // }

    fn num_swing_elements(&self) -> PyResult<usize> {
        if self.swing_score_mapping.is_empty() {
            return Err(pyo3::exceptions::PyRuntimeError::new_err(
                "call `compute_swing` method before calling `num_swing_elements`",
            ));
        }
        let n_elements = self
            .swing_score_mapping
            .values()
            .map(|i| i.len())
            .sum();
        Ok(n_elements)
    }

    fn predict(&self, users: &Bound<'_, PyList>, items: &Bound<'_, PyList>) -> PyResult<Vec<f32>> {
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
                get_row(&self.user_interactions, u, false),
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
                        compute_pred("ranking", &k_nb_swings, &k_nb_labels)?
                    }
                }
                _ => self.default_pred,
            };
            preds.push(pred);
        }
        Ok(preds)
    }

    fn recommend<'py>(
        &self,
        py: Python<'py>,
        users: &Bound<'py, PyList>,
        n_rec: usize,
        filter_consumed: bool,
        random_rec: bool,
    ) -> PyResult<(Vec<Bound<'py, PyList>>, Bound<'py, PyList>)> {
        let mut recs = Vec::new();
        let mut additional_rec_counts = Vec::new();
        for u in users {
            let u: i32 = u.extract()?;
            let consumed = self
                .user_consumed
                .get(&u)
                .map_or(FxHashSet::default(), FxHashSet::from_iter);

            match get_row(&self.user_interactions, usize::try_from(u)?, false) {
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
                        additional_rec_counts.push(n_rec);
                        recs.push(PyList::empty(py));
                    } else {
                        let items = get_rec_items(item_scores, n_rec, random_rec);
                        additional_rec_counts.push(n_rec - items.len());
                        recs.push(PyList::new(py, items)?);
                    }
                }
                None => {
                    additional_rec_counts.push(n_rec);
                    recs.push(PyList::empty(py));
                }
            }
        }

        let additional_rec_counts = PyList::new(py, additional_rec_counts)?;
        Ok((recs, additional_rec_counts))
    }
}

#[pyfunction]
#[pyo3(name = "save_swing")]
pub fn save(model: &PySwing, path: &str, model_name: &str) -> PyResult<()> {
    save_model(model, path, model_name, "Swing")?;
    Ok(())
}

#[pyfunction]
#[pyo3(name = "load_swing")]
pub fn load(path: &str, model_name: &str) -> PyResult<PySwing> {
    let model = load_model(path, model_name, "Swing")?;
    Ok(model)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[pyclass]
    struct PySparseMatrix {
        #[pyo3(get)]
        sparse_indices: Vec<i32>,
        #[pyo3(get)]
        sparse_indptr: Vec<usize>,
        #[pyo3(get)]
        sparse_data: Vec<f32>,
    }

    fn get_swing_model() -> Result<PySwing, Box<dyn std::error::Error>> {
        let top_k = 10;
        let alpha = 1.0;
        let cache_common_num = 100;
        let n_users = 3;
        let n_items = 5;
        let default_pred = 0.0;
        let swing = Python::with_gil(|py| -> PyResult<PySwing> {
            // item_interactions:
            // [
            //     [1, 1, 1],
            //     [1, 1, 0],
            //     [1, 0, 1],
            //     [1, 1, 1],
            //     [0, 0, 1],
            // ]
            let item_interactions = Bound::new(
                py,
                PySparseMatrix {
                    sparse_indices: vec![0, 1, 2, 0, 1, 0, 2, 0, 1, 2, 2],
                    sparse_indptr: vec![0, 3, 5, 7, 10, 11],
                    sparse_data: vec![1.0; 11],
                },
            )?;
            let user_interactions = Bound::new(
                py,
                PySparseMatrix {
                    sparse_indices: vec![0, 1, 2, 3, 0, 1, 3, 0, 2, 3, 4],
                    sparse_indptr: vec![0, 4, 7, 11],
                    sparse_data: vec![1.0; 11],
                },
            )?;
            let user_consumed = [
                (0, vec![0, 1]),
                (1, vec![0, 1]),
                (2, vec![0, 1, 2]),
            ]
            .into_py_dict(py)?;

            let mut swing = PySwing::new(
                top_k,
                alpha,
                cache_common_num,
                n_users,
                n_items,
                &user_interactions,
                &item_interactions,
                &user_consumed,
                default_pred,
            )?;
            swing.compute_swing(2)?;
            Ok(swing)
        })?;
        Ok(swing)
    }

    #[test]
    fn test_swing_training() -> Result<(), Box<dyn std::error::Error>> {
        pyo3::prepare_freethreaded_python();
        let match_item_0 = |model: &PySwing, p: usize, i: i32, s: f32| {
            let (item, score) = model.swing_score_mapping[&0][p];
            item == i && (score - s).abs() < 1e-10
        };
        let user_weights = [
            4_f32.sqrt().recip(),
            3_f32.sqrt().recip(),
            4_f32.sqrt().recip(),
        ];
        let common_nums = [2.0, 2.0, 1.0]; // user_0_1, user_0_2, user_1_2;
        let swing_0_1 = user_weights[0] * user_weights[1] * (1_f32 + common_nums[0]).recip();
        let swing_0_2 = user_weights[0] * user_weights[2] * (1_f32 + common_nums[1]).recip();
        let swing_0_3 = user_weights[0] * user_weights[1] * (1_f32 + common_nums[0]).recip()
            + user_weights[0] * user_weights[2] * (1_f32 + common_nums[1]).recip()
            + user_weights[1] * user_weights[2] * (1_f32 + common_nums[2]).recip();
        // let swing_0_4 = 0_f32;
        let swing_model = get_swing_model()?;
        assert_eq!(swing_model.swing_score_mapping[&0].len(), 3);
        assert!(match_item_0(&swing_model, 0, 3, swing_0_3));
        assert!(match_item_0(&swing_model, 1, 1, swing_0_1));
        assert!(match_item_0(&swing_model, 2, 2, swing_0_2));
        Ok(())
    }

    #[test]
    fn test_save_model() -> Result<(), Box<dyn std::error::Error>> {
        pyo3::prepare_freethreaded_python();
        let model = get_swing_model()?;
        let cur_dir = std::env::current_dir()?
            .to_string_lossy()
            .to_string();
        let model_name = "swing_model";
        save(&model, &cur_dir, model_name)?;

        let new_model: PySwing = load(&cur_dir, model_name)?;
        Python::with_gil(|py| -> PyResult<()> {
            let users = PyList::new(py, vec![5, 1])?;
            let rec_result = new_model.recommend(py, &users, 10, true, false)?;
            assert_eq!(rec_result.0.len(), 2);
            Ok(())
        })?;

        std::fs::remove_file(std::env::current_dir()?.join(format!("{model_name}.gz")))?;
        Ok(())
    }
}
