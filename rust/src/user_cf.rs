use fxhash::{FxHashMap, FxHashSet};
use pyo3::prelude::*;
use pyo3::types::*;
use serde::{Deserialize, Serialize};

use crate::incremental::{update_by_sims, update_cosine, update_sum_squares};
use crate::inference::{compute_pred, get_intersect_neighbors, get_rec_items};
use crate::serialization::{load_model, save_model};
use crate::similarities::{compute_sum_squares, forward_cosine, invert_cosine, sort_by_sims};
use crate::sparse::{get_row, CsrMatrix};
use crate::utils::CumValues;

#[pyclass(module = "recfarm", name = "UserCF")]
#[derive(Serialize, Deserialize)]
pub struct PyUserCF {
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
impl PyUserCF {
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
        task: &str,
        k_sim: usize,
        n_users: usize,
        n_items: usize,
        min_common: usize,
        user_interactions: &Bound<'_, PyAny>,
        item_interactions: &Bound<'_, PyAny>,
        user_consumed: &Bound<'_, PyDict>,
        default_pred: f32,
    ) -> PyResult<Self> {
        let user_interactions: CsrMatrix<i32, f32> = user_interactions.extract()?;
        let item_interactions: CsrMatrix<i32, f32> = item_interactions.extract()?;
        let user_consumed: FxHashMap<i32, Vec<i32>> = user_consumed.extract()?;
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

    /// forward index: sparse matrix of `user` interaction
    /// invert index: sparse matrix of `item` interaction
    fn compute_similarities(&mut self, invert: bool, num_threads: usize) -> PyResult<()> {
        self.sum_squares = compute_sum_squares(&self.user_interactions, self.n_users);
        let cosine_sims = if invert {
            invert_cosine(
                &self.item_interactions,
                &self.sum_squares,
                &mut self.cum_values,
                self.n_users,
                self.n_items,
                self.min_common,
            )?
        } else {
            std::env::set_var("RAYON_NUM_THREADS", format!("{num_threads}"));
            // rayon::ThreadPoolBuilder::new().num_threads(num_threads).build_global().unwrap();
            forward_cosine(
                &self.user_interactions,
                &self.sum_squares,
                &mut self.cum_values,
                self.n_users,
                self.min_common,
            )?
        };
        sort_by_sims(self.n_users, &cosine_sims, &mut self.sim_mapping)?;
        Ok(())
    }

    fn num_sim_elements(&self) -> PyResult<usize> {
        let n_elements = self.sim_mapping.values().map(|i| i.0.len()).sum();
        Ok(n_elements)
    }

    /// sparse matrix of `item` interaction
    fn predict(&self, users: &Bound<'_, PyList>, items: &Bound<'_, PyList>) -> PyResult<Vec<f32>> {
        let mut preds = Vec::new();
        let users: Vec<i32> = users.extract()?;
        let items: Vec<usize> = items.extract()?;
        for (&u, &i) in users.iter().zip(items.iter()) {
            if usize::try_from(u)? == self.n_users || i == self.n_items {
                preds.push(self.default_pred);
                continue;
            }
            let pred = match (
                self.sim_mapping.get(&u),
                get_row(&self.item_interactions, i, false),
            ) {
                (Some((sim_users, sim_values)), Some(user_labels)) => {
                    let sim_num = std::cmp::min(self.k_sim, sim_users.len());
                    let mut user_sims: Vec<(i32, f32)> = sim_users[..sim_num]
                        .iter()
                        .zip(sim_values[..sim_num].iter())
                        .map(|(u, s)| (*u, *s))
                        .collect();
                    user_sims.sort_unstable_by_key(|&(u, _)| u);
                    let user_labels: Vec<(i32, f32)> = user_labels.collect();
                    let (k_nb_sims, k_nb_labels) =
                        get_intersect_neighbors(&user_sims, &user_labels, self.k_sim);
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
    fn recommend<'py>(
        &self,
        py: Python<'py>,
        users: &Bound<'py, PyList>,
        n_rec: usize,
        filter_consumed: bool,
        random_rec: bool,
    ) -> PyResult<(Vec<Bound<'py, PyList>>, Bound<'py, PyList>)> {
        let mut recs = Vec::new();
        let mut no_rec_indices = Vec::new();
        for (k, u) in users.iter().enumerate() {
            let u: i32 = u.extract()?;
            let consumed = self
                .user_consumed
                .get(&u)
                .map_or(FxHashSet::default(), FxHashSet::from_iter);
            if let Some((sim_users, sim_values)) = self.sim_mapping.get(&u) {
                let mut item_scores: FxHashMap<i32, f32> = FxHashMap::default();
                let sim_num = std::cmp::min(self.k_sim, sim_users.len());
                for (&v, &u_v_sim) in sim_users[..sim_num]
                    .iter()
                    .zip(sim_values[..sim_num].iter())
                {
                    let v = usize::try_from(v)?;
                    if let Some(row) = get_row(&self.user_interactions, v, false) {
                        for (i, v_i_score) in row {
                            if filter_consumed && consumed.contains(&i) {
                                continue;
                            }
                            item_scores
                                .entry(i)
                                .and_modify(|score| *score += u_v_sim * v_i_score)
                                .or_insert(u_v_sim * v_i_score);
                        }
                    }
                }

                if item_scores.is_empty() {
                    recs.push(PyList::empty(py));
                    no_rec_indices.push(k);
                    continue;
                }
                let items = get_rec_items(item_scores, n_rec, random_rec);
                recs.push(PyList::new(py, items)?);
            } else {
                recs.push(PyList::empty(py));
                no_rec_indices.push(k);
            }
        }

        let no_rec_indices = PyList::new(py, no_rec_indices)?;
        Ok((recs, no_rec_indices))
    }

    /// update on new sparse interactions
    fn update_similarities(
        &mut self,
        user_interactions: &Bound<'_, PyAny>,
        item_interactions: &Bound<'_, PyAny>,
    ) -> PyResult<()> {
        let new_user_interactions: CsrMatrix<i32, f32> = user_interactions.extract()?;
        let new_item_interactions: CsrMatrix<i32, f32> = item_interactions.extract()?;
        update_sum_squares(&mut self.sum_squares, &new_user_interactions, self.n_users);
        let cosine_sims = update_cosine(
            &new_item_interactions,
            &self.sum_squares,
            &mut self.cum_values,
            self.n_users,
            self.min_common,
        )?;
        update_by_sims(self.n_users, &cosine_sims, &mut self.sim_mapping)?;

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
}

#[pyfunction]
#[pyo3(name = "save_user_cf")]
pub fn save(model: &PyUserCF, path: &str, model_name: &str) -> PyResult<()> {
    save_model(model, path, model_name, "UserCF")?;
    Ok(())
}

#[pyfunction]
#[pyo3(name = "load_user_cf")]
pub fn load(path: &str, model_name: &str) -> PyResult<PyUserCF> {
    let model = load_model(path, model_name, "UserCF")?;
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
            let user_interactions = Bound::new(
                py,
                PySparseMatrix {
                    sparse_indices: vec![0, 1, 0, 1, 1, 2, 0, 1, 2, 1, 2],
                    sparse_indptr: vec![0, 2, 4, 6, 9, 11],
                    sparse_data: vec![1., 1., 2., 1., 1., 1., 2., 1., 1., 1., 2.],
                },
            )?;
            let item_interactions = Bound::new(
                py,
                PySparseMatrix {
                    sparse_indices: vec![0, 1, 3, 0, 1, 2, 3, 4, 2, 3, 4],
                    sparse_indptr: vec![0, 3, 8, 11, 11],
                    sparse_data: vec![1., 2., 2., 1., 1., 1., 1., 1., 1., 1., 2.],
                },
            )?;
            let user_consumed = [
                (0, vec![0, 1]),
                (1, vec![0, 1]),
                (2, vec![1, 2]),
                (3, vec![0, 1, 2]),
                (4, vec![1, 2]),
            ]
            .into_py_dict(py)?;

            let mut user_cf = PyUserCF::new(
                task,
                k_sim,
                n_users,
                n_items,
                min_common,
                &user_interactions,
                &item_interactions,
                &user_consumed,
                default_pred,
            )?;
            user_cf.compute_similarities(true, 1)?;
            Ok(user_cf)
        })?;
        Ok(user_cf)
    }

    #[test]
    fn test_user_cf_training() -> Result<(), Box<dyn std::error::Error>> {
        let get_nbs = |model: &PyUserCF, u: i32| model.sim_mapping[&u].0.to_owned();
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
        let get_nbs = |model: &PyUserCF, u: i32| model.sim_mapping[&u].0.to_owned();
        pyo3::prepare_freethreaded_python();
        let mut user_cf = get_user_cf()?;
        Python::with_gil(|py| -> PyResult<()> {
            // larger user_interactions:
            // [
            //     [0, 0, 0, 0, 0],
            //     [3, 0, 0, 0, 0],
            //     [5, 0, 0, 0, 0],
            //     [0, 0, 0, 0, 0],
            //     [0, 0, 0, 0, 0],
            //     [2, 2, 1, 2, 0],
            // ]
            let user_interactions = Bound::new(
                py,
                PySparseMatrix {
                    sparse_indices: vec![0, 0, 0, 1, 2, 3],
                    sparse_indptr: vec![0, 0, 1, 2, 2, 2, 6],
                    sparse_data: vec![3.0, 5.0, 2.0, 2.0, 1.0, 2.0],
                },
            )?;
            let item_interactions = Bound::new(
                py,
                PySparseMatrix {
                    sparse_indices: vec![1, 2, 5, 5, 5, 5],
                    sparse_indptr: vec![0, 3, 4, 5, 6, 6],
                    sparse_data: vec![3.0, 5.0, 2.0, 2.0, 1.0, 2.0],
                },
            )?;
            let _user_consumed = [
                (0, vec![0, 1]),
                (1, vec![0, 1]),
                (2, vec![1, 2]),
                (3, vec![0, 1, 2]),
                (4, vec![1, 2]),
                (5, vec![0, 1, 2, 3]),
            ]
            .into_py_dict(py)?;

            user_cf.n_users = 6;
            user_cf.n_items = 5;
            user_cf.user_consumed = _user_consumed.extract::<FxHashMap<i32, Vec<i32>>>()?;
            user_cf.update_similarities(&user_interactions, &item_interactions)?;
            let users = PyList::new(py, vec![5, 1])?;
            let rec_result = user_cf.recommend(py, &users, 10, true, false)?;
            assert_eq!(rec_result.0.len(), 2);
            Ok(())
        })?;
        assert_eq!(get_nbs(&user_cf, 0), vec![1, 3, 2, 4]);
        assert_eq!(get_nbs(&user_cf, 1), vec![2, 0, 3, 5, 4]);
        assert_eq!(get_nbs(&user_cf, 2), vec![1, 4, 3, 5, 0]);
        assert_eq!(get_nbs(&user_cf, 3), vec![1, 0, 2, 4]);
        assert_eq!(get_nbs(&user_cf, 4), vec![2, 3, 0, 1]);
        assert_eq!(get_nbs(&user_cf, 5), vec![2, 1]);

        Python::with_gil(|py| -> PyResult<()> {
            // smaller user_interactions:
            // [
            //     [0, 0, 0, 3, 2],
            //     [0, 0, 0, 0, 0],
            //     [0, 0, 0, 0, 0],
            //     [0, 1, 0, 4, 3],
            // ]
            let user_interactions = Bound::new(
                py,
                PySparseMatrix {
                    sparse_indices: vec![3, 4, 1, 3, 4],
                    sparse_indptr: vec![0, 2, 2, 2, 5],
                    sparse_data: vec![3.0, 2.0, 1.0, 4.0, 3.0],
                },
            )?;
            let item_interactions = Bound::new(
                py,
                PySparseMatrix {
                    sparse_indices: vec![3, 0, 3, 0, 3],
                    sparse_indptr: vec![0, 0, 1, 1, 3, 5],
                    sparse_data: vec![1.0, 3.0, 4.0, 2.0, 3.0],
                },
            )?;
            user_cf.update_similarities(&user_interactions, &item_interactions)?;
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

    #[test]
    fn test_save_model() -> Result<(), Box<dyn std::error::Error>> {
        pyo3::prepare_freethreaded_python();
        let model = get_user_cf()?;
        let cur_dir = std::env::current_dir()?
            .to_string_lossy()
            .to_string();
        let model_name = "user_cf_model";
        save(&model, &cur_dir, model_name)?;

        let new_model: PyUserCF = load(&cur_dir, model_name)?;
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
