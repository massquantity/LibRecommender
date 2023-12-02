use fxhash::FxHashMap;
use pyo3::prelude::*;
use pyo3::types::*;
use serde::{Deserialize, Serialize};

use crate::sparse::CsrMatrix;
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
    user_consumed: FxHashMap<i32, Vec<i32>>,
    default_pred: f32,
}

#[pymethods]
impl PyItemCF {
    #[new]
    fn new(
        task: &str,
        k_sim: usize,
        n_users: usize,
        n_items: usize,
        min_common: usize,
        user_interactions: &PyAny,
        user_consumed: &PyDict,
        default_pred: f32,
    ) -> PyResult<Self> {
        let user_interactions: CsrMatrix<i32, f32> = user_interactions.extract()?;
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
            user_consumed,
            default_pred,
        })
    }
}
