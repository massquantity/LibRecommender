#![allow(clippy::too_many_arguments)]

use pyo3::prelude::*;

mod similarities;
mod user_cf;

/// RecFarm module
#[pymodule]
fn recfarm(_py: Python, m: &PyModule) -> PyResult<()> {
    m.add_class::<user_cf::UserCF>()?;
    Ok(())
}
