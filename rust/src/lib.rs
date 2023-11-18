#![allow(clippy::too_many_arguments)]

use pyo3::prelude::*;

mod incremental;
mod serialization;
mod similarities;
mod sparse;
mod user_cf;

const VERSION: &str = env!("CARGO_PKG_VERSION");

/// RecFarm module
#[pymodule]
fn recfarm(_py: Python, m: &PyModule) -> PyResult<()> {
    m.add_class::<user_cf::PyUserCF>()?;
    m.add_function(wrap_pyfunction!(user_cf::save, m)?)?;
    m.add_function(wrap_pyfunction!(user_cf::load, m)?)?;
    m.add("__version__", VERSION)?;
    Ok(())
}
