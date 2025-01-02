#![allow(clippy::too_many_arguments)]

use pyo3::prelude::*;

mod graph;
mod incremental;
mod inference;
mod item_cf;
mod ordering;
mod serialization;
mod similarities;
mod sparse;
mod swing;
mod user_cf;
mod utils;

const VERSION: &str = env!("CARGO_PKG_VERSION");

/// RecFarm module
#[pymodule]
fn recfarm(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<user_cf::PyUserCF>()?;
    m.add_class::<item_cf::PyItemCF>()?;
    m.add_class::<swing::PySwing>()?;
    m.add_function(wrap_pyfunction!(user_cf::save, m)?)?;
    m.add_function(wrap_pyfunction!(user_cf::load, m)?)?;
    m.add_function(wrap_pyfunction!(item_cf::save, m)?)?;
    m.add_function(wrap_pyfunction!(item_cf::load, m)?)?;
    m.add_function(wrap_pyfunction!(swing::save, m)?)?;
    m.add_function(wrap_pyfunction!(swing::load, m)?)?;
    m.add_function(wrap_pyfunction!(utils::build_consumed, m)?)?;
    m.add("__version__", VERSION)?;
    Ok(())
}
