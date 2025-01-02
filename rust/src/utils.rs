use fxhash::FxHashMap;
use pyo3::prelude::*;
use pyo3::types::{IntoPyDict, PyDict, PyList};

/// (x1, x2, prod, count)
pub(crate) type CumValues = (i32, i32, f32, usize);

#[pyfunction]
#[pyo3(name = "build_consumed_unique")]
pub fn build_consumed<'py>(
    py: Python<'py>,
    user_indices: &Bound<'py, PyList>,
    item_indices: &Bound<'py, PyList>,
) -> PyResult<(Bound<'py, PyDict>, Bound<'py, PyDict>)> {
    let add_or_insert = |mapping: &mut FxHashMap<i32, Vec<i32>>, k: i32, v: i32| {
        mapping
            .entry(k)
            .and_modify(|consumed| consumed.push(v))
            .or_insert_with(|| vec![v]);
    };
    let user_indices: Vec<i32> = user_indices.extract()?;
    let item_indices: Vec<i32> = item_indices.extract()?;
    let mut user_consumed: FxHashMap<i32, Vec<i32>> = FxHashMap::default();
    let mut item_consumed: FxHashMap<i32, Vec<i32>> = FxHashMap::default();
    for (&u, &i) in user_indices.iter().zip(item_indices.iter()) {
        add_or_insert(&mut user_consumed, u, i);
        add_or_insert(&mut item_consumed, i, u);
    }
    // remove consecutive repeated elements
    user_consumed.values_mut().for_each(|v| v.dedup());
    item_consumed.values_mut().for_each(|v| v.dedup());
    let user_consumed_py: Bound<'py, PyDict> = user_consumed.into_py_dict(py)?;
    let item_consumed_py: Bound<'py, PyDict> = item_consumed.into_py_dict(py)?;
    Ok((user_consumed_py, item_consumed_py))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_build_consumed() -> Result<(), Box<dyn std::error::Error>> {
        let get_values = |mapping: &Bound<'_, PyDict>, k: i32| -> PyResult<Vec<i32>> {
            mapping.get_item(k)?.unwrap().extract()
        };
        pyo3::prepare_freethreaded_python();
        Ok(Python::with_gil(|py| -> PyResult<()> {
            let user_indices = PyList::new(py, vec![1, 1, 1, 2, 2, 1, 2, 3, 2, 3])?;
            let item_indices = PyList::new(py, vec![11, 11, 999, 0, 11, 11, 999, 11, 999, 0])?;
            let (user_consumed, item_consumed) = build_consumed(py, &user_indices, &item_indices)?;
            assert_eq!(get_values(&user_consumed, 1)?, vec![11, 999, 11]);
            assert_eq!(get_values(&user_consumed, 2)?, vec![0, 11, 999]);
            assert_eq!(get_values(&user_consumed, 3)?, vec![11, 0]);
            assert_eq!(get_values(&item_consumed, 11)?, vec![1, 2, 1, 3]);
            assert_eq!(get_values(&item_consumed, 999)?, vec![1, 2]);
            assert_eq!(get_values(&item_consumed, 0)?, vec![2, 3]);
            Ok(())
        })?)
    }
}
