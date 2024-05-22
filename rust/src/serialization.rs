use std::fs::{File, OpenOptions};
use std::io::{Read, Write};
use std::path::Path;

use flate2::read::GzDecoder;
use flate2::write::GzEncoder;
use flate2::Compression;
use pyo3::exceptions::PyIOError;
use pyo3::PyResult;
use serde::de::DeserializeOwned;
use serde::Serialize;

pub fn save_model<T: Serialize>(
    model: &T,
    path: &str,
    model_name: &str,
    class_name: &str,
) -> PyResult<()> {
    let file_name = format!("{model_name}.gz");
    let model_path = Path::new(path).join(file_name);
    let file = OpenOptions::new()
        .write(true)
        .create(true)
        .open(model_path.as_path())?;
    let mut encoder = GzEncoder::new(file, Compression::new(1));
    let model_bytes: Vec<u8> = match bincode::serialize(model) {
        Ok(bytes) => bytes,
        Err(e) => return Err(PyIOError::new_err(e.to_string())),
    };
    encoder.write_all(&model_bytes)?;
    encoder.finish()?;
    println!(
        "Save `{class_name}` model to `{}`",
        model_path.canonicalize()?.display()
    );
    Ok(())
}

pub fn load_model<T: DeserializeOwned>(
    path: &str,
    model_name: &str,
    class_name: &str,
) -> PyResult<T> {
    let file_name = format!("{model_name}.gz");
    let model_path = Path::new(path).join(file_name);
    let file = File::open(model_path.as_path())?;
    let mut decoder = GzDecoder::new(file);
    let mut model_bytes: Vec<u8> = Vec::new();
    decoder.read_to_end(&mut model_bytes)?;
    let model: T = match bincode::deserialize(&model_bytes) {
        Ok(m) => m,
        Err(e) => return Err(PyIOError::new_err(e.to_string())),
    };
    println!(
        "Load `{class_name}` model from `{}`",
        model_path.canonicalize()?.display()
    );
    Ok(model)
}
