use log::info;
use std::path::{Path, PathBuf};
use walkdir::WalkDir;

use crate::errors::{ServingError, ServingResult};

pub(crate) fn find_index_path(path: Option<String>) -> ServingResult<String> {
    let cur_dir = match path {
        Some(p) => PathBuf::from(p),
        None => std::env::current_dir().unwrap(),
    };
    // search in two level parent directory
    let dual_parent = cur_dir
        .parent()
        .and_then(|p| p.parent())
        .unwrap_or_else(|| Path::new("/"));
    let walk_dirs = WalkDir::new(dual_parent).into_iter().filter_map(|d| d.ok());
    for entry in walk_dirs {
        let file_name = entry.file_name().to_string_lossy();
        if file_name.starts_with("faiss_index") && !entry.path().is_dir() {
            info!("Found faiss index in {}", entry.path().display());
            return Ok(entry.path().to_string_lossy().to_string());
        }
    }
    Err(ServingError::NotFound("faiss index"))
}
