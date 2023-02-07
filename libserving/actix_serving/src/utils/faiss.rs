use std::path::{Component::RootDir, Path, PathBuf};

use walkdir::WalkDir;

use crate::errors::{ServingError, ServingResult};

pub(crate) fn find_index_path(path: Option<String>) -> ServingResult<String> {
    let cur_dir = path.map_or(std::env::current_dir()?, PathBuf::from);
    // search in two level parent directory
    let dual_parent = cur_dir
        .parent()
        .and_then(|p| p.parent())
        .unwrap_or_else(|| Path::new(RootDir.as_os_str()));
    let walk_dirs = WalkDir::new(dual_parent)
        .into_iter()
        .filter_map(|d| d.ok());
    for entry in walk_dirs {
        let file_name = entry.file_name().to_string_lossy();
        if file_name.starts_with("faiss_index") && !entry.path().is_dir() {
            log::info!("Found faiss index in {}", entry.path().display());
            return Ok(entry.path().to_string_lossy().into_owned());
        }
    }
    Err(ServingError::NotFound("faiss index"))
}
