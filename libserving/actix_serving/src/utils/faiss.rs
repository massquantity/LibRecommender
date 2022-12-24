use faiss::index::IndexImpl;
use faiss::read_index;
use log::info;
use std::path::Path;
use walkdir::WalkDir;

pub fn load_faiss_index() -> IndexImpl {
    let index_path = find_index_path();
    match read_index(index_path) {
        Ok(i) => i,
        Err(e) => panic!("Failed to read faiss index: {}", e),
    }
}

fn find_index_path() -> String {
    let mut index_path: String = String::from("");
    let cur_dir = std::env::current_dir().unwrap();
    // search in two level parent directory
    let dual_parent = cur_dir
        .parent()
        .unwrap_or_else(|| Path::new("/"))
        .parent()
        .unwrap_or_else(|| Path::new("/"));
    let walk_dirs = WalkDir::new(dual_parent).into_iter().filter_map(|d| d.ok());
    for entry in walk_dirs {
        let file_name = entry.file_name().to_string_lossy();
        if file_name.starts_with("faiss_index") && !entry.path().is_dir() {
            index_path = entry.path().to_string_lossy().to_string();
            break;
        }
    }
    if index_path.is_empty() {
        panic!("Failed to find faiss index in {}", dual_parent.display());
    } else {
        info!("Found faiss index in {}", index_path);
        index_path
    }
}
