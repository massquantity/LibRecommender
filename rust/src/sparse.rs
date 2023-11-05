pub struct SparseMatrix<T = i32, U = f32> {
    pub indices: Vec<T>,
    pub indptr: Vec<usize>,
    pub data: Vec<U>,
}
