use serde::{Deserialize, Serialize};

#[derive(Serialize, Deserialize)]
pub struct Param {
    pub user: String,
    pub n_rec: usize,
}

#[derive(Debug, Serialize, Deserialize)]
pub struct Recommendation {
    pub rec_list: Vec<String>,
}

#[derive(Deserialize)]
pub struct Prediction {
    pub outputs: Vec<f32>,
}
