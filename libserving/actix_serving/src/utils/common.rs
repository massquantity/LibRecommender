use serde::{Deserialize, Serialize};

#[derive(Deserialize)]
pub struct Param {
    pub user: String,
    pub n_rec: usize,
}

#[derive(Serialize)]
pub struct Recommendation {
    pub rec_list: Vec<String>,
}

#[derive(Deserialize)]
pub struct Prediction {
    pub outputs: Vec<f32>,
}
