use std::collections::HashMap;

use serde::{Deserialize, Serialize};
use serde_json::Value;

use crate::errors::ServingError;

#[derive(Serialize, Deserialize)]
pub struct Payload {
    pub user: String,
    pub n_rec: usize,
}

#[derive(Serialize, Deserialize)]
pub struct RealtimePayload {
    pub user: String,
    pub n_rec: usize,
    pub user_feats: Option<HashMap<String, Value>>,
    pub seq: Option<Vec<Value>>,
}

#[derive(Debug, Serialize, Deserialize)]
pub struct Recommendation {
    pub rec_list: Vec<String>,
}

#[derive(Deserialize)]
pub struct Prediction {
    pub outputs: Vec<f32>,
}

#[derive(Debug, Deserialize)]
pub struct RankedItems {
    pub outputs: Vec<u32>,
}

pub fn get_env() -> Result<(String, u16, usize, String), ServingError> {
    let host = std::env::var("REDIS_HOST").unwrap_or_else(|_| String::from("127.0.0.1"));
    let port = std::env::var("PORT")
        .map_err(|e| ServingError::EnvError(e, "PORT"))?
        .parse::<u16>()?;
    let workers = std::env::var("WORKERS").map_or(Ok(4), |w| w.parse::<usize>())?;
    let log_level = std::env::var("RUST_LOG").unwrap_or_else(|_| String::from("info"));
    Ok((host, port, workers, log_level))
}
