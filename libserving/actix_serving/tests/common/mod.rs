use std::process::Command;
use std::time;

use assert_cmd::prelude::*;
use serde::Serialize;

#[derive(Serialize)]
pub struct InvalidParam {
    pub user: i32,
    pub n_rec: usize,
}

pub fn start_server(model_type: &str) {
    Command::cargo_bin("actix_serving")
        .unwrap()
        .env("REDIS_HOST", "localhost")
        .env("PORT", "8080")
        .env("MODEL_TYPE", model_type)
        .env("RUST_LOG", "debug")
        .env("WORKERS", "2")
        .spawn()
        .expect("Failed to start actix server");
    // std::env::set_var("RUST_LOG", "debug");
    // let cmd = Command::new("./target/debug/actix_serving")
    //    .env_clear()
    //    .env("WORKERS", "2")
    //    .spawn()
    //    .unwrap();
    std::thread::sleep(time::Duration::from_secs(1));
}

pub fn stop_server() {
    Command::new("pkill")
        .arg("actix_serving")
        .output()
        .expect("Failed to stop actix server");
    std::thread::sleep(time::Duration::from_millis(200));
}
