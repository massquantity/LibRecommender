use std::borrow::Borrow;

use actix_web::{middleware::Logger, web, App, HttpServer};

use actix_serving::embed_deploy::{init_embed_state, EmbedAppState};
use actix_serving::errors::ServingError;
use actix_serving::redis_ops;
use actix_serving::{embed_serving, knn_serving, tf_serving};

fn get_env() -> Result<(String, u16, usize, String), ServingError> {
    let host = std::env::var("REDIS_HOST").unwrap_or_else(|_| String::from("127.0.0.1"));
    let port = std::env::var("PORT")
        .map_err(|e| ServingError::EnvError(e, "PORT"))?
        .parse::<u16>()?;
    let workers = std::env::var("WORKERS").map_or(Ok(4), |w| w.parse::<usize>())?;
    let log_level = std::env::var("RUST_LOG").unwrap_or_else(|_| String::from("info"));
    Ok((host, port, workers, log_level))
}

#[actix_web::main]
async fn main() -> std::io::Result<()> {
    let (redis_host, port, workers, log_level) = get_env()?;
    std::env::set_var("RUST_LOG", log_level);
    // std::env::set_var("RUST_BACKTRACE", "1");
    env_logger::init();

    let model_type = std::env::var("MODEL_TYPE").expect("Failed to get model type from env");
    let embed_state: Option<web::Data<EmbedAppState>> = init_embed_state(&model_type)?;
    // let redis = redis::Client::open(format!("redis://{}:6379", redis_host)).unwrap();
    let redis_pool = redis_ops::create_redis_pool(redis_host);
    HttpServer::new(move || {
        let logger = Logger::default();
        let app = App::new()
            .wrap(logger)
            .app_data(web::Data::new(redis_pool.clone()));

        match model_type.as_str() {
            "knn" => app.service(knn_serving),
            "embed" => {
                // let faiss_index = RefCell::new(load_faiss_index());
                let index_state = match embed_state.borrow() {
                    Some(faiss_index) => faiss_index,
                    None => {
                        eprintln!("Failed to load faiss index in embed serving");
                        std::process::exit(1)
                    }
                };
                app.app_data(web::Data::clone(index_state))
                    .service(embed_serving)
            }
            "tf" => app.service(tf_serving),
            other => {
                eprintln!("Unknown model type: `{other}`");
                std::process::exit(1)
            }
        }
    })
    .workers(workers)
    .bind(("0.0.0.0", port))?
    .run()
    .await
}
