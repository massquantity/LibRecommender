use std::borrow::Borrow;
use std::io::ErrorKind;

use actix_web::{middleware::Logger, web, App, HttpServer};
use deadpool_redis::{Config, Pool, Runtime};

use actix_serving::embed_deploy::{init_embed_state, EmbedAppState};
use actix_serving::{embed_serving, knn_serving, tf_serving};

fn get_env() -> std::io::Result<(String, u16, usize, String)> {
    let host = std::env::var("REDIS_HOST").unwrap_or_else(|_| String::from("127.0.0.1"));
    let port = std::env::var("PORT")
        .map_err(|e| {
            let message = ["Failed to get port", e.to_string().as_str()].join(", ");
            std::io::Error::new(ErrorKind::NotFound, message)
        })?
        .parse::<u16>()
        .map_err(|e| std::io::Error::new(ErrorKind::Other, e))?;
    let workers = std::env::var("WORKERS")
        .unwrap_or_else(|_| String::from("4"))
        .parse::<usize>()
        .map_err(|e| std::io::Error::new(ErrorKind::Other, e))?;
    let log_level = std::env::var("RUST_LOG").unwrap_or_else(|_| String::from("info"));
    Ok((host, port, workers, log_level))
}

fn create_redis_pool(host: String) -> Pool {
    let cfg = Config::from_url(format!("redis://{}:6379", host));
    cfg.create_pool(Some(Runtime::Tokio1)).expect("Failed to create redis pool")
}

#[actix_web::main]
async fn main() -> std::io::Result<()> {
    let (redis_host, port, workers, log_level) = get_env()?;
    std::env::set_var("RUST_LOG", log_level);
    // std::env::set_var("RUST_BACKTRACE", "1");
    env_logger::init();

    let model_type = std::env::var("MODEL_TYPE").unwrap_or_else(|_| String::from("no_model"));
    let embed_state: Option<web::Data<EmbedAppState>> = init_embed_state(&model_type)?;
    // let redis = redis::Client::open(format!("redis://{}:6379", redis_host)).unwrap();
    let redis_pool = create_redis_pool(redis_host);
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
                    None => panic!("Failed to load faiss index in embed serving"),
                };
                app.app_data(web::Data::clone(index_state))
                    .service(embed_serving)
            }
            "tf" => app.service(tf_serving),
            other => {
                let message = if other == "no_model" {
                    "Failed to parse model type from env".to_owned()
                } else {
                    format!("Unknown model type: {other}")
                };
                panic!("{}", message)
            }
        }
    })
    .workers(workers)
    .bind(("0.0.0.0", port))?
    .run()
    .await
}
