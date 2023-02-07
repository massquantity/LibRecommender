use actix_web::http::{header::ContentType, StatusCode};
use actix_web::{http, middleware::Logger, web, App, HttpResponse, HttpServer};

use actix_serving::embed_deploy::{init_embed_state, EmbedAppState};
use actix_serving::errors::ServingError;
use actix_serving::redis_ops;
use actix_serving::tf_deploy::{init_tf_state, TfAppState};
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

async fn not_found_handler(
    path: web::Path<String>,
    req_method: http::Method,
) -> actix_web::Result<HttpResponse> {
    match req_method {
        http::Method::POST => {
            let model_type = path.into_inner();
            let message = format!(
                "`{model_type}/recommend` is not available, \
                make sure you've started the right service."
            );
            let resp_body = HttpResponse::build(StatusCode::NOT_FOUND)
                .content_type(ContentType::plaintext())
                .body(message);
            Ok(resp_body)
        }
        _ => Ok(HttpResponse::MethodNotAllowed().finish()),
    }
}

#[actix_web::main]
async fn main() -> std::io::Result<()> {
    let (redis_host, port, workers, log_level) = get_env()?;
    std::env::set_var("RUST_LOG", log_level);
    // std::env::set_var("RUST_BACKTRACE", "1");
    env_logger::init();

    let model_type = std::env::var("MODEL_TYPE").expect("Failed to get `MODEL_TYPE` from env");
    let embed_state: Option<web::Data<EmbedAppState>> = init_embed_state(&model_type)?;
    let tf_state: Option<web::Data<TfAppState>> = init_tf_state(&model_type)?;
    let redis_pool = web::Data::new(redis_ops::create_redis_pool(redis_host)?);
    // create the shared `web::Data` outside the `HttpServer::new` closure
    // https://docs.rs/actix-web/4.2.1/actix_web/struct.App.html#shared-mutable-state
    HttpServer::new(move || {
        let logger = Logger::default();
        let app = App::new()
            .wrap(logger)
            .app_data(redis_pool.clone());

        match model_type.as_str() {
            "knn" => app.service(knn_serving),
            "embed" => {
                // let faiss_index = RefCell::new(load_faiss_index());
                let index_state = embed_state
                    .as_ref()
                    .expect("Failed to load faiss index in embed serving");
                app.app_data(web::Data::clone(index_state))
                    .service(embed_serving)
            }
            "tf" => {
                let client_state = tf_state
                    .as_ref()
                    .expect("Failed to get reqwest client in tf serving");
                app.app_data(web::Data::clone(client_state))
                    .service(tf_serving)
            }
            other => {
                eprintln!("Unknown model type: `{other}`");
                std::process::exit(1)
            }
        }
        .service(
            web::resource("/{model_type}/recommend")
                .name("404")
                .route(web::to(not_found_handler)),
        )
    })
    .workers(workers)
    .bind(("0.0.0.0", port))?
    .run()
    .await
}
