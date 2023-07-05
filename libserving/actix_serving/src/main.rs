use actix_web::http::{header::ContentType, StatusCode};
use actix_web::{http, middleware::Logger, web, App, HttpResponse, HttpServer};
use once_cell::sync::Lazy;

use actix_serving::common::get_env;
use actix_serving::embed_deploy::{init_emb_state, EmbedAppState};
use actix_serving::redis_ops;
use actix_serving::tf_deploy::{init_tf_state, TfAppState};
use actix_serving::{embed_serving, knn_serving, tf_serving};

static EMB_STATE: Lazy<web::Data<EmbedAppState>> = Lazy::new(|| web::Data::new(init_emb_state()));

static TF_STATE: Lazy<web::Data<TfAppState>> = Lazy::new(|| web::Data::new(init_tf_state()));

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
            // let faiss_index = RefCell::new(load_faiss_index());
            "embed" => app
                .app_data(web::Data::clone(&EMB_STATE))
                .service(embed_serving),
            "tf" => app
                .app_data(web::Data::clone(&TF_STATE))
                .service(tf_serving),
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
