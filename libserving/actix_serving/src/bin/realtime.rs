use actix_web::{middleware::Logger, web, App, HttpServer};
use once_cell::sync::Lazy;

use actix_serving::common::get_env;
use actix_serving::online_serving;
use actix_serving::redis_ops::create_redis_pool;
use actix_serving::tf_deploy::{init_tf_state, TfAppState};

static TF_STATE: Lazy<web::Data<TfAppState>> = Lazy::new(|| web::Data::new(init_tf_state()));

#[actix_web::main]
async fn main() -> std::io::Result<()> {
    let (redis_host, port, workers, log_level) = get_env()?;
    std::env::set_var("RUST_LOG", log_level);
    env_logger::init();

    let redis_pool = web::Data::new(create_redis_pool(redis_host)?);
    HttpServer::new(move || {
        App::new()
            .wrap(Logger::default())
            .app_data(redis_pool.clone())
            .app_data(web::Data::clone(&TF_STATE))
            .service(online_serving)
    })
    .workers(workers)
    .bind(("0.0.0.0", port))?
    .run()
    .await
}
