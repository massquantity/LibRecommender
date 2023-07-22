use tonic::transport::Server;

use actix_serving::online_deploy_grpc::recommend_proto::recommend_server::RecommendServer;
use actix_serving::online_deploy_grpc::RecommendService;
use actix_serving::redis_ops;

#[tokio::main(worker_threads = 4)]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    std::env::set_var("RUST_LOG", "info");
    env_logger::init();

    let addr = "[::1]:50051".parse()?;
    let redis_pool = redis_ops::create_redis_pool(String::from("127.0.0.1"))
        .expect("Failed to connect to redis pool");
    let service = RecommendService { redis_pool };

    Server::builder()
        .add_service(RecommendServer::new(service))
        .serve(addr)
        .await?;

    Ok(())
}
