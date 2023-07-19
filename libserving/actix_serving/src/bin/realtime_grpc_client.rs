use std::collections::HashMap;

use actix_serving::online_deploy_grpc::recommend_proto::recommend_client::RecommendClient;
use actix_serving::online_deploy_grpc::recommend_proto::{
    feature::Value as FeatValue, Feature, RecRequest,
};

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    let mut client = RecommendClient::connect("http://[::1]:50051").await?;

    let user = String::from("1");
    let n_rec = 11;
    let feature_sparse = (
        String::from("sex"),
        Feature {
            value: Some(FeatValue::StringVal(String::from("F"))),
        },
    );
    let feature_dense = (
        String::from("age"),
        Feature {
            value: Some(FeatValue::IntVal(33)),
        },
    );
    let request = RecRequest {
        user,
        n_rec,
        user_feats: HashMap::from([feature_sparse, feature_dense]),
        seq: vec![1, 2, 3],
    };

    let response = client
        .get_recommendation(tonic::Request::new(request))
        .await?;

    println!("rec for user: {:?}", response.into_inner().items);
    Ok(())
}
