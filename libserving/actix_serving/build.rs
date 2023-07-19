fn main() -> Result<(), Box<dyn std::error::Error>> {
    tonic_build::compile_protos("proto/recommend.proto")?;

    tonic_build::configure()
        .build_server(false)
        .compile(
            &["proto/tensorflow_serving/apis/prediction_service.proto"],
            &["proto"],
        )?;

    Ok(())
}
