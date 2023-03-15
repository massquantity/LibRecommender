use std::time::Instant;

use clap::{Parser, ValueEnum};
use futures::StreamExt;
use serde_json::json;

use actix_serving::common::Recommendation;

#[derive(Parser)]
#[command(author, version, about, long_about = None)]
struct Args {
    #[arg(long, default_value_t = String::from("127.0.0.1"))]
    host: String,

    #[arg(long, default_value_t = 8080, value_parser = clap::value_parser!(u16).range(1..))]
    port: u16,

    /// Request user id
    #[arg(long)]
    user: String,

    /// Number of recommendations
    #[arg(long)]
    n_rec: u32,

    /// Number of requests
    #[arg(long)]
    n_times: u32,

    /// Number of threads
    #[arg(long)]
    n_threads: u32,

    /// Number of buffered futures
    #[arg(long, default_value_t = 64)]
    n_buffers: u32,

    /// Type of algorithm
    #[arg(value_enum)]
    algo: Algo,
}

#[derive(Copy, Clone, ValueEnum)]
enum Algo {
    Knn,
    Embed,
    Tf,
}

// cargo run --release --bin benchmark -- --help
// benchmark --user <USER> --n-rec <N_REC> --n-times <N_TIMES> --n-threads <N_THREADS> <ALGO>
fn main() -> Result<(), Box<dyn std::error::Error>> {
    let args = Args::parse();
    let runtime = tokio::runtime::Builder::new_multi_thread()
        .enable_all()
        .worker_threads(args.n_threads as usize)
        .thread_name("benchmark")
        .build()?;
    runtime.block_on(concurrent_requests(args))
}

async fn concurrent_requests(args: Args) -> Result<(), Box<dyn std::error::Error>> {
    let start = Instant::now();
    let client = reqwest::Client::new();
    let host = args.host;
    let port = args.port;
    let algo = match args.algo {
        Algo::Knn => "knn",
        Algo::Embed => "embed",
        Algo::Tf => "tf",
    };
    let url = format!("http://{host}:{port}/{algo}/recommend");
    let urls = vec![url; args.n_times as usize];
    let bodies = futures::stream::iter(urls)
        .map(|url| {
            let client = client.clone();
            let req = json!({"user": args.user, "n_rec": args.n_rec});
            async move {
                let resp = client.post(url).json(&req).send().await?;
                resp.json::<Recommendation>().await
            }
        })
        .buffer_unordered(args.n_buffers as usize)
        .collect::<Vec<_>>()
        .await;

    let duration = start.elapsed().as_secs_f64();
    println!(
        "# of requests: {}, first rec: {:?}",
        bodies.len(),
        bodies.first().as_ref().unwrap().as_ref().unwrap()
    );
    println!(
        "total time: {:.8?}s, {:.8} ms/request",
        duration,
        duration / args.n_times as f64 * 1000_f64
    );
    Ok(())
}
