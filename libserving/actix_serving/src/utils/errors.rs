use actix_web::http::StatusCode;
use actix_web::HttpResponse;

pub type ServingResult<T> = std::result::Result<T, ServingError>;

#[derive(thiserror::Error, Debug)]
pub enum ServingError {
    #[error("error: `{0}` doesn't exist in redis")]
    NotExist(&'static str),
    #[error("error: `{0}` not found")]
    NotFound(&'static str),
    #[error("error: unknown model `{0}`")]
    UnknownModel(String),
    #[error("error: redis error, {0}")]
    RedisError(#[from] redis::RedisError),
    #[error("error: failed to get redis pool, {0}")]
    RedisPoolError(#[from] deadpool_redis::PoolError),
    #[error(transparent)]
    JsonParseError(#[from] serde_json::Error),
    #[error(transparent)]
    ParseError(#[from] std::num::ParseIntError),
    #[error("faiss error: {0}")]
    FaissError(#[source] faiss::error::Error),
    #[error("error: failed to get prediction from tf serving, {0}")]
    TfServingError(#[from] reqwest::Error),
    #[error("error: failed to get environment variable `{1}`")]
    EnvError(#[source] std::env::VarError, &'static str),
    #[error("error: request timeout")]
    Timeout,
    #[error("error: {0}")]
    Other(&'static str),
}

impl actix_web::error::ResponseError for ServingError {
    fn status_code(&self) -> StatusCode {
        match *self {
            ServingError::NotExist(_) => StatusCode::BAD_REQUEST,
            ServingError::Timeout => StatusCode::REQUEST_TIMEOUT,
            ServingError::TfServingError(_) => StatusCode::GATEWAY_TIMEOUT,
            _ => StatusCode::INTERNAL_SERVER_ERROR,
        }
    }

    fn error_response(&self) -> HttpResponse {
        HttpResponse::build(self.status_code()).body(self.to_string())
    }
}

impl From<ServingError> for std::io::Error {
    fn from(e: ServingError) -> Self {
        std::io::Error::new(std::io::ErrorKind::Other, e.to_string())
    }
}
