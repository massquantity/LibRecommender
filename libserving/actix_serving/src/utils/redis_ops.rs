use deadpool_redis::{
    redis::{cmd, AsyncCommands},
    Config, Connection, Pool, Runtime,
};
use futures::stream::StreamExt;
use redis::{AsyncIter, RedisError};
use serde::de::DeserializeOwned;

use crate::errors::{ServingError, ServingResult};

pub struct RedisFeatKeys {
    pub user_index: &'static str,
    pub item_index: &'static str,
    pub user_value: &'static str,
    pub item_value: &'static str,
}

pub fn create_redis_pool(host: String) -> ServingResult<Pool> {
    let cfg = Config::from_url(format!("redis://{}:6379", host));
    cfg.create_pool(Some(Runtime::Tokio1))
        .map_err(ServingError::RedisCreatePoolError)
}

pub async fn list_all_keys(conn: &mut Connection) -> Result<Vec<String>, RedisError> {
    let iter: AsyncIter<String> = conn.scan().await?;
    let mut keys = iter.collect::<Vec<_>>().await;
    keys.sort();
    Ok(keys)
}

pub async fn check_exists(
    conn: &mut Connection,
    key: &str,
    field: &str,
    command: &str,
) -> Result<(), RedisError> {
    let ex = match command {
        "get" | "lrange" => cmd("EXISTS").arg(key).query_async(conn).await,
        "hget" => {
            cmd("HEXISTS")
                .arg(key)
                .arg(field)
                .query_async(conn)
                .await
        }
        _ => {
            return Err(RedisError::from(std::io::Error::new(
                std::io::ErrorKind::InvalidInput,
                format!("Unknown redis command: {}", command),
            )))
        }
    };
    match ex {
        Ok(true) => Ok(()),
        Ok(false) | Err(_) => Err(RedisError::from((
            redis::ErrorKind::ResponseError,
            "not exists",
        ))),
    }
}

pub async fn get_str(
    conn: &mut Connection,
    key: &str,
    field: &str,
    command: &str,
) -> Result<String, RedisError> {
    match command {
        "get" => cmd("GET").arg(key).query_async(conn).await,
        "hget" => {
            cmd("HGET")
                .arg(&[key, field])
                .query_async(conn)
                .await
        }
        _ => Err(RedisError::from(std::io::Error::new(
            std::io::ErrorKind::InvalidInput,
            format!("Unknown redis command: {}", command),
        ))),
    }
}

pub async fn get_multi_str(
    conn: &mut Connection,
    key: &str,
    fields: &[usize],
) -> Result<Vec<String>, RedisError> {
    if fields.len() == 1 {
        let strs = conn.hget(key, fields).await?;
        Ok(vec![strs])
    } else {
        Ok(conn.hget(key, fields).await?)
    }
}

pub async fn get_vec<T>(conn: &mut Connection, key: &str, field: &str) -> ServingResult<T>
where
    T: DeserializeOwned,
{
    let strs = get_str(conn, key, field, "hget").await?;
    let vecs = serde_json::from_str(&strs)?;
    Ok(vecs)
}
