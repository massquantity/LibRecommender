use actix_web::error;
use futures::stream::StreamExt;
use redis::{AsyncCommands, AsyncIter};
use serde::de::DeserializeOwned;

pub async fn list_all_keys(
    conn: &mut redis::aio::Connection,
) -> Result<Vec<String>, actix_web::Error> {
    let iter: AsyncIter<String> = conn.scan()
        .await
        .map_err(error::ErrorInternalServerError)?;
    let mut keys = iter.collect::<Vec<_>>().await;
    keys.sort();
    Ok(keys)
}

pub async fn check_exists(
    conn: &mut redis::aio::Connection,
    key: &str,
    field: &str,
    command: &str,
) -> Result<(), String> {
    let ex = match command {
        "get" | "lrange" => redis::cmd("EXISTS")
            .arg(key)
            .query_async(conn)
            .await,
        "hget" => redis::cmd("HEXISTS")
            .arg(key)
            .arg(field)
            .query_async(conn)
            .await,
        _ => return Err(format!("Unknown redis command: {}", command)),
    };
    match ex {
        Ok(true) => Ok(()),
        Ok(false) | Err(_) => Err(
            format!("{} {} {} doesn't exist", command, key, field)
        )
    }
}

pub async fn get_str(
    conn: &mut redis::aio::Connection,
    key: &str,
    field: &str,
    command: &str,
) -> Result<String, actix_web::Error> {
    match command {
        "get" => conn.get(key).await,
        "hget" => conn.hget(key, field).await,
        _ => return Err(error::ErrorInternalServerError(
            format!("Unknown redis command: {}", command)
        )),
    }.map_err(error::ErrorInternalServerError)
}

pub async fn get_multi_str(
    conn: &mut redis::aio::Connection,
    key: &str,
    fields: &[usize],
) -> Result<Vec<String>, actix_web::Error> {
    if fields.len() == 1 {
        let strs = conn
            .hget(key, fields)
            .await
            .map_err(error::ErrorInternalServerError)?;
        Ok(vec![strs])
    } else {
        Ok(conn
            .hget(key, fields)
            .await
            .map_err(error::ErrorInternalServerError)?
        )
    }
}

pub async fn get_vec<T>(
    conn: &mut redis::aio::Connection,
    key: &str,
    field: &str,
) -> actix_web::Result<T>
where
    T: DeserializeOwned
{
    let strs = get_str(conn, key, field, "hget").await?;
    let vecs = serde_json::from_str(&strs)?;
    Ok(vecs)
}
