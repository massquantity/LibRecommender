use actix_serving::common::{Payload, Recommendation};
use pretty_assertions::assert_eq;

mod common;
use common::{start_server, stop_server, InvalidParam};

// cargo test --package actix_serving --test embed -- --test-threads=1
#[test]
fn test_main_embed_serving() {
    start_server("embed");
    let req = Payload {
        user: String::from("10"),
        n_rec: 3,
    };
    let resp: Recommendation = reqwest::blocking::Client::new()
        .post("http://localhost:8080/embed/recommend")
        .json(&req)
        .send()
        .unwrap()
        .json()
        .unwrap();
    assert_eq!(resp.rec_list.len(), 3);
    stop_server();
}

#[test]
fn test_bad_request() {
    start_server("embed");
    let invalid_req = InvalidParam { user: 10, n_rec: 3 };
    let resp = reqwest::blocking::Client::new()
        .post("http://localhost:8080/embed/recommend")
        .json(&invalid_req)
        .send()
        .unwrap();
    assert_eq!(resp.status(), reqwest::StatusCode::BAD_REQUEST);
    stop_server();
}

#[test]
fn test_not_found() {
    start_server("embed");
    let req = Payload {
        user: String::from("10"),
        n_rec: 3,
    };
    let resp = reqwest::blocking::Client::new()
        .post("http://localhost:8080/nooo_embed/recommend")
        .json(&req)
        .send()
        .unwrap();
    assert_eq!(resp.status(), reqwest::StatusCode::NOT_FOUND);
    assert_eq!(
        resp.text().unwrap(),
        "`nooo_embed/recommend` is not available, make sure you've started the right service."
    );
    stop_server();
}

#[test]
fn test_method_not_allowed() {
    start_server("embed");
    let resp = reqwest::blocking::get("http://localhost:8080/embed/recommend").unwrap();
    assert_eq!(resp.status(), reqwest::StatusCode::METHOD_NOT_ALLOWED);
    stop_server();
}
