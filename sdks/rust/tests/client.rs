mod support;

use axum::http::Method;
use serde_json::json;
use support::{SECRET, Server, claims};
use vision_agents::{Client, ClientOptions, Error, ListSessionsQuery, types};

#[tokio::test]
async fn a_router_with_nothing_in_front_is_told_the_customer() {
    let server = Server::start().await;
    server.route(
        Method::GET,
        "/health",
        200,
        json!({"status": "ok", "dependencies": {}}),
    );

    server.client().get_health().await.unwrap();

    let seen = server.request(Method::GET, "/health");
    assert_eq!(seen.header("x-customer-id"), "examples");
    assert_eq!(seen.header("authorization"), "");
}

#[tokio::test]
async fn a_backend_signs_a_server_token_per_request() {
    let server = Server::start().await;
    server.route(
        Method::GET,
        "/health",
        200,
        json!({"status": "ok", "dependencies": {}}),
    );

    let client = server.backend();
    assert!(client.server_side());
    client.get_health().await.unwrap();

    let seen = server.request(Method::GET, "/health");
    assert_eq!(seen.header("x-api-key"), "key");
    assert_eq!(seen.header("stream-auth-type"), "server");
    let claims = claims(seen.header("authorization"));
    assert_eq!(claims["server"], true);
    assert!(claims["exp"].as_u64().unwrap() > claims["iat"].as_u64().unwrap());
}

#[tokio::test]
async fn a_user_holds_their_own_token_and_is_not_a_backend() {
    let server = Server::start().await;
    server.route(
        Method::GET,
        "/health",
        200,
        json!({"status": "ok", "dependencies": {}}),
    );

    let user = server.backend().as_user("ada", "ada-token").unwrap();
    assert!(!user.server_side());
    user.get_health().await.unwrap();

    let seen = server.request(Method::GET, "/health");
    assert_eq!(seen.header("authorization"), "Bearer ada-token");
    assert_eq!(seen.header("stream-auth-type"), "jwt");
}

#[tokio::test]
async fn a_backend_acting_for_a_user_names_them() {
    let server = Server::start().await;
    server.route(
        Method::GET,
        "/health",
        200,
        json!({"status": "ok", "dependencies": {}}),
    );

    let client = Client::new(ClientOptions {
        url: Some(server.url.clone()),
        api_key: Some("key".into()),
        api_secret: Some(SECRET.into()),
        user_id: Some("ada".into()),
        ..Default::default()
    })
    .unwrap();
    client.get_health().await.unwrap();

    assert_eq!(
        server
            .request(Method::GET, "/health")
            .header("x-stream-user-id"),
        "ada"
    );
}

#[tokio::test]
async fn the_proxy_is_given_the_credential_its_own_way() {
    let server = Server::start().await;
    server.route(
        Method::GET,
        "/health",
        200,
        json!({"status": "ok", "dependencies": {}}),
    );

    let client = Client::new(ClientOptions {
        url: Some(server.url.clone()),
        api_key: Some("key".into()),
        api_secret: Some(SECRET.into()),
        authenticate: Some(true),
        ..Default::default()
    })
    .unwrap();
    client.get_health().await.unwrap();

    let seen = server.request(Method::GET, "/health");
    assert_eq!(seen.header("api_key"), "key");
    assert_eq!(seen.header("stream-auth-type"), "jwt");
    assert_eq!(claims(seen.header("authorization"))["server"], true);
}

#[test]
fn a_client_that_does_not_say_who_is_calling_is_refused() {
    let refused = Client::new(ClientOptions {
        url: Some("http://localhost:1".into()),
        customer_id: Some(String::new()),
        ..Default::default()
    });
    assert!(matches!(refused, Err(Error::Configuration(_))));

    let keyless = Client::new(ClientOptions {
        url: Some("http://localhost:1".into()),
        api_key: Some("key".into()),
        api_secret: Some(String::new()),
        ..Default::default()
    });
    assert!(matches!(keyless, Err(Error::Configuration(_))));
}

#[tokio::test]
async fn a_refusal_carries_the_status_and_what_the_router_said() {
    let server = Server::start().await;
    server.route(
        Method::GET,
        "/v1/agents/sessions/missing",
        404,
        json!({"error": "no such session"}),
    );

    let error = server.client().get_session("missing").await.unwrap_err();

    assert_eq!(error.status(), Some(404));
    match error {
        Error::Router {
            operation, message, ..
        } => {
            assert_eq!(operation, "getSession");
            assert_eq!(message, "no such session");
        }
        other => panic!("expected a refusal, got {other:?}"),
    }
}

#[tokio::test]
async fn an_answer_with_no_body_is_nothing() {
    let server = Server::start().await;
    server.route(Method::DELETE, "/v1/agents/sessions/s1", 204, json!(null));

    server.client().close_session("s1").await.unwrap();

    server.request(Method::DELETE, "/v1/agents/sessions/s1");
}

#[tokio::test]
async fn a_path_parameter_stays_one_segment() {
    let server = Server::start().await;
    server.route(
        Method::GET,
        "/v1/agents/sessions/a%2Fb%20c",
        200,
        support::session("a/b c"),
    );

    let session = server.client().get_session("a/b c").await.unwrap();

    assert_eq!(session.id, "a/b c");
}

#[tokio::test]
async fn what_was_left_out_is_left_out() {
    let server = Server::start().await;
    server.route(
        Method::POST,
        "/v1/agents/sessions",
        201,
        support::session("s1"),
    );
    server.route(Method::GET, "/v1/agents/sessions", 200, json!([]));

    let client = server.client();
    client
        .create_session(&types::CreateSessionRequest {
            text: Some(true),
            ..Default::default()
        })
        .await
        .unwrap();
    client
        .list_sessions(&ListSessionsQuery {
            limit: Some(5),
            ..Default::default()
        })
        .await
        .unwrap();

    assert_eq!(
        server.request(Method::POST, "/v1/agents/sessions").body,
        json!({"text": true})
    );
    assert_eq!(
        server.request(Method::GET, "/v1/agents/sessions").query,
        "limit=5"
    );
}

#[tokio::test]
async fn an_answer_that_is_not_what_the_spec_says_is_a_decode_error() {
    let server = Server::start().await;
    server.route(Method::GET, "/v1/agents/sessions/s1", 200, json!({"id": 7}));

    let error = server.client().get_session("s1").await.unwrap_err();

    assert!(matches!(error, Error::Decode { .. }), "{error:?}");
}

#[tokio::test]
async fn a_guest_is_minted_and_later_claimed_by_a_backend() {
    let server = Server::start().await;
    server.route(
        Method::POST,
        "/v1/agents/guests",
        201,
        json!({"id": "guest-1", "token": "guest-token", "name": "Visitor"}),
    );
    server.route(
        Method::POST,
        "/v1/agents/guests/claim",
        200,
        json!({"guest_id": "guest-1", "user_id": "ada", "sessions_moved": 2}),
    );

    let client = server.backend();
    let guest = client
        .guest_user(&types::GuestUserRequest {
            name: Some("Visitor".into()),
            ..Default::default()
        })
        .await
        .unwrap();
    let as_guest = client.as_guest(&guest).unwrap();

    let refused = as_guest.claim_guest("guest-1", "ada").await.unwrap_err();
    assert!(matches!(refused, Error::Configuration(_)));
    assert!(
        server
            .requests(Method::POST, "/v1/agents/guests/claim")
            .is_empty()
    );

    let claimed = client.claim_guest("guest-1", "ada").await.unwrap();
    assert_eq!(claimed.sessions_moved, 2);
    assert_eq!(
        server.request(Method::POST, "/v1/agents/guests").body,
        json!({"name": "Visitor"})
    );
    assert_eq!(
        server.request(Method::POST, "/v1/agents/guests/claim").body,
        json!({"guest_id": "guest-1", "user_id": "ada"})
    );
}

#[tokio::test]
async fn a_stream_call_is_created_with_a_server_token() {
    let server = Server::start().await;
    server.route(
        Method::POST,
        "/api/v2/video/call/agent/demo",
        201,
        json!({"call": {}}),
    );

    let stream = server.stream();
    let call = stream.create_call(&"demo".into(), "jean").await.unwrap();

    assert_eq!((call.id.as_str(), call.kind.as_str()), ("demo", "agent"));
    let seen = server.request(Method::POST, "/api/v2/video/call/agent/demo");
    assert_eq!(seen.query, "api_key=key");
    assert_eq!(seen.header("stream-auth-type"), "jwt");
    assert_eq!(claims(seen.header("authorization"))["server"], true);
    assert_eq!(seen.body, json!({"data": {"created_by_id": "jean"}}));
}

#[tokio::test]
async fn a_call_stream_refuses_is_reported() {
    let server = Server::start().await;
    server.route(
        Method::POST,
        "/api/v2/video/call/agent/demo",
        403,
        json!({"message": "no"}),
    );

    let error = server
        .stream()
        .create_call(&"demo".into(), "jean")
        .await
        .unwrap_err();

    assert_eq!(error.status(), Some(403));
}

#[tokio::test]
async fn a_monitoring_link_carries_a_token_for_a_listener() {
    let server = Server::start().await;
    let link = server
        .stream()
        .monitor_url(&vision_agents::Call::new("demo"), "monitor-s1", "Monitor")
        .unwrap();

    let parsed = url::Url::parse(&link).unwrap();
    assert_eq!(parsed.path(), "/demo/join/demo");
    let query: std::collections::HashMap<_, _> = parsed.query_pairs().into_owned().collect();
    assert_eq!(query["api_key"], "key");
    assert_eq!(query["skip_lobby"], "true");
    assert_eq!(query["user_name"], "Monitor");
    assert_eq!(claims(&query["token"])["user_id"], "monitor-s1");
}
