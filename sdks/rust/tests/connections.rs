mod support;

use axum::http::Method;
use serde_json::{Value, json};
use support::{SECRET, Seen, Server, claims, connection, connections, refusal};
use vision_agents::{Client, ClientOptions, Error, types};

/// One of `user`'s connections to `connector`, as the router describes it.
fn users(id: &str, connector: &str, user: &str) -> Value {
    let mut connection = connection(id, connector, None);
    connection["owner"] = json!({"type": "user", "user_id": user});
    connection["status"] = json!("pending");
    connection
}

/// A backend that acts for `user_id` on its own, as `ClientOptions::user_id` makes one.
fn acting_backend(server: &Server, user_id: &str, authenticate: bool) -> Client {
    Client::new(ClientOptions {
        url: Some(server.url.clone()),
        api_key: Some("key".into()),
        api_secret: Some(SECRET.into()),
        user_id: Some(user_id.into()),
        authenticate: Some(authenticate),
        ..Default::default()
    })
    .unwrap()
}

/// Every `X-Stream-User-Id` a request carried.
fn acting_users(seen: &Seen) -> Vec<&str> {
    seen.headers
        .get_all("x-stream-user-id")
        .iter()
        .map(|value| value.to_str().unwrap())
        .collect()
}

#[tokio::test]
async fn a_users_connection_names_them_as_owner_and_acting_user() {
    let server = Server::start().await;
    server.route(
        Method::POST,
        "/v1/agents/connections",
        201,
        users("conn_1", "linear", "ada"),
    );

    let created = server
        .backend()
        .connections()
        .create("linear", Some("work"), Some("ada"))
        .await
        .unwrap();

    assert_eq!(created.id, "conn_1");
    let seen = server.request(Method::POST, "/v1/agents/connections");
    assert_eq!(
        seen.body,
        json!({
            "connector_id": "linear",
            "label": "work",
            "owner": {"type": "user", "user_id": "ada"},
        })
    );
    assert_eq!(acting_users(&seen), ["ada"]);
}

#[tokio::test]
async fn an_app_connection_names_no_user_even_from_a_client_acting_for_one() {
    let server = Server::start().await;
    server.route(
        Method::POST,
        "/v1/agents/connections",
        201,
        connection("conn_1", "linear", None),
    );
    let client = acting_backend(&server, "bob", false);

    client
        .connections()
        .create("linear", None, None)
        .await
        .unwrap();
    client
        .connections()
        .create("linear", None, Some(""))
        .await
        .unwrap();

    let seen = server.requests(Method::POST, "/v1/agents/connections");
    assert_eq!(seen.len(), 2);
    for seen in seen {
        assert_eq!(
            seen.body,
            json!({"connector_id": "linear", "owner": {"type": "app"}})
        );
        assert_eq!(acting_users(&seen), Vec::<&str>::new());
    }
}

#[tokio::test]
async fn every_call_sends_the_user_it_names_and_only_that_one() {
    let server = Server::start().await;
    server.route(
        Method::POST,
        "/v1/agents/connections",
        201,
        users("conn_1", "linear", "ada"),
    );
    server.route(
        Method::POST,
        "/v1/agents/connections/conn_1/authorizations",
        201,
        json!({
            "id": "auth_1", "kind": "consent", "launch_url": "https://example.com/launch",
            "handoff_token": "handoff", "expires_at": "2026-11-19T10:10:00Z",
        }),
    );
    server.route(
        Method::GET,
        "/v1/agents/connections/conn_1",
        200,
        users("conn_1", "linear", "ada"),
    );
    server.route(
        Method::GET,
        "/v1/agents/connections",
        200,
        connections(vec![users("conn_1", "linear", "ada")], None),
    );
    server.route(
        Method::DELETE,
        "/v1/agents/connections/conn_1",
        204,
        Value::Null,
    );
    // A client acting for somebody else on its own still names only the user each call names.
    let client = acting_backend(&server, "bob", false);
    let connections = client.connections();

    for user in [Some("ada"), None, Some("")] {
        connections.create("linear", None, user).await.unwrap();
        connections.authorize("conn_1", None, user).await.unwrap();
        connections.get("conn_1", user).await.unwrap();
        connections.list(None, user).await.unwrap();
        connections.delete("conn_1", user).await.unwrap();
    }

    let seen = server.seen();
    assert_eq!(seen.len(), 15);
    for (index, seen) in seen.iter().enumerate() {
        let expected: Vec<&str> = if index < 5 { vec!["ada"] } else { vec![] };
        assert_eq!(
            acting_users(seen),
            expected,
            "{} {}",
            seen.method,
            seen.path
        );
    }
}

#[tokio::test]
async fn behind_the_proxy_a_user_is_named_by_the_header_beside_a_server_token() {
    let server = Server::start().await;
    server.route(
        Method::GET,
        "/v1/agents/connections/conn_1",
        200,
        users("conn_1", "linear", "ada"),
    );
    let client = acting_backend(&server, "bob", true);

    client
        .connections()
        .get("conn_1", Some("ada"))
        .await
        .unwrap();
    client.connections().get("conn_1", None).await.unwrap();

    let seen = server.requests(Method::GET, "/v1/agents/connections/conn_1");
    for (seen, expected) in seen.iter().zip([vec!["ada"], vec![]]) {
        assert_eq!(acting_users(seen), expected);
        let token = seen.header("authorization").trim_start_matches("Bearer ");
        assert_eq!(claims(token)["server"], json!(true));
        assert_eq!(claims(token)["user_id"], Value::Null);
    }
}

#[tokio::test]
async fn authorizing_answers_where_and_how_long_the_consent_runs() {
    let server = Server::start().await;
    server.route(
        Method::POST,
        "/v1/agents/connections/conn_1/authorizations",
        201,
        json!({
            "id": "auth_1", "kind": "consent", "launch_url": "https://example.com/launch",
            "handoff_token": "handoff", "expires_at": "2026-11-19T10:10:00Z",
        }),
    );
    let connections = server.backend().connections();

    let consent = connections
        .authorize(
            "conn_1",
            Some("https://app.example.com/connected"),
            Some("ada"),
        )
        .await
        .unwrap();
    connections.authorize("conn_1", None, None).await.unwrap();

    assert_eq!(consent.kind, types::AuthorizationKind::Consent);
    assert_eq!(consent.launch_url, "https://example.com/launch");
    assert_eq!(consent.handoff_token, "handoff");
    assert_eq!(consent.expires_at, "2026-11-19T10:10:00Z");
    let seen = server.requests(Method::POST, "/v1/agents/connections/conn_1/authorizations");
    assert_eq!(
        seen[0].body,
        json!({"return_url": "https://app.example.com/connected"})
    );
    assert_eq!(seen[1].body, json!({}));
}

#[tokio::test]
async fn listing_reads_the_owners_connections_every_page() {
    let server = Server::start().await;
    server.route(
        Method::GET,
        "/v1/agents/connections",
        200,
        connections(
            vec![
                users("conn_1", "linear", "ada"),
                users("conn_2", "linear", "ada"),
            ],
            Some("c1"),
        ),
    );
    server.route(
        Method::GET,
        "/v1/agents/connections",
        200,
        connections(vec![users("conn_3", "linear", "ada")], None),
    );

    let listed = server
        .backend()
        .connections()
        .list(Some("linear"), Some("ada"))
        .await
        .unwrap();

    let ids: Vec<&str> = listed.iter().map(|c| c.id.as_str()).collect();
    assert_eq!(ids, ["conn_1", "conn_2", "conn_3"]);
    let seen = server.requests(Method::GET, "/v1/agents/connections");
    assert_eq!(seen.len(), 2);
    assert_eq!(
        seen[0].query,
        "owner_type=user&connector_id=linear&limit=200"
    );
    assert_eq!(
        seen[1].query,
        "owner_type=user&connector_id=linear&limit=200&cursor=c1"
    );
    for seen in &seen {
        assert_eq!(acting_users(seen), ["ada"]);
    }
}

#[tokio::test]
async fn the_apps_list_names_no_connector_unless_asked() {
    let server = Server::start().await;
    server.route(
        Method::GET,
        "/v1/agents/connections",
        200,
        connections(vec![connection("conn_1", "linear", None)], None),
    );

    let listed = server
        .backend()
        .connections()
        .list(None, None)
        .await
        .unwrap();

    assert_eq!(listed.len(), 1);
    let seen = server.request(Method::GET, "/v1/agents/connections");
    assert_eq!(seen.query, "owner_type=app&limit=200");
}

#[tokio::test]
async fn listing_stops_at_a_page_that_says_no_more_or_names_no_cursor() {
    for last in [
        // A cursor on a page that says there is no more is not followed.
        json!({"items": [connection("conn_1", "linear", None)], "has_more": false, "next_cursor": "c1"}),
        // Nor is an empty cursor on a page that says there is.
        json!({"items": [connection("conn_1", "linear", None)], "has_more": true, "next_cursor": ""}),
        // Nor a page with neither, whose items are null.
        json!({"items": null, "has_more": true}),
    ] {
        let server = Server::start().await;
        server.route(Method::GET, "/v1/agents/connections", 200, last.clone());
        server.route(
            Method::GET,
            "/v1/agents/connections",
            200,
            connections(vec![connection("conn_2", "linear", None)], None),
        );

        let listed = server
            .backend()
            .connections()
            .list(None, None)
            .await
            .unwrap();

        assert_eq!(
            server.requests(Method::GET, "/v1/agents/connections").len(),
            1,
            "{last}"
        );
        let expected = usize::from(!last["items"].is_null());
        assert_eq!(listed.len(), expected, "{last}");
    }
}

#[tokio::test]
async fn deleting_a_connection_sends_no_force() {
    let server = Server::start().await;
    server.route(
        Method::DELETE,
        "/v1/agents/connections/conn_1",
        204,
        Value::Null,
    );

    server
        .backend()
        .connections()
        .delete("conn_1", Some("ada"))
        .await
        .unwrap();

    let seen = server.request(Method::DELETE, "/v1/agents/connections/conn_1");
    assert_eq!(seen.query, "");
    assert_eq!(acting_users(&seen), ["ada"]);
}

#[tokio::test]
async fn a_connection_with_no_id_is_refused_before_the_router() {
    let server = Server::start().await;
    let connections = server.backend().connections();

    let refused = [
        connections.authorize("", None, Some("ada")).await.map(drop),
        connections.get("", Some("ada")).await.map(drop),
        connections.delete("", Some("ada")).await,
    ];

    for refused in refused {
        assert!(
            matches!(refused, Err(Error::Configuration(_))),
            "{refused:?}"
        );
    }
    assert!(server.seen().is_empty());
}

#[tokio::test]
async fn another_users_connection_is_the_routers_not_found() {
    let server = Server::start().await;
    server.route(
        Method::GET,
        "/v1/agents/connections/conn_1",
        404,
        refusal("not_found", "connection_not_found", "no such connection"),
    );

    let refused = server
        .backend()
        .connections()
        .get("conn_1", Some("bob"))
        .await
        .unwrap_err();

    assert_eq!(refused.status(), Some(404));
    let Error::Router(failure) = refused else {
        panic!("not the router's refusal: {refused:?}");
    };
    assert_eq!(failure.code, "connection_not_found");
}
