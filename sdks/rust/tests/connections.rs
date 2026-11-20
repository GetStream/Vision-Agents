mod support;

use axum::http::Method;
use serde_json::{Value, json};
use support::{SECRET, Seen, Server, claims, connection, connections, refusal};
use vision_agents::{
    AuthorizeOptions, Client, ClientOptions, ConnectionFilter, Error, NewConnection, types,
};

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

/// A new connection to `linear`, for `user_id` if there is one.
fn new(user_id: Option<&str>) -> NewConnection {
    NewConnection {
        connector: "linear".into(),
        user_id: user_id.map(Into::into),
        ..Default::default()
    }
}

/// An authorization for `user_id` if there is one.
fn authorization(user_id: Option<&str>) -> AuthorizeOptions {
    AuthorizeOptions {
        user_id: user_id.map(Into::into),
        ..Default::default()
    }
}

/// A list of the connections of `user_id` if there is one.
fn filter(user_id: Option<&str>) -> ConnectionFilter {
    ConnectionFilter {
        user_id: user_id.map(Into::into),
        ..Default::default()
    }
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
        .create(&NewConnection {
            label: Some("work".into()),
            ..new(Some("ada"))
        })
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
async fn an_app_connection_names_no_owner_and_leaves_the_clients_own_user_as_it_is() {
    let server = Server::start().await;
    server.route(
        Method::POST,
        "/v1/agents/connections",
        201,
        connection("conn_1", "linear", None),
    );
    let client = acting_backend(&server, "bob", false);

    client.connections().create(&new(None)).await.unwrap();
    client.connections().create(&new(Some(""))).await.unwrap();

    let seen = server.requests(Method::POST, "/v1/agents/connections");
    assert_eq!(seen.len(), 2);
    for seen in seen {
        assert_eq!(
            seen.body,
            json!({"connector_id": "linear", "owner": {"type": "app"}})
        );
        // The client was built for bob; a call naming nobody does not take that from it.
        assert_eq!(acting_users(&seen), ["bob"]);
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
    // A call naming a user is that user's alone, whoever the client was built for; one naming
    // nobody leaves the client's own user as it is.
    let client = acting_backend(&server, "bob", false);
    let connections = client.connections();

    for user in [Some("ada"), None, Some("")] {
        connections.create(&new(user)).await.unwrap();
        connections
            .authorize("conn_1", &authorization(user))
            .await
            .unwrap();
        connections.get("conn_1", user).await.unwrap();
        connections.list(&filter(user)).await.unwrap();
        connections.delete("conn_1", user).await.unwrap();
    }

    let seen = server.seen();
    assert_eq!(seen.len(), 15);
    for (index, seen) in seen.iter().enumerate() {
        let expected = if index < 5 { ["ada"] } else { ["bob"] };
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
    // A named user rides the header beside a server token; none leaves bob's own token.
    let [named, unnamed] = &seen[..] else {
        panic!("{} requests", seen.len());
    };
    assert_eq!(acting_users(named), ["ada"]);
    let token = named.header("authorization").trim_start_matches("Bearer ");
    assert_eq!(claims(token)["server"], json!(true));
    assert_eq!(claims(token)["user_id"], Value::Null);
    assert_eq!(acting_users(unnamed), Vec::<&str>::new());
    let token = unnamed
        .header("authorization")
        .trim_start_matches("Bearer ");
    assert_eq!(claims(token)["user_id"], json!("bob"));
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
            &AuthorizeOptions {
                return_url: Some("https://app.example.com/connected".into()),
                user_id: Some("ada".into()),
            },
        )
        .await
        .unwrap();
    connections
        .authorize("conn_1", &AuthorizeOptions::default())
        .await
        .unwrap();

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
        .list(&ConnectionFilter {
            connector: Some("linear".into()),
            user_id: Some("ada".into()),
        })
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
        .list(&ConnectionFilter::default())
        .await
        .unwrap();

    assert_eq!(listed.len(), 1);
    let seen = server.request(Method::GET, "/v1/agents/connections");
    assert_eq!(seen.query, "owner_type=app&limit=200");
}

#[tokio::test]
async fn a_list_naming_an_empty_user_is_the_apps_not_a_users() {
    let server = Server::start().await;
    server.route(
        Method::GET,
        "/v1/agents/connections",
        200,
        connections(vec![connection("conn_1", "linear", None)], None),
    );

    server
        .backend()
        .connections()
        .list(&filter(Some("")))
        .await
        .unwrap();

    let seen = server.request(Method::GET, "/v1/agents/connections");
    assert_eq!(seen.query, "owner_type=app&limit=200");
    assert_eq!(acting_users(&seen), Vec::<&str>::new());
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
            .list(&ConnectionFilter::default())
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
        connections
            .authorize("", &authorization(Some("ada")))
            .await
            .map(drop),
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

/// An API key and its user's token, no secret: what a device holds.
fn device(server: &Server) -> Client {
    Client::new(ClientOptions {
        url: Some(server.url.clone()),
        api_key: Some("key".into()),
        token: Some("jean-token".into()),
        user_id: Some("jean".into()),
        authenticate: Some(false),
        ..Default::default()
    })
    .unwrap()
}

#[tokio::test]
async fn a_device_naming_a_user_is_refused_before_the_router() {
    let server = Server::start().await;
    let connections = device(&server).connections();

    let refused = [
        connections.create(&new(Some("ada"))).await.map(drop),
        connections
            .authorize("conn_1", &authorization(Some("ada")))
            .await
            .map(drop),
        connections.get("conn_1", Some("ada")).await.map(drop),
        connections.list(&filter(Some("ada"))).await.map(drop),
        connections.delete("conn_1", Some("ada")).await,
    ];

    for refused in refused {
        match refused {
            Err(Error::Configuration(message)) => {
                assert!(message.contains("acts only as that user"), "{message}")
            }
            other => panic!("not refused as a configuration error: {other:?}"),
        }
    }
    assert!(server.seen().is_empty());
}

#[tokio::test]
async fn a_device_naming_no_user_sends_what_it_always_did() {
    let server = Server::start().await;
    server.route(
        Method::GET,
        "/v1/agents/connections/conn_1",
        200,
        connection("conn_1", "github", None),
    );
    server.route(
        Method::GET,
        "/v1/agents/connections",
        200,
        connections(vec![], None),
    );
    let connections = device(&server).connections();

    for user in [None, Some("")] {
        connections.get("conn_1", user).await.unwrap();
        connections.list(&filter(user)).await.unwrap();
    }

    let seen = server.seen();
    assert_eq!(seen.len(), 4);
    for seen in seen {
        assert_eq!(seen.header("authorization"), "Bearer jean-token");
        assert_eq!(seen.header("stream-auth-type"), "jwt");
        assert!(acting_users(&seen).is_empty());
    }
}

#[tokio::test]
async fn a_customer_id_client_still_names_a_user() {
    let server = Server::start().await;
    server.route(
        Method::POST,
        "/v1/agents/connections",
        201,
        users("conn_1", "linear", "ada"),
    );
    let page = server.client().as_user("jean", "jean-token").unwrap();

    page.connections().create(&new(Some("ada"))).await.unwrap();

    let seen = server.request(Method::POST, "/v1/agents/connections");
    assert_eq!(
        seen.body["owner"],
        json!({"type": "user", "user_id": "ada"})
    );
    assert_eq!(acting_users(&seen), ["ada"]);
    assert_eq!(seen.header("x-customer-id"), "examples");
    assert_eq!(seen.header("authorization"), "");
}
