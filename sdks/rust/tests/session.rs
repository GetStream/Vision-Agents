mod support;

use std::time::Duration;

use axum::http::Method;
use serde_json::json;
use support::{Server, item, response, session};
use vision_agents::{Session, Tools, WatchOptions, types};

fn weather() -> Tools {
    let tools = Tools::new();
    tools.register(
        "weather",
        "The weather in a city",
        json!({"type": "object", "properties": {"city": {"type": "string"}}}),
        |arguments| async move {
            let city = arguments["city"].as_str().unwrap_or("").to_string();
            if city.is_empty() {
                return Err("which city?".to_string());
            }
            Ok(format!("sunny in {city}"))
        },
    );
    tools
}

async fn open(server: &Server, tools: Tools) -> (Session, support::Accepted) {
    server.route(Method::POST, "/v1/agents/sessions", 201, session("s1"));
    let client = server.client();
    let opening = tokio::spawn(async move {
        Session::open(
            &client,
            types::CreateSessionRequest {
                text: Some(true),
                ..Default::default()
            },
            tools,
            WatchOptions::default(),
        )
        .await
    });
    let socket = server.accept().await;
    (opening.await.unwrap().unwrap(), socket)
}

#[tokio::test]
async fn a_session_declares_its_tools_and_watches_its_events() {
    let server = Server::start().await;
    let (session, socket) = open(&server, weather()).await;

    assert_eq!(socket.path, "/v1/agents/sessions/s1/events");
    assert_eq!(socket.query, "decisions=false");
    assert_eq!(socket.header("x-customer-id"), "examples");
    let sent = server.request(Method::POST, "/v1/agents/sessions").body;
    assert_eq!(sent["tools"][0]["name"], "weather");
    assert_eq!(
        sent["tools"][0]["parameters"]["properties"]["city"]["type"],
        "string"
    );
    assert_eq!(session.id(), "s1");
}

#[tokio::test]
async fn commands_are_sent_on_the_socket() {
    let server = Server::start().await;
    let (session, mut socket) = open(&server, Tools::new()).await;

    session.say("hello").await.unwrap();
    session.respond("what is new").await.unwrap();
    session.interrupt().await.unwrap();
    session.set_instructions("be brief").await.unwrap();

    assert_eq!(
        socket.next().await.unwrap(),
        json!({"type": "say", "text": "hello"})
    );
    assert_eq!(
        socket.next().await.unwrap(),
        json!({"type": "respond", "text": "what is new"})
    );
    assert_eq!(socket.next().await.unwrap(), json!({"type": "interrupt"}));
    assert_eq!(
        socket.next().await.unwrap(),
        json!({"type": "instructions", "instructions": "be brief"})
    );
}

#[tokio::test]
async fn events_arrive_in_order_and_end_with_the_conversation() {
    let server = Server::start().await;
    let (session, mut socket) = open(&server, Tools::new()).await;

    socket.send(json!({"type": "heard", "text": "hi", "participant": {"id": "p1", "user_id": "ada", "name": "Ada"}})).await;
    socket
        .send(json!({"type": "responded", "text": "hello", "interrupted": true}))
        .await;
    socket.close().await;

    let heard = session.next_event().await.unwrap();
    assert_eq!(heard.kind, "heard");
    assert_eq!(heard.participant.unwrap().user_id, "ada");
    let responded = session.next_event().await.unwrap();
    assert!(responded.interrupted);
    assert_eq!(responded.text, "hello");
    assert!(session.next_event().await.is_none());
    assert!(!session.live());
}

#[tokio::test]
async fn a_tool_call_is_answered_by_the_watcher_whether_or_not_anybody_reads_events() {
    let server = Server::start().await;
    let (_session, mut socket) = open(&server, weather()).await;

    socket.send(json!({"type": "tool_call", "id": "t1", "name": "weather", "arguments": "{\"city\":\"Oslo\"}"})).await;
    socket
        .send(json!({"type": "tool_call", "id": "t2", "name": "weather", "arguments": "{}"}))
        .await;
    socket
        .send(json!({"type": "tool_call", "id": "t3", "name": "nothing", "arguments": ""}))
        .await;

    let mut results = Vec::new();
    for _ in 0..3 {
        results.push(socket.expect("tool_result").await);
    }
    results.sort_by_key(|result| result["tool_call_id"].as_str().unwrap().to_string());
    assert_eq!(
        results[0],
        json!({"type": "tool_result", "tool_call_id": "t1", "output": "sunny in Oslo"})
    );
    assert_eq!(
        results[1],
        json!({"type": "tool_result", "tool_call_id": "t2", "error": "which city?"})
    );
    assert!(
        results[2]["error"]
            .as_str()
            .unwrap()
            .contains("no such tool")
    );
}

#[tokio::test]
async fn a_cancelled_tool_call_is_not_answered() {
    let server = Server::start().await;
    let tools = Tools::new();
    tools.register("slow", "Takes a while", json!({}), |_| async {
        tokio::time::sleep(Duration::from_secs(30)).await;
        Ok::<_, String>("done")
    });
    tools.register("fast", "Does not", json!({}), |_| async {
        Ok::<_, String>("done")
    });
    let (_session, mut socket) = open(&server, tools).await;

    socket
        .send(json!({"type": "tool_call", "id": "slow-1", "name": "slow", "arguments": "{}"}))
        .await;
    socket
        .send(json!({"type": "tool_cancel", "id": "slow-1"}))
        .await;
    socket
        .send(json!({"type": "tool_call", "id": "fast-1", "name": "fast", "arguments": "{}"}))
        .await;

    assert_eq!(socket.expect("tool_result").await["tool_call_id"], "fast-1");
}

#[tokio::test]
async fn dropping_a_session_ends_the_conversation() {
    let server = Server::start().await;
    let (session, mut socket) = open(&server, Tools::new()).await;

    drop(session);

    assert_eq!(socket.next().await.unwrap(), json!({"type": "close"}));
}

#[tokio::test]
async fn within_closes_the_session_once_the_scope_is_done() {
    let server = Server::start().await;
    let (session, mut socket) = open(&server, Tools::new()).await;

    let router = tokio::spawn(async move {
        assert_eq!(
            socket.next().await.unwrap(),
            json!({"type": "say", "text": "bye"})
        );
        assert_eq!(socket.next().await.unwrap(), json!({"type": "close"}));
        socket.close().await;
    });
    let scoped = tokio::spawn(session.within(async |session| {
        session.say("bye").await?;
        Ok(session.id().to_string())
    }));
    let said = scoped.await.unwrap().unwrap();

    assert_eq!(said, "s1");
    router.await.unwrap();
}

#[tokio::test]
async fn a_socket_that_cannot_open_closes_the_session_it_was_for() {
    let server = Server::start().await;
    server.route(Method::POST, "/v1/agents/sessions", 201, session("s1"));
    server.route(Method::DELETE, "/v1/agents/sessions/s1", 204, json!(null));
    server.route(
        Method::GET,
        "/v1/agents/sessions/s1/events",
        403,
        json!({"error": "not yours"}),
    );

    let refused = Session::open(
        &server.client(),
        Default::default(),
        Tools::new(),
        WatchOptions::default(),
    )
    .await
    .unwrap_err();

    assert_eq!(refused.status(), Some(403));
    assert!(refused.to_string().contains("not yours"), "{refused}");
    server.request(Method::DELETE, "/v1/agents/sessions/s1");
}

#[tokio::test]
async fn a_turn_is_created_and_read_back() {
    let server = Server::start().await;
    let (session, _socket) = open(&server, Tools::new()).await;
    server.route(
        Method::POST,
        "/v1/agents/sessions/s1/responses",
        201,
        response("r1", "s1"),
    );
    server.route(
        Method::GET,
        "/v1/agents/sessions/s1/responses",
        200,
        json!([response("r1", "s1")]),
    );
    server.route(
        Method::GET,
        "/v1/agents/sessions/s1/responses/items",
        200,
        json!([item(0, "r1"), item(1, "r1")]),
    );

    let turn = session.responses.create("what is on today").await.unwrap();
    let listed = session.responses.list(None, None).await.unwrap();
    let items = turn.items.all().await.unwrap();

    assert_eq!(turn.id(), "r1");
    assert_eq!(listed.len(), 1);
    assert_eq!(items.len(), 2);
    assert_eq!(
        server
            .request(Method::POST, "/v1/agents/sessions/s1/responses")
            .body,
        json!({"text": "what is on today"})
    );
    assert_eq!(
        server
            .request(Method::GET, "/v1/agents/sessions/s1/responses/items")
            .query,
        "response_id=r1&limit=200&offset=0"
    );
}

#[tokio::test]
async fn a_whole_conversation_is_read_a_page_at_a_time() {
    let server = Server::start().await;
    let page: Vec<_> = (0..200).map(|ordinal| item(ordinal, "r1")).collect();
    server.route(
        Method::GET,
        "/v1/agents/sessions/s1/responses/items",
        200,
        json!(page),
    );
    server.route(
        Method::GET,
        "/v1/agents/sessions/s1/responses/items",
        200,
        json!([item(200, "r2")]),
    );

    let items = server
        .client()
        .agent("jean")
        .sessions
        .responses("s1")
        .items
        .all()
        .await
        .unwrap();

    assert_eq!(items.len(), 201);
    let queries: Vec<_> = server
        .requests(Method::GET, "/v1/agents/sessions/s1/responses/items")
        .into_iter()
        .map(|seen| seen.query)
        .collect();
    assert_eq!(queries, ["limit=200&offset=0", "limit=200&offset=200"]);
}

#[tokio::test]
async fn a_conversation_is_rewound_to_a_response() {
    let server = Server::start().await;
    server.route(
        Method::POST,
        "/v1/agents/sessions/s1/rewind",
        204,
        json!(null),
    );
    let responses = server.client().agent("jean").sessions.responses("s1");

    responses.rewind("r1").await.unwrap();
    let recorded: types::AgentResponseItem = serde_json::from_value(item(3, "r2")).unwrap();
    responses.rewind(&recorded).await.unwrap();

    let bodies: Vec<_> = server
        .requests(Method::POST, "/v1/agents/sessions/s1/rewind")
        .into_iter()
        .map(|seen| seen.body)
        .collect();
    assert_eq!(
        bodies,
        [json!({"response_id": "r1"}), json!({"response_id": "r2"})]
    );
}

#[tokio::test]
async fn a_rewind_to_nothing_is_refused_before_it_is_sent() {
    let server = Server::start().await;

    let refused = server
        .client()
        .agent("jean")
        .sessions
        .responses("s1")
        .rewind("")
        .await;

    assert!(matches!(
        refused,
        Err(vision_agents::Error::Configuration(_))
    ));
    assert!(server.seen().is_empty());
}

#[tokio::test]
async fn a_fork_at_a_response_runs_the_same_functions() {
    let server = Server::start().await;
    let (parent, _parent_socket) = open(&server, weather()).await;
    server.route(
        Method::POST,
        "/v1/agents/sessions/s1/fork",
        201,
        session("s2"),
    );

    let forking = tokio::spawn(async move {
        let fork = parent
            .fork(&types::ForkSessionRequest {
                response_id: Some("r1".into()),
                ..Default::default()
            })
            .await
            .unwrap();
        (parent, fork)
    });
    let mut socket = server.accept().await;
    let (_parent, fork) = forking.await.unwrap();

    assert_eq!(fork.id(), "s2");
    assert_eq!(socket.path, "/v1/agents/sessions/s2/events");
    assert_eq!(
        server
            .request(Method::POST, "/v1/agents/sessions/s1/fork")
            .body,
        json!({"response_id": "r1"})
    );
    socket.send(json!({"type": "tool_call", "id": "t1", "name": "weather", "arguments": "{\"city\":\"Rome\"}"})).await;
    assert_eq!(
        socket.expect("tool_result").await["output"],
        "sunny in Rome"
    );
}

#[tokio::test]
async fn an_agents_conversations_are_listed_searched_and_opened_by_name() {
    let server = Server::start().await;
    server.route(
        Method::GET,
        "/v1/agents/sessions",
        200,
        json!([session("s1")]),
    );
    server.route(
        Method::GET,
        "/v1/agents/sessions/search",
        200,
        json!([session("s1")]),
    );
    server.route(Method::POST, "/v1/agents/sessions", 201, session("s3"));
    let agent = server.client().agent("docs");

    agent
        .sessions
        .query(vision_agents::ListSessionsQuery {
            project: Some("p".into()),
            ..Default::default()
        })
        .await
        .unwrap();
    agent
        .sessions
        .search("pricing", Default::default())
        .await
        .unwrap();
    let opening = {
        let agent = agent.clone();
        tokio::spawn(async move {
            agent
                .sessions
                .create(
                    types::CreateSessionRequest {
                        title: Some("Plans".into()),
                        ..Default::default()
                    },
                    Tools::new(),
                )
                .await
        })
    };
    let _socket = server.accept().await;
    opening.await.unwrap().unwrap();

    assert_eq!(
        server.request(Method::GET, "/v1/agents/sessions").query,
        "agent=docs&project=p"
    );
    assert_eq!(
        server
            .request(Method::GET, "/v1/agents/sessions/search")
            .query,
        "q=pricing&agent=docs"
    );
    assert_eq!(
        server.request(Method::POST, "/v1/agents/sessions").body,
        json!({"agent": "docs", "text": true, "title": "Plans"})
    );
}
