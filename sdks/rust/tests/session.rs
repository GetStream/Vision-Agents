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
        Session::open(&client, Default::default(), tools, WatchOptions::default()).await
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
async fn a_tool_declared_whole_carries_its_title_and_who_runs_it() {
    let server = Server::start().await;
    let tools = Tools::new();
    tools.register_tool(
        types::SessionTool {
            name: "locate".into(),
            description: "Where the person is".into(),
            display_title: Some("Finding you".into()),
            executor: Some(types::SessionToolExecutor::Client),
            ..Default::default()
        },
        |_| async { Ok::<_, String>("Oslo") },
    );
    let (_session, mut socket) = open(&server, tools).await;

    let sent = server.request(Method::POST, "/v1/agents/sessions").body;
    assert_eq!(
        sent["tools"],
        json!([{"name": "locate", "description": "Where the person is",
                "display_title": "Finding you", "executor": "client"}])
    );
    socket
        .send(
            json!({"type": "tool_call", "id": "t1", "name": "locate", "arguments": "{}",
                     "request_id": "req-1", "turn_id": "turn-1"}),
        )
        .await;
    assert_eq!(
        socket.expect("tool_result").await,
        json!({"type": "tool_result", "tool_call_id": "t1", "output": "Oslo",
               "request_id": "req-1", "turn_id": "turn-1"})
    );
}

#[tokio::test]
async fn commands_are_sent_on_the_socket() {
    let server = Server::start().await;
    let (session, mut socket) = open(&server, Tools::new()).await;

    session.say("hello").await.unwrap();
    session.interrupt().await.unwrap();

    assert_eq!(
        socket.next().await.unwrap(),
        json!({"type": "say", "text": "hello"})
    );
    assert_eq!(socket.next().await.unwrap(), json!({"type": "interrupt"}));
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
async fn a_socket_that_cannot_open_stops_the_session_it_was_for() {
    let server = Server::start().await;
    server.route(Method::POST, "/v1/agents/sessions", 201, session("s1"));
    server.route(
        Method::POST,
        "/v1/agents/sessions/s1/stop",
        204,
        json!(null),
    );
    server.route(
        Method::GET,
        "/v1/agents/sessions/s1/events",
        403,
        support::refusal("permission", "forbidden", "not yours"),
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
    server.request(Method::POST, "/v1/agents/sessions/s1/stop");
    assert!(
        server
            .requests(Method::DELETE, "/v1/agents/sessions/s1")
            .is_empty()
    );
}

#[tokio::test]
async fn a_conversation_and_its_memories_are_deleted_apart_from_stopping_it() {
    let server = Server::start().await;
    let (session, _socket) = open(&server, Tools::new()).await;
    server.route(Method::DELETE, "/v1/agents/sessions/*", 204, json!(null));
    server.route(Method::DELETE, "/v1/agents/users/*", 204, json!(null));
    let client = server.client();

    session.delete_memories().await.unwrap();
    session.delete().await.unwrap();
    client
        .agent("jean")
        .sessions
        .delete_memories("s2")
        .await
        .unwrap();
    client.agent("jean").sessions.delete("s2").await.unwrap();
    client.memories().truncate("ada").await.unwrap();

    let deleted: Vec<_> = server
        .seen()
        .into_iter()
        .filter(|seen| seen.method == Method::DELETE)
        .map(|seen| seen.path)
        .collect();
    assert_eq!(
        deleted,
        [
            "/v1/agents/sessions/s1/memories",
            "/v1/agents/sessions/s1",
            "/v1/agents/sessions/s2/memories",
            "/v1/agents/sessions/s2",
            "/v1/agents/users/ada/memories",
        ]
    );
    assert!(matches!(
        client.memories().truncate("").await,
        Err(vision_agents::Error::Configuration(_))
    ));
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
        json!({"items": [response("r1", "s1")], "has_more": false}),
    );
    server.route(
        Method::GET,
        "/v1/agents/sessions/s1/responses/items",
        200,
        json!({"items": [item(0, "r1"), item(1, "r1")], "has_more": false}),
    );

    let turn = session.responses.create("what is on today").await.unwrap();
    let listed = session.responses.list(None, None).await.unwrap();
    let items = turn.items.all().await.unwrap();

    assert_eq!(turn.id(), "r1");
    assert_eq!(listed.items.len(), 1);
    assert!(!listed.has_more);
    assert_eq!(items.len(), 2);
    let sent = server
        .request(Method::POST, "/v1/agents/sessions/s1/responses")
        .body;
    assert_eq!(sent["text"], "what is on today");
    assert_eq!(sent.as_object().unwrap().len(), 2);
    assert_eq!(
        server
            .request(Method::GET, "/v1/agents/sessions/s1/responses/items")
            .query,
        "response_id=r1&limit=200"
    );
}

#[tokio::test]
async fn every_text_question_carries_a_fresh_request_id_and_one_with_media_none() {
    let server = Server::start().await;
    let (session, _socket) = open(&server, Tools::new()).await;
    server.route(
        Method::POST,
        "/v1/agents/sessions/s1/responses",
        201,
        response("r1", "s1"),
    );

    session.responses.create("first").await.unwrap();
    session
        .responses
        .create_with(&types::CreateResponseRequest {
            text: "second".into(),
            request_id: Some("mine".into()),
            ..Default::default()
        })
        .await
        .unwrap();
    session
        .responses
        .create_with(&types::CreateResponseRequest {
            text: "what is this?".into(),
            images: Some(vec![types::ImageSource {
                url: "https://example.com/cat.png".into(),
                ..Default::default()
            }]),
            request_id: Some("mine".into()),
            ..Default::default()
        })
        .await
        .unwrap();

    let sent: Vec<_> = server
        .requests(Method::POST, "/v1/agents/sessions/s1/responses")
        .into_iter()
        .map(|seen| seen.body)
        .collect();
    let ids: Vec<&str> = sent[..2]
        .iter()
        .map(|body| body["request_id"].as_str().unwrap())
        .collect();
    for id in &ids {
        assert_eq!(id.len(), 32);
        assert!(id.chars().all(|c| c.is_ascii_hexdigit()));
    }
    assert_ne!(ids[0], ids[1]);
    assert!(sent[2].get("request_id").is_none());
}

#[tokio::test]
async fn a_whole_conversation_is_read_a_page_at_a_time() {
    let server = Server::start().await;
    let page: Vec<_> = (0..200).map(|ordinal| item(ordinal, "r1")).collect();
    server.route(
        Method::GET,
        "/v1/agents/sessions/s1/responses/items",
        200,
        json!({"items": page, "has_more": true, "next_cursor": "c2"}),
    );
    server.route(
        Method::GET,
        "/v1/agents/sessions/s1/responses/items",
        200,
        json!({"items": [item(200, "r2")], "has_more": false}),
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
    assert_eq!(queries, ["limit=200", "limit=200&cursor=c2"]);
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
async fn a_session_is_updated_with_only_what_was_set() {
    let server = Server::start().await;
    let (chat, _socket) = open(&server, weather()).await;
    server.route(Method::PATCH, "/v1/agents/sessions/s1", 200, session("s1"));
    server.route(Method::PATCH, "/v1/agents/sessions/s9", 200, session("s9"));

    let updated = chat
        .update(&types::UpdateSessionRequest {
            llm: Some("llm-thinking".into()),
            thinking: Some(types::UpdateSessionRequestThinking::High),
            ..Default::default()
        })
        .await
        .unwrap();
    let renamed = server
        .client()
        .agent("jean")
        .sessions
        .update(
            "s9",
            &types::UpdateSessionRequest {
                title: Some("Pricing".into()),
                ..Default::default()
            },
        )
        .await
        .unwrap();

    assert_eq!(updated.id, "s1");
    assert_eq!(renamed.id, "s9");
    assert_eq!(
        server.request(Method::PATCH, "/v1/agents/sessions/s1").body,
        json!({"llm": "llm-thinking", "thinking": "high"})
    );
    assert_eq!(
        server.request(Method::PATCH, "/v1/agents/sessions/s9").body,
        json!({"title": "Pricing"})
    );
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
        Method::POST,
        "/v1/agents/sessions/query",
        200,
        json!({"items": [session("s1")], "has_more": true, "next_cursor": "c2"}),
    );
    server.route(Method::POST, "/v1/agents/sessions", 201, session("s3"));
    let agent = server.client().agent("docs");

    let page = agent
        .sessions
        .query(vision_agents::Query {
            project_id: Some("p".into()),
            state: Some("live".into()),
            cursor: Some("c1".into()),
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

    assert_eq!(page.items[0].id, "s1");
    assert_eq!(page.next_cursor.as_deref(), Some("c2"));
    let queries: Vec<_> = server
        .requests(Method::POST, "/v1/agents/sessions/query")
        .into_iter()
        .map(|seen| seen.body)
        .collect();
    assert_eq!(
        queries,
        [
            json!({"filter": {"agent": "docs", "project_id": "p", "state": "live"}, "cursor": "c1"}),
            json!({"filter": {"agent": "docs", "text": {"$q": "pricing"}}}),
        ]
    );
    assert_eq!(
        server.request(Method::POST, "/v1/agents/sessions").body,
        json!({"agent": "docs", "title": "Plans"})
    );
}

#[tokio::test]
async fn a_session_asked_to_start_voice_is_on_its_call() {
    let server = Server::start().await;
    server.route(Method::POST, "/v1/agents/sessions", 201, session("s1"));
    let agent = server.client().agent("docs");

    let opening = tokio::spawn(async move {
        agent
            .sessions
            .create(
                types::CreateSessionRequest {
                    start_voice: Some(true),
                    ..Default::default()
                },
                Tools::new(),
            )
            .await
    });
    let _socket = server.accept().await;
    let session = opening.await.unwrap().unwrap();

    assert_eq!(
        server.request(Method::POST, "/v1/agents/sessions").body,
        json!({"agent": "docs", "start_voice": true})
    );
    assert!(session.voice_started());
}

#[tokio::test]
async fn voice_is_started_and_stopped_on_a_session_held_in_writing() {
    let server = Server::start().await;
    let mut written = session("s1");
    written["call_id"] = json!("");
    server.route(Method::POST, "/v1/agents/sessions", 201, written.clone());
    server.route(
        Method::POST,
        "/v1/agents/sessions/s1/voice",
        200,
        session("s1"),
    );
    server.route(Method::DELETE, "/v1/agents/sessions/s1/voice", 200, written);
    let client = server.client();
    let opening = tokio::spawn(async move {
        Session::open(
            &client,
            Default::default(),
            Tools::new(),
            WatchOptions::default(),
        )
        .await
    });
    let _socket = server.accept().await;
    let session = opening.await.unwrap().unwrap();

    assert!(!session.voice_started());
    let started = session.start_voice().await.unwrap();
    assert!(session.voice_started());
    assert_eq!(started.call_id, "call");
    session.stop_voice().await.unwrap();

    assert!(!session.voice_started());
    assert_eq!(
        server
            .requests(Method::POST, "/v1/agents/sessions/s1/voice")
            .len(),
        1
    );
    assert_eq!(
        server
            .requests(Method::DELETE, "/v1/agents/sessions/s1/voice")
            .len(),
        1
    );
}

#[tokio::test]
async fn a_conversation_is_resumed_by_the_id_of_its_session() {
    let server = Server::start().await;
    server.route(Method::GET, "/v1/agents/sessions/s1", 200, session("s1"));
    let agent = server.client().agent("docs");

    let resuming = tokio::spawn(async move { agent.sessions.resume("s1", Tools::new()).await });
    let socket = server.accept().await;
    let resumed = resuming.await.unwrap().unwrap();

    assert_eq!(resumed.id(), "s1");
    assert_eq!(socket.path, "/v1/agents/sessions/s1/events");
    assert!(
        server
            .requests(Method::POST, "/v1/agents/sessions")
            .is_empty()
    );
}
