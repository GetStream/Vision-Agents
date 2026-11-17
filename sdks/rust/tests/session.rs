mod support;

use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::{Arc, Mutex};
use std::time::Duration;

use axum::http::Method;
use serde_json::{Value, json};
use support::{Accepted, Server, item, refusal, response, session};
use vision_agents::{Error, Query, Session, ToolCallContext, Tools, WatchOptions, types};

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
    for _ in 0..2 {
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
    // Every watcher is sent every call and the first answer wins, so a call for a tool held
    // elsewhere is left for whoever holds it.
    assert!(
        tokio::time::timeout(Duration::from_millis(300), socket.expect("tool_result"))
            .await
            .is_err(),
        "a call for a tool this session does not hold was answered"
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
        .query(Query {
            filter: json!({"project_id": "p"}),
            limit: Some(50),
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
            json!({"filter": {"agent": "docs", "project_id": "p"}, "limit": 50, "cursor": "c1"}),
            json!({"filter": {"agent": "docs", "text": {"$q": "pricing"}}}),
        ]
    );
    assert_eq!(
        server.request(Method::POST, "/v1/agents/sessions").body,
        json!({"agent": "docs", "title": "Plans"})
    );
}

/// The bodies the router saw for the session query, in the order they were sent.
fn queries(server: &Server) -> Vec<serde_json::Value> {
    server
        .requests(Method::POST, "/v1/agents/sessions/query")
        .into_iter()
        .map(|seen| seen.body)
        .collect()
}

fn empty_page(server: &Server) {
    server.route(
        Method::POST,
        "/v1/agents/sessions/query",
        200,
        json!({"items": [], "has_more": false}),
    );
}

fn sorted(field: types::SessionSortField, direction: Option<i64>) -> types::SessionSort {
    types::SessionSort {
        field,
        direction: direction.map(|direction| direction.try_into().unwrap()),
    }
}

#[tokio::test]
async fn a_query_sends_the_filter_as_written_narrowed_to_the_agent() {
    let server = Server::start().await;
    empty_page(&server);
    let sessions = server.client().agent("docs").sessions;

    let filters = [
        serde_json::Value::Null,
        json!({}),
        json!({"project_id": "p"}),
        json!({"$or": [{"project_id": "web"}, {"project_id": "ios"}]}),
        json!({"$and": [
            {"user_id": "u1"},
            {"$or": [{"project_id": "web"}, {"project_id": {"$eq": "ios"}}]},
        ]}),
        json!({"project_id": {"$in": ["web", "ios"]}}),
        json!({"review_score": {"$gte": 4, "$lt": 5}, "user_id": {"$eq": "u1"}}),
        json!({"review_score": 5}),
        json!({"agent": "someone-else", "project_id": "p"}),
    ];
    for filter in filters {
        sessions
            .query(Query {
                filter,
                ..Default::default()
            })
            .await
            .unwrap();
    }

    assert_eq!(
        queries(&server),
        [
            json!({"filter": {"agent": "docs"}}),
            json!({"filter": {"agent": "docs"}}),
            json!({"filter": {"agent": "docs", "project_id": "p"}}),
            json!({"filter": {
                "agent": "docs",
                "$or": [{"project_id": "web"}, {"project_id": "ios"}],
            }}),
            json!({"filter": {
                "agent": "docs",
                "$and": [
                    {"user_id": "u1"},
                    {"$or": [{"project_id": "web"}, {"project_id": {"$eq": "ios"}}]},
                ],
            }}),
            json!({"filter": {"agent": "docs", "project_id": {"$in": ["web", "ios"]}}}),
            json!({"filter": {
                "agent": "docs",
                "review_score": {"$gte": 4, "$lt": 5},
                "user_id": {"$eq": "u1"},
            }}),
            json!({"filter": {"agent": "docs", "review_score": 5}}),
            json!({"filter": {"agent": "docs", "project_id": "p"}}),
        ]
    );
}

#[tokio::test]
async fn a_query_sends_each_sort() {
    use types::SessionSortField::{LastMessageAt, Pinned, Relevance, ReviewScore};
    let server = Server::start().await;
    empty_page(&server);
    let sessions = server.client().agent("docs").sessions;

    let sorts = [
        vec![sorted(LastMessageAt, None)],
        vec![sorted(LastMessageAt, Some(1))],
        vec![sorted(Pinned, None)],
        vec![sorted(Pinned, Some(-1)), sorted(LastMessageAt, Some(1))],
        vec![sorted(ReviewScore, Some(-1))],
        vec![sorted(ReviewScore, Some(1))],
    ];
    for sort in sorts {
        sessions
            .query(Query {
                sort,
                ..Default::default()
            })
            .await
            .unwrap();
    }
    sessions
        .search(
            "lisbon",
            Query {
                sort: vec![sorted(Relevance, Some(-1))],
                ..Default::default()
            },
        )
        .await
        .unwrap();

    let agent = json!({"agent": "docs"});
    assert_eq!(
        queries(&server),
        [
            json!({"filter": agent, "sort": [{"field": "last_message_at"}]}),
            json!({"filter": agent, "sort": [{"field": "last_message_at", "direction": 1}]}),
            json!({"filter": agent, "sort": [{"field": "pinned"}]}),
            json!({"filter": agent, "sort": [
                {"field": "pinned", "direction": -1},
                {"field": "last_message_at", "direction": 1},
            ]}),
            json!({"filter": agent, "sort": [{"field": "review_score", "direction": -1}]}),
            json!({"filter": agent, "sort": [{"field": "review_score", "direction": 1}]}),
            json!({
                "filter": {"agent": "docs", "text": {"$q": "lisbon"}},
                "sort": [{"field": "relevance", "direction": -1}],
            }),
        ]
    );
}

#[tokio::test]
async fn a_search_takes_the_filter_and_pages_like_a_query() {
    let server = Server::start().await;
    server.route(
        Method::POST,
        "/v1/agents/sessions/query",
        200,
        json!({"items": [session("s1")], "has_more": true, "next_cursor": "c2"}),
    );
    let sessions = server.client().agent("docs").sessions;

    let first = sessions
        .search(
            "pricing",
            Query {
                filter: json!({"project_id": "p"}),
                limit: Some(10),
                ..Default::default()
            },
        )
        .await
        .unwrap();
    sessions
        .search(
            "pricing",
            Query {
                filter: json!({"project_id": "p"}),
                limit: Some(10),
                cursor: first.next_cursor.clone(),
                ..Default::default()
            },
        )
        .await
        .unwrap();

    assert!(first.has_more);
    let filter = json!({"agent": "docs", "project_id": "p", "text": {"$q": "pricing"}});
    assert_eq!(
        queries(&server),
        [
            json!({"filter": filter, "limit": 10}),
            json!({"filter": filter, "limit": 10, "cursor": "c2"}),
        ]
    );
}

#[tokio::test]
async fn a_filter_the_router_could_not_read_is_refused_before_a_request() {
    let server = Server::start().await;
    let sessions = server.client().agent("docs").sessions;

    let not_an_object = sessions
        .query(Query {
            filter: json!(["project_id", "p"]),
            ..Default::default()
        })
        .await;
    let two_texts = sessions
        .search(
            "pricing",
            Query {
                filter: json!({"text": {"$q": "plans"}}),
                ..Default::default()
            },
        )
        .await;

    assert!(matches!(not_an_object, Err(Error::Configuration(_))));
    assert!(matches!(two_texts, Err(Error::Configuration(_))));
    assert!(server.seen().is_empty());
}

#[tokio::test]
async fn a_session_is_pinned_and_unpinned_and_says_when() {
    let server = Server::start().await;
    let (chat, _socket) = open(&server, weather()).await;
    let mut pinned = session("s9");
    pinned["pinned_at"] = json!("2026-11-11T09:00:00Z");
    server.route(
        Method::POST,
        "/v1/agents/sessions/query",
        200,
        json!({"items": [pinned, session("s8")], "has_more": false}),
    );
    server.route(Method::PATCH, "/v1/agents/sessions/s9", 200, pinned.clone());
    server.route(Method::PATCH, "/v1/agents/sessions/s1", 200, session("s1"));
    let sessions = server.client().agent("jean").sessions;

    let by_id = sessions.pin("s9").await.unwrap();
    sessions.unpin("s9").await.unwrap();
    sessions
        .update(
            "s9",
            &types::UpdateSessionRequest {
                pinned: Some(true),
                title: Some("Pricing".into()),
                ..Default::default()
            },
        )
        .await
        .unwrap();
    chat.pin().await.unwrap();
    let unpinned = chat.unpin().await.unwrap();
    let page = sessions.query(Query::default()).await.unwrap();

    assert_eq!(by_id.pinned_at.as_deref(), Some("2026-11-11T09:00:00Z"));
    assert_eq!(unpinned.pinned_at, None);
    assert_eq!(
        page.items[0].pinned_at.as_deref(),
        Some("2026-11-11T09:00:00Z")
    );
    assert_eq!(page.items[1].pinned_at, None);
    let bodies = |path: &str| -> Vec<_> {
        server
            .requests(Method::PATCH, path)
            .into_iter()
            .map(|seen| seen.body)
            .collect()
    };
    assert_eq!(
        bodies("/v1/agents/sessions/s9"),
        [
            json!({"pinned": true}),
            json!({"pinned": false}),
            json!({"pinned": true, "title": "Pricing"}),
        ]
    );
    assert_eq!(
        bodies("/v1/agents/sessions/s1"),
        [json!({"pinned": true}), json!({"pinned": false})]
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

fn tool(name: &str) -> types::SessionTool {
    types::SessionTool {
        name: name.into(),
        description: format!("{name} for the test"),
        ..Default::default()
    }
}

/// Registers a function that is handed its call, as a tool the person cannot interrupt.
fn holding() -> Tools {
    let tools = Tools::new();
    tools
        .register_with_tool_call_context(
            tool("pay"),
            |tool_call_context: ToolCallContext, _| async move {
                tool_call_context.disallow_interruptions().await?;
                tokio::time::sleep(Duration::from_millis(200)).await;
                Ok::<_, Error>("paid")
            },
        )
        .unwrap();
    tools.register("fast", "Does not take long", json!({}), |_| async {
        Ok::<_, String>("done")
    });
    tools
}

#[tokio::test]
async fn a_function_is_told_who_its_call_answers_and_can_say_how_it_is_going() {
    let server = Server::start().await;
    let tools = Tools::new();
    tools
        .register_with_tool_call_context(tool("book"), |tool_call_context: ToolCallContext, _| async move {
            tool_call_context.update("Looking at flights")
                .await
                .map_err(|error| error.to_string())?;
            tool_call_context.say("One moment.")
                .await
                .map_err(|error| error.to_string())?;
            Ok::<_, String>(json!({
                "id": tool_call_context.id, "session": tool_call_context.session_id, "turn": tool_call_context.turn_id,
                "user": [tool_call_context.user.id, tool_call_context.user.name], "tier": tool_call_context.user.custom["tier"],
                "cid": tool_call_context.conversation.cid, "approved": tool_call_context.approved,
            }))
        })
        .unwrap();
    let (_session, mut socket) = open(&server, tools).await;

    socket
        .send(
            json!({"type": "tool_call", "id": "t1", "name": "book", "arguments": "{}",
                     "turn_id": "turn-1", "approved": true,
                     "user": {"id": "u1", "name": "Ada", "custom": {"tier": "gold"}},
                     "conversation": {"cid": "messaging:c1", "custom": {}}}),
        )
        .await;

    assert_eq!(
        socket.next().await.unwrap(),
        json!({"type": "tool_update", "tool_call_id": "t1", "text": "Looking at flights", "verbatim": false})
    );
    assert_eq!(
        socket.next().await.unwrap(),
        json!({"type": "tool_update", "tool_call_id": "t1", "text": "One moment.", "verbatim": true})
    );
    let output: serde_json::Value = serde_json::from_str(
        socket.expect("tool_result").await["output"]
            .as_str()
            .unwrap(),
    )
    .unwrap();
    assert_eq!(
        output,
        json!({"id": "t1", "session": "s1", "turn": "turn-1", "user": ["u1", "Ada"],
               "tier": "gold", "cid": "messaging:c1", "approved": true})
    );
}

#[tokio::test]
async fn a_call_the_router_runs_elsewhere_is_left_alone() {
    let server = Server::start().await;
    let (_session, mut socket) = open(&server, holding()).await;

    socket
        .send(json!({"type": "tool_call", "id": "f1", "name": "fast", "arguments": "{}", "run": false}))
        .await;
    socket
        .send(json!({"type": "tool_call", "id": "f2", "name": "fast", "arguments": "{}"}))
        .await;

    assert_eq!(socket.expect("tool_result").await["tool_call_id"], "f2");
}

#[tokio::test]
async fn a_held_call_is_not_cancelled_by_the_person_talking_over_the_agent() {
    let server = Server::start().await;
    let (session, mut socket) = open(&server, holding()).await;

    socket
        .send(json!({"type": "tool_call", "id": "p1", "name": "pay", "arguments": "{}"}))
        .await;
    assert_eq!(
        socket.next().await.unwrap(),
        json!({"type": "tool_hold", "tool_call_id": "p1"})
    );
    socket.send(json!({"type": "tool_held", "id": "p1"})).await;
    socket
        .send(json!({"type": "tool_cancel", "id": "p1"}))
        .await;

    assert_eq!(
        socket.expect("tool_result").await,
        json!({"type": "tool_result", "tool_call_id": "p1", "output": "paid"})
    );
    assert_eq!(session.next_event().await.unwrap().kind, "tool_held");
}

#[tokio::test]
async fn a_hold_asked_for_too_late_is_cancelled_with_the_call() {
    let server = Server::start().await;
    let (_session, mut socket) = open(&server, holding()).await;

    socket
        .send(json!({"type": "tool_call", "id": "p1", "name": "pay", "arguments": "{}"}))
        .await;
    socket.expect("tool_hold").await;
    socket
        .send(json!({"type": "tool_cancel", "id": "p1"}))
        .await;
    socket
        .send(json!({"type": "tool_call", "id": "f1", "name": "fast", "arguments": "{}"}))
        .await;

    assert_eq!(socket.expect("tool_result").await["tool_call_id"], "f1");
    assert!(
        tokio::time::timeout(Duration::from_millis(400), socket.expect("tool_result"))
            .await
            .is_err(),
        "a cancelled call was answered"
    );
}

#[tokio::test]
async fn a_call_that_outlasts_its_timeout_is_stopped_and_fails() {
    let server = Server::start().await;
    let tools = Tools::new();
    tools
        .register_with_tool_call_context(
            types::SessionTool {
                timeout_ms: Some(50),
                ..tool("slow")
            },
            |_, _| async {
                tokio::time::sleep(Duration::from_secs(30)).await;
                Ok::<_, String>("done")
            },
        )
        .unwrap();
    let (_session, mut socket) = open(&server, tools).await;

    socket
        .send(json!({"type": "tool_call", "id": "s1", "name": "slow", "arguments": "{}"}))
        .await;

    assert_eq!(
        socket.expect("tool_result").await,
        json!({"type": "tool_result", "tool_call_id": "s1", "error": "slow took longer than 50 ms"})
    );
}

#[tokio::test]
async fn a_tool_is_declared_with_only_the_policies_it_sets() {
    let server = Server::start().await;
    let tools = Tools::new();
    tools
        .register_with_tool_call_context(
            types::SessionTool {
                interruptible: Some(true),
                timeout_ms: Some(5000),
                approval: Some(types::SessionToolApproval {
                    title: "Refund {amount}?".into(),
                    ..Default::default()
                }),
                parameters: json!({"type": "object", "properties": {"amount": {"type": "number"}}})
                    .as_object()
                    .unwrap()
                    .clone(),
                ..tool("refund")
            },
            |_, _| async { Ok::<_, String>("refunded") },
        )
        .unwrap();
    let (_session, _socket) = open(&server, tools).await;

    let sent = server.request(Method::POST, "/v1/agents/sessions").body;
    assert_eq!(sent["tools"][0]["interruptible"], true);
    assert_eq!(sent["tools"][0]["timeout_ms"], 5000);
    assert_eq!(
        sent["tools"][0]["approval"],
        json!({"title": "Refund {amount}?"})
    );
}

#[test]
fn a_policy_that_cannot_work_is_refused_when_it_is_registered() {
    let tools = Tools::new();
    let refused = |declaration: types::SessionTool| {
        tools
            .register_with_tool_call_context(declaration, |_, _| async { Ok::<_, String>("") })
            .err()
            .map(|error| error.to_string())
    };
    let asking = |title: &str| types::SessionTool {
        approval: Some(types::SessionToolApproval {
            title: title.into(),
            ..Default::default()
        }),
        parameters: json!({"type": "object", "properties": {"amount": {}}})
            .as_object()
            .unwrap()
            .clone(),
        ..tool("refund")
    };

    assert_eq!(
        refused(asking("Refund {amount} to {card}?")).as_deref(),
        Some("refund's approval names {card}, which is not one of its parameters")
    );
    assert_eq!(
        refused(asking("  ")).as_deref(),
        Some("refund's approval question cannot be blank")
    );
    assert_eq!(
        refused(types::SessionTool {
            timeout_ms: Some(-1),
            ..tool("refund")
        })
        .as_deref(),
        Some("refund has a negative timeout_ms")
    );
    assert_eq!(
        refused(asking("Refund {amount}? {not a placeholder}")),
        None
    );
    assert!(tools.has("refund"));
}

const SESSION_TOOLS: &str = "/v1/agents/sessions/s1/tools";
const RESERVE_TABLE: &str = "/v1/agents/sessions/s1/tools/reserve_table";

/// The ids of the calls a function made by [`answering`] was run for, in every test.
static RAN: Mutex<Vec<String>> = Mutex::new(Vec::new());

/// Numbers the calls [`answer`] makes, so that no two tests ask with the same id.
static ASKED: AtomicUsize = AtomicUsize::new(0);

/// A registry with one function, `name`, that answers `by` and puts the id of each call it is
/// run for in [`RAN`] as it starts.
fn answering(name: &str, by: &'static str) -> Tools {
    let tools = Tools::new();
    tools
        .register_with_tool_call_context(
            types::SessionTool {
                name: name.into(),
                description: format!("{name} for the test"),
                parameters: json!({"type": "object"}).as_object().unwrap().clone(),
                ..Default::default()
            },
            move |call: ToolCallContext, _| {
                RAN.lock().unwrap().push(call.id);
                async move { Ok::<_, String>(by) }
            },
        )
        .unwrap();
    tools
}

/// `tools` with `fast` beside them, which [`answer`] calls after the call it asks about.
fn probed(tools: Tools) -> Tools {
    tools.register("fast", "Answers at once", json!({}), |_| async {
        Ok::<_, String>("done")
    });
    tools
}

/// What the session answers a call of `name` with now, or `None` when it leaves the call to
/// another watcher: a call of `fast` sent after it is then answered first, and no function
/// made by [`answering`] was run for it, however late it would have answered.
async fn answer(socket: &mut Accepted, name: &str) -> Option<Value> {
    let asked = format!("asked-{}", ASKED.fetch_add(1, Ordering::Relaxed));
    socket
        .send(json!({"type": "tool_call", "id": asked, "name": name, "arguments": "{}"}))
        .await;
    socket
        .send(json!({"type": "tool_call", "id": "probe", "name": "fast", "arguments": "{}"}))
        .await;
    let first = socket.expect("tool_result").await;
    if first["tool_call_id"] == "probe" {
        // Each call runs on a task of its own, started in the order the calls came on the
        // test's one thread, so a function run for this one had started before the probe
        // answered.
        assert!(
            !RAN.lock().unwrap().contains(&asked),
            "{name} was run, and has not answered yet"
        );
        return None;
    }
    assert_eq!(socket.expect("tool_result").await["tool_call_id"], "probe");
    Some(first["output"].clone())
}

/// The changes to the sessions' tools the router was asked for, in order.
fn tool_changes(server: &Server) -> Vec<String> {
    server
        .seen()
        .into_iter()
        .filter(|seen| seen.path.contains("/tools"))
        .map(|seen| format!("{} {}", seen.method, seen.path))
        .collect()
}

#[tokio::test]
async fn a_tool_added_while_the_session_runs_is_declared_and_run_here() {
    let server = Server::start().await;
    let (session, mut socket) = open(&server, probed(Tools::new())).await;
    server.route(Method::POST, SESSION_TOOLS, 204, Value::Null);

    session
        .add_tools(answering("reserve_table", "added"))
        .await
        .unwrap();

    assert_eq!(
        server.request(Method::POST, SESSION_TOOLS).body,
        json!({"tools": [{"name": "reserve_table", "description": "reserve_table for the test",
                          "parameters": {"type": "object"}}]})
    );
    assert_eq!(
        answer(&mut socket, "reserve_table").await,
        Some(json!("added"))
    );
}

#[tokio::test]
async fn a_tool_the_router_refused_is_not_run_here() {
    let server = Server::start().await;
    let (session, mut socket) = open(&server, probed(Tools::new())).await;
    server.route(
        Method::POST,
        SESSION_TOOLS,
        409,
        refusal(
            "conflict",
            "conflict",
            "the session offers reserve_table from elsewhere",
        ),
    );

    let refused = session
        .add_tools(answering("reserve_table", "added"))
        .await
        .unwrap_err();

    assert_eq!(refused.status(), Some(409));
    assert_eq!(answer(&mut socket, "reserve_table").await, None);
}

#[tokio::test]
async fn a_removed_tool_is_no_longer_run_here() {
    let server = Server::start().await;
    let (session, mut socket) = open(&server, probed(Tools::new())).await;
    server
        .route(Method::POST, SESSION_TOOLS, 204, Value::Null)
        .route(Method::DELETE, RESERVE_TABLE, 204, Value::Null);
    session
        .add_tools(answering("reserve_table", "added"))
        .await
        .unwrap();

    session.remove_tool("reserve_table").await.unwrap();

    server.request(Method::DELETE, RESERVE_TABLE);
    assert_eq!(answer(&mut socket, "reserve_table").await, None);
}

#[tokio::test]
async fn a_name_the_session_already_runs_keeps_its_function() {
    let server = Server::start().await;
    let (session, mut socket) = open(&server, probed(answering("reserve_table", "own"))).await;
    // The second add's answer is lost, which leaves its fate unknown here; the router never
    // replaces a tool the session has.
    server
        .route(Method::POST, SESSION_TOOLS, 204, Value::Null)
        .route(Method::POST, SESSION_TOOLS, 0, Value::Null);
    session
        .add_tools(answering("book", "first booking"))
        .await
        .unwrap();
    let again = answering("reserve_table", "added");
    again.register("book", "Books again", json!({}), |_| async {
        Ok::<_, String>("second booking")
    });

    let lost = session.add_tools(again).await.unwrap_err();

    assert!(matches!(lost, Error::Transport { .. }), "{lost:?}");
    assert_eq!(
        answer(&mut socket, "reserve_table").await,
        Some(json!("own"))
    );
    assert_eq!(
        answer(&mut socket, "book").await,
        Some(json!("first booking"))
    );
}

#[tokio::test]
async fn a_tool_withdrawn_from_the_sessions_own_runs_the_one_added_in_its_place() {
    let server = Server::start().await;
    let (session, mut socket) = open(&server, probed(answering("reserve_table", "own"))).await;
    server
        .route(Method::DELETE, RESERVE_TABLE, 204, Value::Null)
        .route(Method::POST, SESSION_TOOLS, 204, Value::Null);

    session.remove_tool("reserve_table").await.unwrap();
    assert_eq!(answer(&mut socket, "reserve_table").await, None);
    session
        .add_tools(answering("reserve_table", "replacement"))
        .await
        .unwrap();

    assert_eq!(
        answer(&mut socket, "reserve_table").await,
        Some(json!("replacement"))
    );
}

#[tokio::test]
async fn a_tool_whose_adding_was_never_answered_is_withdrawn_before_its_name_is_added_again() {
    let server = Server::start().await;
    let (session, mut socket) = open(&server, probed(Tools::new())).await;
    server
        .route(Method::POST, SESSION_TOOLS, 0, Value::Null)
        .route(Method::POST, SESSION_TOOLS, 204, Value::Null)
        // The lost add never reached the session, so it has no tool by the name to withdraw.
        .route(
            Method::DELETE,
            RESERVE_TABLE,
            404,
            refusal(
                "not_found",
                "not_found",
                "the session's caller has no tool by that name",
            ),
        );
    let lost = session
        .add_tools(answering("reserve_table", "first"))
        .await
        .unwrap_err();
    assert!(matches!(lost, Error::Transport { .. }), "{lost:?}");
    // The router may have added it, so its calls are still run here.
    assert_eq!(
        answer(&mut socket, "reserve_table").await,
        Some(json!("first"))
    );

    session
        .add_tools(answering("reserve_table", "second"))
        .await
        .unwrap();

    assert_eq!(
        tool_changes(&server),
        [
            format!("POST {SESSION_TOOLS}"),
            format!("DELETE {RESERVE_TABLE}"),
            format!("POST {SESSION_TOOLS}"),
        ]
    );
    assert_eq!(
        answer(&mut socket, "reserve_table").await,
        Some(json!("second"))
    );
}

#[tokio::test]
async fn a_tool_the_router_has_none_by_is_withdrawn_only_when_its_adding_was_never_answered() {
    let server = Server::start().await;
    let (session, mut socket) = open(&server, probed(Tools::new())).await;
    server
        .route(Method::POST, SESSION_TOOLS, 0, Value::Null)
        .route(
            Method::DELETE,
            RESERVE_TABLE,
            404,
            refusal(
                "not_found",
                "not_found",
                "the session's caller has no tool by that name",
            ),
        );

    let absent = session.remove_tool("reserve_table").await.unwrap_err();
    assert_eq!(absent.status(), Some(404));
    session
        .add_tools(answering("reserve_table", "lost"))
        .await
        .unwrap_err();
    session.remove_tool("reserve_table").await.unwrap();

    assert_eq!(answer(&mut socket, "reserve_table").await, None);
}

#[tokio::test]
async fn changes_to_a_sessions_tools_are_made_one_at_a_time() {
    let server = Server::start().await;
    let (session, _socket) = open(&server, probed(Tools::new())).await;
    server
        .route(Method::POST, SESSION_TOOLS, 0, Value::Null)
        .route(Method::POST, SESSION_TOOLS, 204, Value::Null)
        .route(
            Method::DELETE,
            RESERVE_TABLE,
            404,
            refusal(
                "not_found",
                "not_found",
                "the session's caller has no tool by that name",
            ),
        );
    session
        .add_tools(answering("reserve_table", "lost"))
        .await
        .unwrap_err();

    // Two at once would both withdraw the unsettled name, and one could take back what the
    // other just added.
    let _ = tokio::join!(
        session.add_tools(answering("reserve_table", "one")),
        session.add_tools(answering("reserve_table", "other")),
    );

    assert_eq!(
        tool_changes(&server),
        [
            format!("POST {SESSION_TOOLS}"),
            format!("DELETE {RESERVE_TABLE}"),
            format!("POST {SESSION_TOOLS}"),
            format!("POST {SESSION_TOOLS}"),
        ]
    );
}

#[tokio::test]
async fn adding_no_tools_is_refused_before_a_request() {
    let server = Server::start().await;
    let (session, _socket) = open(&server, probed(Tools::new())).await;

    let refused = session.add_tools(Tools::new()).await.unwrap_err();

    assert!(matches!(refused, Error::Configuration(_)), "{refused:?}");
    assert!(tool_changes(&server).is_empty());
}

#[tokio::test]
async fn a_closed_session_lets_go_of_the_functions_it_was_given() {
    let server = Server::start().await;
    let (session, mut socket) = open(&server, probed(Tools::new())).await;
    server.route(Method::POST, SESSION_TOOLS, 204, Value::Null);
    let resource = Arc::new(());
    let held = Arc::downgrade(&resource);
    let tools = Tools::new();
    tools.register("reserve_table", "Reserve a table", json!({}), move |_| {
        let _resource = resource.clone();
        async { Ok::<_, String>("reserved") }
    });
    session.add_tools(tools).await.unwrap();

    let router = tokio::spawn(async move {
        socket.expect("close").await;
        socket.close().await;
    });
    session.close().await;
    router.await.unwrap();

    assert!(
        held.upgrade().is_none(),
        "a closed session still holds the functions it was given"
    );
}

/// The router's answer to withdrawing a tool by a name the session's caller has none by.
fn no_tool_by_that_name() -> Value {
    refusal(
        "not_found",
        "not_found",
        "the session's caller has no tool by that name",
    )
}

/// The router's answer to a change of the tools by a caller of another kind than the one that
/// opened the session.
fn another_kind_of_caller() -> Value {
    refusal(
        "permission",
        "permission",
        "a session's tools are declared by the kind of caller that opened it",
    )
}

/// What a gateway in front of the router answers when the router's own answer does not get
/// back to it, whether or not the router made the change.
const BAD_GATEWAY: &str = "<html><body><h1>502 Bad Gateway</h1></body></html>";

/// How long a caller waits for a change of the tools before giving up on it.
const GIVING_UP: Duration = Duration::from_millis(100);

#[tokio::test]
async fn a_tool_whose_adding_was_given_up_on_can_be_removed_though_the_router_has_none_by_it() {
    let server = Server::start().await;
    let (session, mut socket) = open(&server, probed(Tools::new())).await;
    server
        .route(Method::POST, SESSION_TOOLS, 1, Value::Null)
        .route(Method::DELETE, RESERVE_TABLE, 404, no_tool_by_that_name());
    let adding = session.add_tools(answering("reserve_table", "added"));
    tokio::time::timeout(GIVING_UP, adding).await.unwrap_err();
    assert_eq!(tool_changes(&server), [format!("POST {SESSION_TOOLS}")]);

    session.remove_tool("reserve_table").await.unwrap();

    assert_eq!(answer(&mut socket, "reserve_table").await, None);
}

#[tokio::test]
async fn a_tool_whose_adding_was_given_up_on_is_withdrawn_before_its_name_is_added_again() {
    let server = Server::start().await;
    let (session, mut socket) = open(&server, probed(Tools::new())).await;
    server
        .route(Method::POST, SESSION_TOOLS, 1, Value::Null)
        .route(Method::POST, SESSION_TOOLS, 204, Value::Null)
        .route(Method::DELETE, RESERVE_TABLE, 404, no_tool_by_that_name());
    let adding = session.add_tools(answering("reserve_table", "first"));
    tokio::time::timeout(GIVING_UP, adding).await.unwrap_err();
    assert_eq!(tool_changes(&server), [format!("POST {SESSION_TOOLS}")]);
    // The router may have added it, so its calls are still run here.
    assert_eq!(
        answer(&mut socket, "reserve_table").await,
        Some(json!("first"))
    );

    session
        .add_tools(answering("reserve_table", "second"))
        .await
        .unwrap();

    assert_eq!(
        answer(&mut socket, "reserve_table").await,
        Some(json!("second"))
    );
}

#[tokio::test]
async fn a_tool_whose_removal_was_never_answered_can_be_removed_though_the_router_has_none_by_it() {
    let server = Server::start().await;
    let (session, mut socket) = open(&server, probed(answering("reserve_table", "own"))).await;
    // The router withdraws it, and its answer is lost on the way.
    server
        .route(Method::DELETE, RESERVE_TABLE, 0, Value::Null)
        .route(Method::DELETE, RESERVE_TABLE, 404, no_tool_by_that_name());
    let lost = session.remove_tool("reserve_table").await.unwrap_err();
    assert!(matches!(lost, Error::Transport { .. }), "{lost:?}");
    // The router may not have withdrawn it, so its calls are still run here.
    assert_eq!(
        answer(&mut socket, "reserve_table").await,
        Some(json!("own"))
    );

    session.remove_tool("reserve_table").await.unwrap();

    assert_eq!(answer(&mut socket, "reserve_table").await, None);
}

#[tokio::test]
async fn a_tool_whose_removal_was_never_answered_is_withdrawn_before_its_name_is_added_again() {
    let server = Server::start().await;
    let (session, mut socket) = open(&server, probed(answering("reserve_table", "own"))).await;
    server
        .route(Method::DELETE, RESERVE_TABLE, 0, Value::Null)
        .route(Method::DELETE, RESERVE_TABLE, 404, no_tool_by_that_name())
        .route(Method::POST, SESSION_TOOLS, 204, Value::Null);
    session.remove_tool("reserve_table").await.unwrap_err();

    session
        .add_tools(answering("reserve_table", "replacement"))
        .await
        .unwrap();

    // The lost removal was sent once, never again on its own.
    assert_eq!(
        tool_changes(&server),
        [
            format!("DELETE {RESERVE_TABLE}"),
            format!("DELETE {RESERVE_TABLE}"),
            format!("POST {SESSION_TOOLS}"),
        ]
    );
    assert_eq!(
        answer(&mut socket, "reserve_table").await,
        Some(json!("replacement"))
    );
}

#[tokio::test]
async fn a_tool_whose_removal_was_given_up_on_can_be_removed_though_the_router_has_none_by_it() {
    let server = Server::start().await;
    let (session, mut socket) = open(&server, probed(answering("reserve_table", "own"))).await;
    server
        .route(Method::DELETE, RESERVE_TABLE, 1, Value::Null)
        .route(Method::DELETE, RESERVE_TABLE, 404, no_tool_by_that_name());
    let removing = session.remove_tool("reserve_table");
    tokio::time::timeout(GIVING_UP, removing).await.unwrap_err();
    assert_eq!(tool_changes(&server), [format!("DELETE {RESERVE_TABLE}")]);
    assert_eq!(
        answer(&mut socket, "reserve_table").await,
        Some(json!("own"))
    );

    session.remove_tool("reserve_table").await.unwrap();

    assert_eq!(answer(&mut socket, "reserve_table").await, None);
}

#[tokio::test]
async fn a_tool_whose_removal_was_given_up_on_is_withdrawn_before_its_name_is_added_again() {
    let server = Server::start().await;
    let (session, mut socket) = open(&server, probed(answering("reserve_table", "own"))).await;
    server
        .route(Method::DELETE, RESERVE_TABLE, 1, Value::Null)
        .route(Method::DELETE, RESERVE_TABLE, 404, no_tool_by_that_name())
        .route(Method::POST, SESSION_TOOLS, 204, Value::Null);
    let removing = session.remove_tool("reserve_table");
    tokio::time::timeout(GIVING_UP, removing).await.unwrap_err();
    assert_eq!(tool_changes(&server), [format!("DELETE {RESERVE_TABLE}")]);

    session
        .add_tools(answering("reserve_table", "replacement"))
        .await
        .unwrap();

    assert_eq!(
        answer(&mut socket, "reserve_table").await,
        Some(json!("replacement"))
    );
}

#[tokio::test]
async fn a_removal_the_router_refused_leaves_the_tool_as_it_was() {
    let server = Server::start().await;
    let (session, mut socket) = open(&server, probed(answering("reserve_table", "own"))).await;
    server
        .route(Method::DELETE, RESERVE_TABLE, 403, another_kind_of_caller())
        .route(Method::DELETE, RESERVE_TABLE, 204, Value::Null)
        .route(
            Method::POST,
            SESSION_TOOLS,
            409,
            refusal("conflict", "conflict", "the session has reserve_table"),
        );
    let refused = session.remove_tool("reserve_table").await.unwrap_err();
    assert_eq!(refused.status(), Some(403));

    // Still the session's: adding it again is refused rather than withdrawn on the way.
    let again = session
        .add_tools(answering("reserve_table", "added"))
        .await
        .unwrap_err();

    assert_eq!(again.status(), Some(409));
    assert_eq!(
        tool_changes(&server),
        [
            format!("DELETE {RESERVE_TABLE}"),
            format!("POST {SESSION_TOOLS}"),
        ]
    );
    assert_eq!(
        answer(&mut socket, "reserve_table").await,
        Some(json!("own"))
    );
}

#[tokio::test]
async fn a_tool_whose_adding_was_never_answered_stays_unsettled_when_its_removal_is_refused() {
    let server = Server::start().await;
    let (session, mut socket) = open(&server, probed(Tools::new())).await;
    server
        .route(Method::POST, SESSION_TOOLS, 0, Value::Null)
        .route(Method::DELETE, RESERVE_TABLE, 403, another_kind_of_caller())
        .route(Method::DELETE, RESERVE_TABLE, 404, no_tool_by_that_name());
    session
        .add_tools(answering("reserve_table", "lost"))
        .await
        .unwrap_err();
    let refused = session.remove_tool("reserve_table").await.unwrap_err();
    assert_eq!(refused.status(), Some(403));

    session.remove_tool("reserve_table").await.unwrap();

    assert_eq!(answer(&mut socket, "reserve_table").await, None);
}

#[tokio::test]
async fn registering_in_the_tools_given_to_a_session_afterwards_changes_nothing_there() {
    let server = Server::start().await;
    let (session, mut socket) = open(&server, probed(Tools::new())).await;
    server.route(Method::POST, SESSION_TOOLS, 204, Value::Null);
    let tools = answering("reserve_table", "added");
    session.add_tools(tools.clone()).await.unwrap();

    tools.register(
        "reserve_table",
        "Reserves another way",
        json!({}),
        |_| async { Ok::<_, String>("registered after") },
    );

    assert_eq!(
        answer(&mut socket, "reserve_table").await,
        Some(json!("added"))
    );
}

#[tokio::test]
async fn a_fork_runs_the_tools_the_session_has_when_it_is_forked() {
    let server = Server::start().await;
    let (parent, _parent_socket) = open(&server, probed(answering("reserve_table", "own"))).await;
    server
        .route(Method::POST, SESSION_TOOLS, 204, Value::Null)
        .route(Method::DELETE, RESERVE_TABLE, 204, Value::Null)
        .route(
            Method::POST,
            "/v1/agents/sessions/s1/fork",
            201,
            session("s2"),
        );
    parent.add_tools(answering("book", "added")).await.unwrap();
    parent.remove_tool("reserve_table").await.unwrap();

    let _fork = parent.fork(&Default::default()).await.unwrap();
    let mut socket = server.accept().await;

    assert_eq!(socket.path, "/v1/agents/sessions/s2/events");
    assert_eq!(answer(&mut socket, "book").await, Some(json!("added")));
    assert_eq!(answer(&mut socket, "reserve_table").await, None);
}

#[tokio::test]
async fn a_tool_whose_adding_a_gateway_failed_is_withdrawn_before_its_name_is_added_again() {
    let server = Server::start().await;
    let (session, mut socket) = open(&server, probed(Tools::new())).await;
    // The router adds it, and the gateway in front of it answers 502.
    server
        .route_raw(Method::POST, SESSION_TOOLS, 502, &[], BAD_GATEWAY)
        .route(Method::POST, SESSION_TOOLS, 204, Value::Null)
        .route(Method::DELETE, RESERVE_TABLE, 204, Value::Null);
    let failed = session
        .add_tools(answering("reserve_table", "first"))
        .await
        .unwrap_err();
    assert_eq!(failed.status(), Some(502));
    // Not a refusal: the router may have added it, so its calls are still run here.
    assert_eq!(
        answer(&mut socket, "reserve_table").await,
        Some(json!("first"))
    );

    session
        .add_tools(answering("reserve_table", "second"))
        .await
        .unwrap();

    assert_eq!(
        tool_changes(&server),
        [
            format!("POST {SESSION_TOOLS}"),
            format!("DELETE {RESERVE_TABLE}"),
            format!("POST {SESSION_TOOLS}"),
        ]
    );
    assert_eq!(
        answer(&mut socket, "reserve_table").await,
        Some(json!("second"))
    );
}

#[tokio::test]
async fn a_tool_whose_removal_a_gateway_failed_is_withdrawn_before_its_name_is_added_again() {
    let server = Server::start().await;
    let (session, mut socket) = open(&server, probed(answering("reserve_table", "own"))).await;
    // The router withdraws it, and the gateway in front of it answers 502.
    server
        .route_raw(Method::DELETE, RESERVE_TABLE, 502, &[], BAD_GATEWAY)
        .route(Method::DELETE, RESERVE_TABLE, 404, no_tool_by_that_name())
        .route(Method::POST, SESSION_TOOLS, 204, Value::Null);
    let failed = session.remove_tool("reserve_table").await.unwrap_err();
    assert_eq!(failed.status(), Some(502));
    // Not a refusal: the router may not have withdrawn it, so its calls are still run here.
    assert_eq!(
        answer(&mut socket, "reserve_table").await,
        Some(json!("own"))
    );

    session
        .add_tools(answering("reserve_table", "replacement"))
        .await
        .unwrap();

    assert_eq!(
        tool_changes(&server),
        [
            format!("DELETE {RESERVE_TABLE}"),
            format!("DELETE {RESERVE_TABLE}"),
            format!("POST {SESSION_TOOLS}"),
        ]
    );
    assert_eq!(
        answer(&mut socket, "reserve_table").await,
        Some(json!("replacement"))
    );
}

#[tokio::test]
async fn a_fork_withdraws_a_name_whose_adding_was_never_answered_before_adding_it_again() {
    let server = Server::start().await;
    let (parent, _parent_socket) = open(&server, probed(Tools::new())).await;
    // The parent's add never reached the router, so neither session has the tool.
    server
        .route(Method::POST, SESSION_TOOLS, 0, Value::Null)
        .route(
            Method::POST,
            "/v1/agents/sessions/s1/fork",
            201,
            session("s2"),
        )
        .route(
            Method::DELETE,
            "/v1/agents/sessions/s2/tools/reserve_table",
            404,
            no_tool_by_that_name(),
        )
        .route(
            Method::POST,
            "/v1/agents/sessions/s2/tools",
            204,
            Value::Null,
        );
    parent
        .add_tools(answering("reserve_table", "first"))
        .await
        .unwrap_err();
    let fork = parent.fork(&Default::default()).await.unwrap();
    let mut socket = server.accept().await;

    fork.add_tools(answering("reserve_table", "second"))
        .await
        .unwrap();

    assert_eq!(
        tool_changes(&server),
        [
            format!("POST {SESSION_TOOLS}"),
            "DELETE /v1/agents/sessions/s2/tools/reserve_table".to_string(),
            "POST /v1/agents/sessions/s2/tools".to_string(),
        ]
    );
    assert_eq!(
        answer(&mut socket, "reserve_table").await,
        Some(json!("second"))
    );
}
