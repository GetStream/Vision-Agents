mod support;

use std::sync::Arc;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::time::Duration;

use axum::http::Method;
use serde_json::json;
use support::{Server, claims, response, session};
use tokio::sync::mpsc;
use vision_agents::{Agent, Dispatch, Error, InboundCall, InboundMessage, Tools};

fn investigate_sdk() -> Tools {
    let tools = Tools::new();
    tools.register(
        "investigate_sdk",
        "Read SDK source",
        json!({"type": "object", "properties": {"sdk": {"type": "string"}}}),
        async |arguments| {
            if arguments["sdk"] == "slow" {
                tokio::time::sleep(Duration::from_millis(300)).await;
            }
            Ok::<_, String>(format!("read {}", arguments["sdk"].as_str().unwrap_or("")))
        },
    );
    tools
}

#[tokio::test]
async fn a_hosted_function_is_declared_and_answered_over_the_dispatch_socket() {
    let server = Server::start().await;
    let dispatch = Dispatch::new(server.client());
    dispatch.host(
        "stream-support",
        investigate_sdk(),
        Some(Duration::from_secs(60)),
    );

    let running = {
        let dispatch = dispatch.clone();
        tokio::spawn(async move { dispatch.run().await })
    };
    let mut socket = server.accept().await;
    assert_eq!(socket.query, "capacity=4&active=0&handles=");
    socket
        .send(json!({"type": "ready", "worker_id": "w-1"}))
        .await;
    assert_eq!(
        socket.expect("host_tools").await,
        json!({
            "type": "host_tools",
            "agent_id": "stream-support",
            "tools": [{
                "name": "investigate_sdk",
                "description": "Read SDK source",
                "parameters": {"type": "object", "properties": {"sdk": {"type": "string"}}},
            }],
            "timeout_ms": 60000,
        })
    );
    socket
        .send(
            json!({"type": "hosting", "agent_id": "stream-support", "tools": ["investigate_sdk"]}),
        )
        .await;

    socket
        .send(json!({"type": "tool_call", "id": "call-1", "session_id": "s", "name": "investigate_sdk", "arguments": r#"{"sdk":"slow"}"#}))
        .await;
    socket
        .send(json!({"type": "tool_call", "id": "call-2", "session_id": "s", "name": "investigate_sdk", "arguments": r#"{"sdk":"android"}"#}))
        .await;
    socket
        .send(json!({"type": "tool_call", "id": "call-3", "session_id": "s", "name": "deploy", "arguments": "{}"}))
        .await;

    let mut answered = Vec::new();
    for _ in 0..3 {
        answered.push(socket.expect("tool_result").await);
    }
    assert_eq!(
        answered,
        vec![
            json!({"type": "tool_result", "id": "call-3", "error": "this worker does not run deploy"}),
            json!({"type": "tool_result", "id": "call-2", "output": "read android"}),
            json!({"type": "tool_result", "id": "call-1", "output": "read slow"}),
        ]
    );

    dispatch.stop();
    running.await.unwrap().unwrap();
}

#[tokio::test]
async fn a_worker_tells_the_router_what_it_hosts_every_time_it_is_ready() {
    let server = Server::start().await;
    let dispatch = Dispatch::new(server.client());
    dispatch.host("stream-support", investigate_sdk(), None);

    let running = {
        let dispatch = dispatch.clone();
        tokio::spawn(async move { dispatch.run().await })
    };
    let mut socket = server.accept().await;
    for worker in ["w-1", "w-2"] {
        socket
            .send(json!({"type": "ready", "worker_id": worker}))
            .await;
        let declared = socket.expect("host_tools").await;
        assert_eq!(declared["agent_id"], "stream-support");
        assert_eq!(declared["timeout_ms"], 0);
    }

    dispatch.stop();
    running.await.unwrap().unwrap();
}

#[tokio::test]
async fn a_worker_whose_tools_are_refused_stops_waiting() {
    let server = Server::start().await;
    let dispatch = Dispatch::new(server.client());
    dispatch.host("stream-support", investigate_sdk(), None);

    let running = {
        let dispatch = dispatch.clone();
        tokio::spawn(async move { dispatch.run().await })
    };
    let mut socket = server.accept().await;
    socket
        .send(json!({"type": "hosting_refused", "agent_id": "stream-support", "reason": "hosting no tools is not hosting"}))
        .await;

    let refused = running.await.unwrap().unwrap_err();
    assert!(matches!(refused, Error::Failed { .. }));
    assert_eq!(
        refused.to_string(),
        "dispatch: the router refused to host tools for agent stream-support: hosting no tools is not hosting"
    );
}

#[tokio::test]
async fn a_worker_answers_calls_and_tells_the_router_how_each_went() {
    let server = Server::start().await;
    let dispatch = Dispatch::with_capacity(server.backend(), 2);
    let (answered, mut calls) = mpsc::unbounded_channel::<InboundCall>();
    dispatch.wait_for_call(move |call| {
        let answered = answered.clone();
        async move {
            answered.send(call.clone()).unwrap();
            if call.call_id == "refused" {
                return Err(Error::Configuration("nobody is free".into()));
            }
            Ok(())
        }
    });

    let running = {
        let dispatch = dispatch.clone();
        tokio::spawn(async move { dispatch.run().await })
    };
    let mut socket = server.accept().await;
    assert_eq!(socket.path, "/v1/dispatch");
    assert_eq!(socket.query, "capacity=2&active=0&handles=call");
    assert_eq!(socket.header("stream-auth-type"), "server");

    socket
        .send(json!({"type": "ready", "worker_id": "w-1"}))
        .await;
    socket
        .send(json!({"type": "call", "work_id": "work-1", "call_id": "c1", "called_number": "+15550100", "custom": {"tier": "gold", "seats": 3}}))
        .await;
    assert_eq!(
        socket.expect("done").await,
        json!({"type": "done", "work_id": "work-1"})
    );
    socket
        .send(json!({"type": "call", "work_id": "work-2", "call_id": "refused"}))
        .await;
    assert_eq!(
        socket.expect("done").await,
        json!({"type": "done", "work_id": "work-2", "error": "nobody is free"})
    );

    let first = calls.recv().await.unwrap();
    assert_eq!(first.call_type, "default");
    assert_eq!(first.called_number, "+15550100");
    assert_eq!(first.custom["tier"], "gold");
    assert_eq!(first.custom["seats"], "3");
    assert_eq!(dispatch.worker_id(), "w-1");

    dispatch.stop();
    running.await.unwrap().unwrap();
}

#[tokio::test]
async fn a_handler_that_panics_is_a_call_nobody_took() {
    let server = Server::start().await;
    let dispatch = Dispatch::new(server.client());
    dispatch.wait_for_call(|_| async { panic!("the handler fell over") });

    let running = {
        let dispatch = dispatch.clone();
        tokio::spawn(async move { dispatch.run().await })
    };
    let mut socket = server.accept().await;
    socket
        .send(json!({"type": "call", "work_id": "work-1", "call_id": "c1"}))
        .await;

    assert_eq!(
        socket.expect("done").await,
        json!({"type": "done", "work_id": "work-1", "error": "the handler panicked"})
    );
    socket.close().await;
    running.await.unwrap().unwrap();
}

#[tokio::test]
async fn work_in_flight_is_finished_before_the_worker_stops() {
    let server = Server::start().await;
    let dispatch = Dispatch::new(server.client());
    let finished = Arc::new(AtomicUsize::new(0));
    {
        let finished = finished.clone();
        dispatch.wait_for_call(move |_| {
            let finished = finished.clone();
            async move {
                tokio::time::sleep(Duration::from_millis(200)).await;
                finished.fetch_add(1, Ordering::SeqCst);
                Ok(())
            }
        });
    }

    let running = {
        let dispatch = dispatch.clone();
        tokio::spawn(async move { dispatch.run().await })
    };
    let mut socket = server.accept().await;
    socket.send(json!({"type": "call", "call_id": "c1"})).await;
    tokio::time::sleep(Duration::from_millis(50)).await;
    dispatch.stop();
    running.await.unwrap().unwrap();

    assert_eq!(finished.load(Ordering::SeqCst), 1);
}

#[tokio::test]
async fn a_worker_with_no_handler_is_refused() {
    let server = Server::start().await;

    let refused = Dispatch::new(server.client()).run().await;

    assert!(matches!(refused, Err(Error::Configuration(_))));
}

#[tokio::test]
async fn the_second_message_on_a_channel_goes_to_the_session_that_answered_the_first() {
    let server = Server::start().await;
    server.route(Method::POST, "/v1/agents/sessions", 201, session("s1"));
    server.route(
        Method::POST,
        "/v1/agents/sessions/s1/responses",
        201,
        response("r1", "s1"),
    );
    let dispatch = Dispatch::new(server.client());
    let created = Arc::new(AtomicUsize::new(0));
    let (replied, mut replies) = mpsc::unbounded_channel::<String>();
    {
        let (dispatch_for_handler, client, created) =
            (dispatch.clone(), server.client(), created.clone());
        dispatch.wait_for_message(move |message| {
            let (dispatch, client, created, replied) = (
                dispatch_for_handler.clone(),
                client.clone(),
                created.clone(),
                replied.clone(),
            );
            async move {
                let session = dispatch
                    .get_or_create_agent(&message, || async move {
                        created.fetch_add(1, Ordering::SeqCst);
                        Ok(Agent::new("support").client(client))
                    })
                    .await?;
                session.responses.create(&message.text).await?;
                replied.send(session.id().to_string()).unwrap();
                Ok(())
            }
        });
    }

    let running = {
        let dispatch = dispatch.clone();
        tokio::spawn(async move { dispatch.run().await })
    };
    let mut worker = server.accept().await;
    worker
        .send(json!({"type": "message", "channel_id": "c1", "agent_id": "support-bot", "text": "hello"}))
        .await;
    let mut conversation = server.accept().await;
    assert_eq!(replies.recv().await.unwrap(), "s1");

    worker
        .send(json!({"type": "message", "channel_id": "c1", "text": "again"}))
        .await;
    replies.recv().await.unwrap();

    assert_eq!(created.load(Ordering::SeqCst), 1);
    let asked: Vec<_> = server
        .requests(Method::POST, "/v1/agents/sessions/s1/responses")
        .into_iter()
        .map(|seen| seen.body["text"].clone())
        .collect();
    assert_eq!(asked, ["hello", "again"]);
    let sent = server.request(Method::POST, "/v1/agents/sessions").body;
    assert_eq!(sent["conversation_id"], "agent:c1");
    assert_eq!(sent["agent_id"], "support-bot");

    dispatch.stop();
    conversation.expect("close").await;
    conversation.close().await;
    running.await.unwrap().unwrap();
}

#[tokio::test]
async fn a_message_nobody_handles_is_done_with_an_error() {
    let server = Server::start().await;
    let dispatch = Dispatch::new(server.client());
    dispatch.wait_for_call(|_| async { Ok(()) });

    let running = {
        let dispatch = dispatch.clone();
        tokio::spawn(async move { dispatch.run().await })
    };
    let mut socket = server.accept().await;
    socket
        .send(json!({"type": "message", "work_id": "work-1", "channel_id": "c1", "text": "hello"}))
        .await;

    assert_eq!(
        socket.expect("done").await,
        json!({"type": "done", "work_id": "work-1", "error": "this worker answers no messages"})
    );
    dispatch.stop();
    running.await.unwrap().unwrap();
}

#[tokio::test]
async fn a_message_left_to_dispatch_is_answered_on_its_session_for_whoever_wrote_it() {
    let server = Server::start().await;
    server.route(
        Method::POST,
        "/v1/agents/sessions/s1/responses",
        201,
        response("r1", "s1"),
    );
    let dispatch = Dispatch::new(server.backend());
    let (handled, mut messages) = mpsc::unbounded_channel::<InboundMessage>();
    {
        let worker = dispatch.clone();
        dispatch.wait_for_message(move |message| {
            let (worker, handled) = (worker.clone(), handled.clone());
            async move {
                handled.send(message.clone()).unwrap();
                worker.answer(&message).await?;
                Ok(())
            }
        });
    }

    let running = {
        let dispatch = dispatch.clone();
        tokio::spawn(async move { dispatch.run().await })
    };
    let mut socket = server.accept().await;
    assert_eq!(socket.query, "capacity=4&active=0&handles=message");
    socket
        .send(json!({"type": "message", "work_id": "work-1", "session_id": "s1", "command_id": "cmd-1",
                     "agent_id": "support", "user_id": "ada", "text": "where is my order?"}))
        .await;
    assert_eq!(
        socket.expect("done").await,
        json!({"type": "done", "work_id": "work-1"})
    );

    let message = messages.recv().await.unwrap();
    assert_eq!(
        (message.session_id.as_str(), message.command_id.as_str()),
        ("s1", "cmd-1")
    );
    let sent = server.request(Method::POST, "/v1/agents/sessions/s1/responses");
    assert_eq!(
        sent.body,
        json!({"text": "where is my order?", "command_id": "cmd-1"})
    );
    assert_eq!(sent.header("x-stream-user-id"), "ada");
    assert_eq!(sent.header("stream-auth-type"), "server");
    assert_eq!(claims(sent.header("authorization"))["server"], true);

    dispatch.stop();
    running.await.unwrap().unwrap();
}

#[tokio::test]
async fn a_message_with_no_session_cannot_be_answered_on_one() {
    let server = Server::start().await;
    let dispatch = Dispatch::new(server.backend());

    let refused = dispatch
        .answer(&InboundMessage {
            channel_id: "c1".into(),
            text: "hello".into(),
            ..Default::default()
        })
        .await;

    assert!(matches!(refused, Err(Error::Configuration(_))));
    assert!(server.seen().is_empty());
}

#[tokio::test]
async fn a_message_a_session_already_holds_gets_no_agent_of_its_own() {
    let server = Server::start().await;
    let dispatch = Dispatch::new(server.client());
    let client = server.client();

    let refused = dispatch
        .get_or_create_agent(
            &InboundMessage {
                channel_id: "c1".into(),
                session_id: "s1".into(),
                ..Default::default()
            },
            || async move { Ok(Agent::new("support").client(client)) },
        )
        .await;

    let Err(Error::Configuration(reason)) = refused else {
        panic!("a message with a session got an agent");
    };
    assert!(reason.contains("answer"), "{reason}");
    assert!(server.seen().is_empty());
}
