mod support;

use std::sync::Arc;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::time::Duration;

use axum::http::Method;
use serde_json::json;
use support::{Server, session};
use tokio::sync::mpsc;
use vision_agents::{Agent, Dispatch, Error, InboundCall, Tools};

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
    assert_eq!(socket.query, "capacity=2");
    assert_eq!(socket.header("stream-auth-type"), "server");

    socket
        .send(json!({"type": "ready", "worker_id": "w-1"}))
        .await;
    socket
        .send(json!({"type": "call", "call_id": "c1", "called_number": "+15550100", "custom": {"tier": "gold", "seats": 3}}))
        .await;
    assert_eq!(
        socket.expect("accepted").await,
        json!({"type": "accepted", "call_id": "c1"})
    );
    socket
        .send(json!({"type": "call", "call_id": "refused"}))
        .await;
    assert_eq!(
        socket.expect("rejected").await,
        json!({"type": "rejected", "call_id": "refused", "reason": "nobody is free"})
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
    socket.send(json!({"type": "call", "call_id": "c1"})).await;

    let rejected = socket.expect("rejected").await;
    assert_eq!(rejected["call_id"], "c1");
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
                session.respond(&message.text).await?;
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
    assert_eq!(conversation.expect("respond").await["text"], "hello");
    assert_eq!(replies.recv().await.unwrap(), "s1");

    worker
        .send(json!({"type": "message", "channel_id": "c1", "text": "again"}))
        .await;
    assert_eq!(conversation.expect("respond").await["text"], "again");
    replies.recv().await.unwrap();

    assert_eq!(created.load(Ordering::SeqCst), 1);
    let sent = server.request(Method::POST, "/v1/agents/sessions").body;
    assert_eq!(sent["conversation_id"], "agent:c1");
    assert_eq!(sent["agent_id"], "support-bot");

    dispatch.stop();
    conversation.expect("close").await;
    conversation.close().await;
    running.await.unwrap().unwrap();
}
