mod support;

use std::path::Path;
use std::time::Duration;

use axum::http::Method;
use serde_json::{Value, json};
use support::{Server, config, session};
use vision_agents::{Agent, Harness, InboundCall, InboundMessage, Session, Skill, folder, types};

fn agent(server: &Server, name: &str) -> Agent {
    Agent::new(name)
        .client(server.client())
        .stream(server.stream())
}

/// Runs `open` while playing the router's side of the socket it opens.
async fn opened<F>(server: &Server, open: F) -> (Session, support::Accepted)
where
    F: Future<Output = vision_agents::Result<Session>> + Send + 'static,
{
    let opening = tokio::spawn(open);
    let socket = server.accept().await;
    (opening.await.unwrap().unwrap(), socket)
}

fn no_call_is_created(server: &Server) {
    assert!(
        server
            .seen()
            .iter()
            .all(|seen| !seen.path.starts_with("/api/v2")),
        "a Stream call was created"
    );
}

fn write(root: &Path, name: &str, content: &str) {
    let path = root.join(name);
    std::fs::create_dir_all(path.parent().unwrap()).unwrap();
    std::fs::write(path, content).unwrap();
}

#[tokio::test]
async fn joining_starts_voice_on_the_sessions_own_call_and_carries_the_agents_configuration() {
    let server = Server::start().await;
    server.route(Method::POST, "/v1/agents/sessions", 201, session("s1"));
    let jean = agent(&server, "rust_sdk_test_agent")
        .cost_tracking([("env", "production")])
        .memory_filter([("user_id", "123"), ("team", "blue")]);

    let (session, _socket) = opened(&server, async move { jean.join().await }).await;

    let sent = server.request(Method::POST, "/v1/agents/sessions").body;
    assert_eq!(
        sent,
        json!({
            "agent": "rust_sdk_test_agent",
            "agent_id": "rust_sdk_test_agent",
            "user_id": "rust_sdk_test_agent",
            "user_name": "rust_sdk_test_agent",
            "start_voice": true,
            "tags": {"env": "production"},
            "memory": {"user_id": "123", "filter": {"team": "blue"}},
        })
    );
    assert!(session.voice_started());
    assert_eq!(
        session.call(),
        vision_agents::Call {
            id: "s1".into(),
            kind: "agent".into()
        }
    );
    no_call_is_created(&server);
}

fn subagent_and_sandbox() -> Harness {
    let mut harness = Harness::standard();
    harness
        .subagents
        .insert("default".into(), "openai/gpt-5.6".into());
    harness.vm = Some(vision_agents::daytona());
    harness
}

#[tokio::test]
async fn an_agent_spelled_out_in_code_sends_its_pipeline_but_never_its_instructions_or_harness() {
    let server = Server::start().await;
    server.route(Method::POST, "/v1/agents/sessions", 201, session("s1"));
    let mut harness = subagent_and_sandbox();
    harness.skills.push(Skill {
        deadline: Duration::from_secs(30),
        ..Skill::new("think", "Work it out", "Reason.")
    });
    let jean = Agent::named("Jean Luc")
        .client(server.client())
        .instructions("You are Jean.")
        .llm("llm-fast")
        .harness(harness);

    let (_session, _socket) = opened(&server, async move { jean.chat().await }).await;

    let sent = server.request(Method::POST, "/v1/agents/sessions").body;
    assert_eq!(sent["user_id"], "jean-luc");
    assert_eq!(sent["llm"], "llm-fast");
    for left_out in [
        "instructions",
        "start_voice",
        "subagent",
        "sandbox",
        "skills",
        "tasks",
        "agent",
    ] {
        assert!(sent.get(left_out).is_none(), "{left_out} was sent");
    }
}

#[tokio::test]
async fn a_conversation_is_resumed_by_the_id_of_its_session() {
    let server = Server::start().await;
    server.route(Method::GET, "/v1/agents/sessions/s1", 200, session("s1"));
    let jean = agent(&server, "jean");

    let (session, socket) = opened(&server, async move { jean.resume("s1").await }).await;

    assert_eq!(session.id(), "s1");
    assert_eq!(socket.path, "/v1/agents/sessions/s1/events");
    assert!(
        server
            .seen()
            .iter()
            .all(|seen| seen.path != "/v1/agents/sessions"),
        "a new session was opened"
    );
}

#[tokio::test]
async fn agent_yaml_names_the_harness_and_the_code_its_subagent_and_sandbox() {
    let server = Server::start().await;
    synced(&server);
    let root = tempfile::tempdir().unwrap();
    write(
        root.path(),
        "agent.yaml",
        "name: jean\nharness: default\nsandbox: daytona\nplugins: [sentry]\ngreeting:\n  text: Hello.\n  mode: variation\n",
    );
    let agent = Agent::from_folder(root.path())
        .unwrap()
        .client(server.client())
        .harness(subagent_and_sandbox());

    agent.sync().await.unwrap();

    let body = server.request(Method::POST, "/v1/agents/sync").body;
    assert_eq!(body["harness"], "default");
    assert_eq!(body["subagent"], "openai/gpt-5.6");
    assert_eq!(body["sandbox"], "daytona");
    assert_eq!(body["plugins"], json!(["sentry"]));
    assert_eq!(
        body["greeting"],
        json!({"text": "Hello.", "mode": "variation"})
    );
    assert_ne!(body["hash"], agent.folder().unwrap().hash().as_str());
}

#[tokio::test]
async fn an_agents_config_is_patched_with_only_what_was_set() {
    let server = Server::start().await;
    server.route(
        Method::GET,
        "/v1/agents/configs",
        200,
        json!([config("cfg-1", "docs")]),
    );
    server.route(
        Method::PATCH,
        "/v1/agents/configs/cfg-1",
        200,
        config("cfg-1", "docs"),
    );
    let docs = server.client().agent("docs");

    let patched = docs
        .update_config(&types::AgentConfigPatch {
            guardrail: Some("Be kind.".into()),
            visible_tools: Some(vec!["athena_*".into()]),
            ..Default::default()
        })
        .await
        .unwrap();
    let missing = server.client().agent("nobody");

    assert_eq!(patched.id, "cfg-1");
    assert_eq!(
        server
            .request(Method::PATCH, "/v1/agents/configs/cfg-1")
            .body,
        json!({"guardrail": "Be kind.", "visible_tools": ["athena_*"]})
    );
    assert!(matches!(
        missing.update_config(&Default::default()).await,
        Err(vision_agents::Error::Configuration(_))
    ));
}

#[tokio::test]
async fn an_inbound_call_is_answered_by_opening_the_session_it_names_with_voice() {
    let server = Server::start().await;
    server.route(Method::POST, "/v1/agents/sessions", 201, session("s1"));
    let call = InboundCall {
        call_id: "s1".into(),
        call_type: "agent".into(),
        session_id: "s1".into(),
        called_number: "+15550100".into(),
        ..Default::default()
    };
    let support = agent(&server, "support");

    let (_session, _socket) = opened(&server, async move { support.answer(&call).await }).await;

    let sent = server.request(Method::POST, "/v1/agents/sessions").body;
    assert_eq!(
        (sent["id"].clone(), sent["start_voice"].clone()),
        (json!("s1"), json!(true))
    );
    assert_eq!(sent["phone"], json!({"number": "+15550100"}));
    assert!(sent.get("call_id").is_none());
    no_call_is_created(&server);
}

#[tokio::test]
async fn an_inbound_call_naming_no_session_is_refused() {
    let server = Server::start().await;

    let refused = agent(&server, "support")
        .answer(&InboundCall {
            call_id: "sip-1".into(),
            ..Default::default()
        })
        .await
        .unwrap_err();

    assert!(matches!(refused, vision_agents::Error::Configuration(_)));
    assert!(server.seen().is_empty());
}

#[tokio::test]
async fn waiting_for_a_call_attaches_the_number_and_answers_the_caller_handed_over() {
    let server = Server::start().await;
    server.route(
        Method::POST,
        "/v1/phone/numbers/%2B15550100/attach",
        200,
        json!({"route_id": "r1", "sip_uri": "sip:agent@example.com", "trunk_id": "t1"}),
    );
    server.route(Method::POST, "/v1/agents/sessions", 201, session("s7"));
    let support = agent(&server, "support");

    let waiting = tokio::spawn(async move { support.wait_for_call("+15550100").await });
    let mut worker = server.accept().await;
    assert_eq!(worker.path, "/v1/dispatch");
    assert_eq!(worker.query, "capacity=1&active=0&handles=call");
    worker
        .send(
            json!({"type": "call", "work_id": "work-1", "call_id": "s7", "call_type": "agent",
                     "session_id": "s7", "called_number": "+15550100"}),
        )
        .await;
    assert_eq!(
        worker.expect("done").await,
        json!({"type": "done", "work_id": "work-1"})
    );
    let mut conversation = server.accept().await;
    conversation
        .send(json!({"type": "heard", "text": "hello?"}))
        .await;
    let session = waiting.await.unwrap().unwrap();

    assert_eq!(session.id(), "s7");
    assert_eq!(
        server
            .request(Method::POST, "/v1/phone/numbers/%2B15550100/attach")
            .body,
        json!({})
    );
    let sent = server.request(Method::POST, "/v1/agents/sessions").body;
    assert_eq!(
        (sent["id"].clone(), sent["start_voice"].clone()),
        (json!("s7"), json!(true))
    );
    assert_eq!(sent["phone"], json!({"number": "+15550100"}));
}

#[tokio::test]
async fn a_message_is_replied_to_in_writing_as_the_agent_it_was_written_to() {
    let server = Server::start().await;
    server.route(Method::POST, "/v1/agents/sessions", 201, session("s1"));
    let message = InboundMessage {
        channel_id: "c1".into(),
        channel_type: "messaging".into(),
        agent_id: "support-bot".into(),
        ..Default::default()
    };
    let support = agent(&server, "support");

    let (_session, _socket) = opened(&server, async move { support.reply(&message).await }).await;

    let sent = server.request(Method::POST, "/v1/agents/sessions").body;
    assert!(sent.get("conversation_id").is_none());
    assert!(sent.get("incognito").is_none());
    assert!(sent.get("start_voice").is_none());
    assert_eq!(sent["agent_id"], "support-bot");
}

#[tokio::test]
async fn an_outbound_call_is_placed_then_its_session_joined_as_navigating() {
    let server = Server::start().await;
    server.route(
        Method::POST,
        "/v1/phone/calls",
        201,
        json!({"status": "ringing", "vendor_call_id": "vendor-9", "session_id": "s1"}),
    );
    server.route(Method::POST, "/v1/agents/sessions", 201, session("s1"));
    let seller = agent(&server, "seller").cost_tracking([("campaign", "spring")]);

    let (_session, _socket) = opened(&server, async move {
        seller.outbound_call("+15550100", "+15550199").await
    })
    .await;

    assert_eq!(
        server.request(Method::POST, "/v1/phone/calls").body,
        json!({"from": "+15550100", "to": "+15550199", "tags": {"campaign": "spring"}})
    );
    let sent = server.request(Method::POST, "/v1/agents/sessions").body;
    assert_eq!(sent["id"], "s1");
    assert_eq!(sent["start_voice"], true);
    assert_eq!(sent["navigating"], true);
    assert_eq!(
        sent["phone"],
        json!({"number": "+15550100", "vendor_call_id": "vendor-9"})
    );
    no_call_is_created(&server);
}

#[tokio::test]
async fn an_outbound_call_needs_both_numbers() {
    let server = Server::start().await;

    let refused = agent(&server, "seller")
        .outbound_call("", "+15550199")
        .await
        .unwrap_err();

    assert!(matches!(refused, vision_agents::Error::Configuration(_)));
    assert!(server.seen().is_empty());
}

#[tokio::test]
async fn a_monitoring_link_names_the_sessions_call() {
    let server = Server::start().await;
    let jean = agent(&server, "jean");
    server.route(Method::POST, "/v1/agents/sessions", 201, session("s1"));
    let client = server.client();
    let (session, _socket) = opened(&server, async move {
        Session::open(
            &client,
            Default::default(),
            Default::default(),
            Default::default(),
        )
        .await
    })
    .await;

    let link = jean.monitor_url(&session).unwrap();

    assert!(
        link.starts_with("https://example.com/demo/join/s1?"),
        "{link}"
    );
    assert!(link.contains("user_name=Monitor"), "{link}");
}

#[tokio::test]
async fn a_conversation_held_in_writing_has_no_call_to_monitor() {
    let server = Server::start().await;
    let jean = agent(&server, "jean");
    let mut written = session("s1");
    written["call_id"] = json!("");
    server.route(Method::POST, "/v1/agents/sessions", 201, written);
    let client = server.client();
    let (session, _socket) = opened(&server, async move {
        Session::open(
            &client,
            Default::default(),
            Default::default(),
            Default::default(),
        )
        .await
    })
    .await;

    assert!(!session.voice_started());
    assert!(matches!(
        jean.monitor_url(&session),
        Err(vision_agents::Error::Configuration(_))
    ));
}

fn jean(root: &Path) {
    write(
        root,
        "agent.yaml",
        "name: jean\nllm: openai/gpt-5.6\nsts: \"\"\ntags:\n  team: support\n",
    );
    write(root, "instructions.md", "You are Jean.\n");
    write(
        root,
        "skills/think.md",
        "---\ndescription: Work it out\ndeadline: 30s\n---\nReason it through.\n",
    );
    write(root, "knowledge/pricing.md", "# Pricing\n\nA penny.\n");
    write(
        root,
        "knowledge/urls.yaml",
        "- url: https://example.com/plans\n  title: Plans\n",
    );
}

fn synced(server: &Server) {
    server.route(
        Method::POST,
        "/v1/agents/sync",
        200,
        json!({"config": config("cfg-1", "jean"), "unchanged": false}),
    );
    server.route(
        Method::GET,
        "/v1/agents/configs",
        200,
        json!([config("cfg-1", "jean")]),
    );
}

#[tokio::test]
async fn a_directory_is_synced_once_and_then_only_read_back() {
    let server = Server::start().await;
    synced(&server);
    let root = tempfile::tempdir().unwrap();
    jean(root.path());
    let agent = Agent::from_folder(root.path())
        .unwrap()
        .client(server.client());

    let stored = agent.sync().await.unwrap();
    let again = agent.sync().await.unwrap();

    assert_eq!((stored.id.as_str(), again.id.as_str()), ("cfg-1", "cfg-1"));
    let body = server.request(Method::POST, "/v1/agents/sync").body;
    let hash = agent.folder().unwrap().hash();
    assert_eq!(
        body,
        json!({
            "name": "jean",
            "hash": hash,
            "instructions": "You are Jean.",
            "skills": [{"name": "think", "description": "Work it out", "instructions": "Reason it through.",
                        "capture_video": false, "deadline_ms": 30000, "config_id": ""}],
            "knowledge": [{"source": "pricing.md", "text": "# Pricing\n\nA penny.\n"}],
            "knowledge_urls": [{"url": "https://example.com/plans", "title": "Plans"}],
            "llm": "openai/gpt-5.6",
            "sts": "",
            "tags": {"team": "support"},
        })
    );
    assert_eq!(folder::read_stamp(root.path()), hash);
    let stamp: Value =
        serde_json::from_str(&std::fs::read_to_string(root.path().join(".agent_sync")).unwrap())
            .unwrap();
    assert!(stamp["synced_at"].as_str().unwrap().ends_with("+00:00"));
    assert_eq!(
        server.request(Method::GET, "/v1/agents/configs").query,
        "name=jean"
    );
}

#[tokio::test]
async fn what_a_directory_leaves_to_dispatch_is_synced_as_written() {
    let server = Server::start().await;
    synced(&server);
    let root = tempfile::tempdir().unwrap();
    write(
        root.path(),
        "agent.yaml",
        "name: jean\ndispatch:\n  text: enabled\n",
    );
    let agent = Agent::from_folder(root.path())
        .unwrap()
        .client(server.client());

    agent.sync().await.unwrap();

    let body = server.request(Method::POST, "/v1/agents/sync").body;
    assert_eq!(body["dispatch"], json!({"text": "enabled"}));
    assert!(body.get("simulations").is_none());
    assert!(body.get("speed").is_none());
}

#[tokio::test]
async fn a_directorys_simulations_and_page_schedules_are_synced_as_declared() {
    let server = Server::start().await;
    synced(&server);
    let root = tempfile::tempdir().unwrap();
    write(root.path(), "agent.yaml", "name: jean\n");
    write(
        root.path(),
        "knowledge/urls.yaml",
        "- url: https://example.com/plans\n  refresh_hours: 24\n",
    );
    write(
        root.path(),
        "simulations/lunch.yaml",
        "- name: lunch\n  scenario: Order a club.\n  assertion: One club.\n  variations: 3\n",
    );
    let agent = Agent::from_folder(root.path())
        .unwrap()
        .client(server.client());

    agent.sync().await.unwrap();

    let body = server.request(Method::POST, "/v1/agents/sync").body;
    assert_eq!(
        body["knowledge_urls"],
        json!([{"url": "https://example.com/plans", "refresh_hours": 24}])
    );
    assert_eq!(
        body["simulations"],
        json!([{"name": "lunch", "scenario": "Order a club.", "assertion": "One club.", "variations": 3}])
    );
}

#[tokio::test]
async fn an_empty_simulations_directory_is_synced_as_none_declared() {
    let server = Server::start().await;
    synced(&server);
    let root = tempfile::tempdir().unwrap();
    write(root.path(), "agent.yaml", "name: jean\n");
    std::fs::create_dir(root.path().join("simulations")).unwrap();
    let agent = Agent::from_folder(root.path())
        .unwrap()
        .client(server.client());

    agent.sync().await.unwrap();

    assert_eq!(
        server.request(Method::POST, "/v1/agents/sync").body["simulations"],
        json!([])
    );
}

#[tokio::test]
async fn cost_tracking_is_part_of_what_a_directory_is_synced_under() {
    let server = Server::start().await;
    synced(&server);
    let root = tempfile::tempdir().unwrap();
    jean(root.path());
    let agent = Agent::from_folder(root.path())
        .unwrap()
        .client(server.client())
        .cost_tracking([("env", "production")]);

    agent.sync().await.unwrap();

    let body = server.request(Method::POST, "/v1/agents/sync").body;
    assert_ne!(body["hash"], agent.folder().unwrap().hash().as_str());
    assert_eq!(
        body["tags"],
        json!({"team": "support", "env": "production"})
    );
}

#[tokio::test]
async fn an_agent_read_from_a_directory_is_stored_before_its_first_session() {
    let server = Server::start().await;
    synced(&server);
    server.route(Method::POST, "/v1/agents/sessions", 201, session("s1"));
    let root = tempfile::tempdir().unwrap();
    jean(root.path());
    let jean = Agent::from_folder(root.path())
        .unwrap()
        .client(server.client())
        .stream(server.stream());

    let (_session, _socket) = opened(&server, async move { jean.join().await }).await;

    let order: Vec<_> = server
        .seen()
        .into_iter()
        .map(|seen| seen.path)
        .filter(|path| path.starts_with("/v1"))
        .collect();
    assert_eq!(order, ["/v1/agents/sync", "/v1/agents/sessions"]);
    let sent = server.request(Method::POST, "/v1/agents/sessions").body;
    assert_eq!(sent["agent"], "jean");
    assert!(sent.get("instructions").is_none());
    assert_eq!(
        server.request(Method::POST, "/v1/agents/sync").body["instructions"],
        "You are Jean."
    );
}

#[tokio::test]
async fn an_agent_spelled_out_in_code_is_stored_by_name() {
    let server = Server::start().await;
    server.route(Method::GET, "/v1/agents/skills", 200, json!([{"id": "sk-1", "config_id": "", "name": "think", "description": "d",
        "instructions": "i", "created_at": "2026-09-24T10:00:00Z", "updated_at": "2026-09-24T10:00:00Z"}]));
    server.route(Method::PUT, "/v1/agents/skills/sk-1", 200, json!({"id": "sk-1", "config_id": "", "name": "think", "description": "d",
        "instructions": "i", "created_at": "2026-09-24T10:00:00Z", "updated_at": "2026-09-24T10:00:00Z"}));
    server.route(Method::GET, "/v1/agents/configs", 200, json!([]));
    server.route(
        Method::POST,
        "/v1/agents/configs",
        201,
        config("cfg-2", "Ada"),
    );
    let mut harness = subagent_and_sandbox();
    harness.name = Some(types::Harness::Default);
    harness
        .skills
        .push(Skill::new("think", "Work it out", "Reason."));
    let ada = Agent::named("Ada")
        .client(server.client())
        .instructions("You are Ada.")
        .harness(harness);

    let stored = ada.sync().await.unwrap();

    assert_eq!(stored.id, "cfg-2");
    assert_eq!(
        server.request(Method::PUT, "/v1/agents/skills/sk-1").body["instructions"],
        "Reason."
    );
    assert_eq!(
        server.request(Method::POST, "/v1/agents/configs").body,
        json!({"name": "Ada", "instructions": "You are Ada.", "skills": ["think"], "harness": "default",
               "subagent": "openai/gpt-5.6", "sandbox": "daytona"})
    );
}

fn page(state: &str, passages: i64) -> Value {
    json!({"id": "page-1", "url": "https://example.com/plans", "namespace": "jean", "state": state, "passages": passages,
           "created_at": "2026-09-24T10:00:00Z", "updated_at": "2026-09-24T10:00:00Z"})
}

#[tokio::test]
async fn a_page_is_added_to_the_agents_knowledge_and_waited_for() {
    let server = Server::start().await;
    server.route(
        Method::POST,
        "/v1/agents/knowledge/urls",
        201,
        page("pending", 0),
    );
    server.route(
        Method::GET,
        "/v1/agents/knowledge/urls/page-1",
        200,
        page("pending", 0),
    );
    server.route(
        Method::GET,
        "/v1/agents/knowledge/urls/page-1",
        200,
        page("indexed", 12),
    );

    let knowledge = agent(&server, "jean").knowledge().unwrap();
    let read = knowledge
        .add_url("https://example.com/plans", "Plans", "", Some(24))
        .await
        .unwrap();

    assert_eq!(read.state, types::KnowledgeUrlState::Indexed);
    assert_eq!(read.passages, 12);
    assert_eq!(
        server
            .request(Method::POST, "/v1/agents/knowledge/urls")
            .body,
        json!({"namespace": "jean", "url": "https://example.com/plans", "title": "Plans", "refresh_hours": 24})
    );
    assert_eq!(
        server
            .requests(Method::GET, "/v1/agents/knowledge/urls/page-1")
            .len(),
        2
    );
}

#[test]
fn a_knowledge_base_belongs_to_a_stored_config() {
    assert!(Agent::named("adhoc").knowledge().is_err());
}

#[test]
fn a_name_becomes_something_a_call_can_be_joined_under() {
    assert_eq!(vision_agents::user_id_of("Jean Luc!"), "jean-luc");
    assert_eq!(vision_agents::user_id_of("¡!"), "vision-agent");
}
