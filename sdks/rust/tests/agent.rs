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

fn created_call(server: &Server) -> String {
    let created: Vec<_> = server
        .seen()
        .into_iter()
        .filter(|seen| seen.path.starts_with("/api/v2/video/call/agent/"))
        .collect();
    assert_eq!(created.len(), 1, "one call is created");
    created[0].path.rsplit('/').next().unwrap().to_string()
}

fn write(root: &Path, name: &str, content: &str) {
    let path = root.join(name);
    std::fs::create_dir_all(path.parent().unwrap()).unwrap();
    std::fs::write(path, content).unwrap();
}

#[tokio::test]
async fn joining_creates_the_call_and_carries_the_agents_configuration() {
    let server = Server::start().await;
    server.route(
        Method::POST,
        "/api/v2/video/call/agent/support-call",
        201,
        json!({}),
    );
    server.route(Method::POST, "/v1/agents/sessions", 201, session("s1"));
    let jean = agent(&server, "rust_sdk_test_agent")
        .cost_tracking([("env", "production")])
        .memory_filter([("user_id", "123"), ("team", "blue")]);

    let (_session, _socket) = opened(&server, async move { jean.join("support-call").await }).await;

    let sent = server.request(Method::POST, "/v1/agents/sessions").body;
    assert_eq!(
        sent,
        json!({
            "agent": "rust_sdk_test_agent",
            "agent_id": "rust_sdk_test_agent",
            "user_id": "rust_sdk_test_agent",
            "user_name": "rust_sdk_test_agent",
            "call_id": "support-call",
            "call_type": "agent",
            "tags": {"env": "production"},
            "memory": {"user_id": "123", "filter": {"team": "blue"}},
        })
    );
    assert_eq!(
        server
            .request(Method::POST, "/api/v2/video/call/agent/support-call")
            .body["data"]["created_by_id"],
        "rust_sdk_test_agent"
    );
}

#[tokio::test]
async fn an_agent_spelled_out_in_code_sends_its_instructions_harness_and_pipeline() {
    let server = Server::start().await;
    server.route(Method::POST, "/v1/agents/sessions", 201, session("s1"));
    let mut harness = Harness::standard();
    harness
        .subagents
        .insert("default".into(), "openai/gpt-5.6".into());
    harness.vm = Some(vision_agents::daytona());
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
    assert_eq!(sent["instructions"], "You are Jean.");
    assert_eq!(sent["llm"], "llm-fast");
    assert_eq!(sent["text"], true);
    assert_eq!(sent["subagent"], "openai/gpt-5.6");
    assert_eq!(sent["sandbox"], "daytona");
    assert_eq!(
        sent["skills"],
        json!([{"name": "think", "description": "Work it out", "instructions": "Reason.", "capture_video": false, "deadline_ms": 30000}])
    );
    assert!(sent.get("agent").is_none());
}

#[tokio::test]
async fn an_empty_skill_list_turns_delegation_off() {
    let server = Server::start().await;
    server.route(Method::POST, "/v1/agents/sessions", 201, session("s1"));
    let quiet = Agent::named("quiet")
        .client(server.client())
        .harness(Harness::default());

    let (_session, _socket) = opened(&server, async move { quiet.chat().await }).await;

    assert_eq!(
        server.request(Method::POST, "/v1/agents/sessions").body["skills"],
        json!([])
    );
}

#[tokio::test]
async fn an_inbound_call_is_answered_in_the_call_it_arrived_in() {
    let server = Server::start().await;
    server.route(Method::POST, "/v1/agents/sessions", 201, session("s1"));
    let call = InboundCall {
        call_id: "sip-1".into(),
        call_type: "default".into(),
        called_number: "+15550100".into(),
        ..Default::default()
    };
    let support = agent(&server, "support");

    let (_session, _socket) = opened(&server, async move { support.answer(&call).await }).await;

    let sent = server.request(Method::POST, "/v1/agents/sessions").body;
    assert_eq!(
        (sent["call_id"].clone(), sent["call_type"].clone()),
        (json!("sip-1"), json!("default"))
    );
    assert_eq!(sent["phone"], json!({"number": "+15550100"}));
    assert!(
        server
            .seen()
            .iter()
            .all(|seen| !seen.path.starts_with("/api/v2")),
        "no call is created"
    );
}

#[tokio::test]
async fn a_message_is_replied_to_in_its_own_channel() {
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
    assert_eq!(sent["conversation_id"], "messaging:c1");
    assert_eq!(sent["persist_conversation"], true);
    assert_eq!(sent["agent_id"], "support-bot");
    assert_eq!(sent["text"], true);
}

#[tokio::test]
async fn an_outbound_call_is_placed_then_joined_as_navigating() {
    let server = Server::start().await;
    server.route(Method::POST, "/api/v2/video/call/agent/*", 201, json!({}));
    server.route(
        Method::POST,
        "/v1/phone/calls",
        201,
        json!({"status": "ringing", "vendor_call_id": "vendor-9"}),
    );
    server.route(Method::POST, "/v1/agents/sessions", 201, session("s1"));
    let seller = agent(&server, "seller").cost_tracking([("campaign", "spring")]);

    let (_session, _socket) = opened(&server, async move {
        seller.outbound_call("+15550100", "+15550199").await
    })
    .await;

    let call_id = created_call(&server);
    assert_eq!(
        server.request(Method::POST, "/v1/phone/calls").body,
        json!({"from": "+15550100", "to": "+15550199", "call_id": call_id, "call_type": "agent", "tags": {"campaign": "spring"}})
    );
    let sent = server.request(Method::POST, "/v1/agents/sessions").body;
    assert_eq!(sent["call_id"], call_id.as_str());
    assert_eq!(sent["navigating"], true);
    assert_eq!(
        sent["phone"],
        json!({"number": "+15550100", "vendor_call_id": "vendor-9"})
    );
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
        link.starts_with("https://example.com/demo/join/call?"),
        "{link}"
    );
    assert!(link.contains("user_name=Monitor"), "{link}");
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
    server.route(Method::POST, "/api/v2/video/call/agent/*", 201, json!({}));
    server.route(Method::POST, "/v1/agents/sessions", 201, session("s1"));
    let root = tempfile::tempdir().unwrap();
    jean(root.path());
    let jean = Agent::from_folder(root.path())
        .unwrap()
        .client(server.client())
        .stream(server.stream());

    let (_session, _socket) = opened(&server, async move { jean.join("").await }).await;

    let order: Vec<_> = server
        .seen()
        .into_iter()
        .map(|seen| seen.path)
        .filter(|path| path.starts_with("/v1"))
        .collect();
    assert_eq!(order, ["/v1/agents/sync", "/v1/agents/sessions"]);
    let sent = server.request(Method::POST, "/v1/agents/sessions").body;
    assert_eq!(sent["agent"], "jean");
    assert_eq!(sent["instructions"], "You are Jean.");
    assert_eq!(created_call(&server).len(), 16);
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
    let mut harness = Harness::standard();
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
        json!({"name": "Ada", "instructions": "You are Ada.", "skills": ["think"]})
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
        .add_url("https://example.com/plans", "Plans", "")
        .await
        .unwrap();

    assert_eq!(read.state, types::KnowledgeUrlState::Indexed);
    assert_eq!(read.passages, 12);
    assert_eq!(
        server
            .request(Method::POST, "/v1/agents/knowledge/urls")
            .body,
        json!({"namespace": "jean", "url": "https://example.com/plans", "title": "Plans"})
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
