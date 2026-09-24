//! Against a real router, when `VISION_AGENTS_URL` names one. Skipped otherwise.
//!
//! `VISION_AGENTS_CUSTOMER_ID` is who the work is billed to (`examples` by default). The
//! call test also needs `STREAM_API_KEY` and `STREAM_API_SECRET`.

use std::time::Duration;

use vision_agents::{Agent, Ask, Client, ClientOptions, Router, StreamApp, WatchOptions, types};

fn client() -> Option<Client> {
    let url = std::env::var("VISION_AGENTS_URL")
        .ok()
        .filter(|url| !url.is_empty())?;
    let customer_id =
        std::env::var("VISION_AGENTS_CUSTOMER_ID").unwrap_or_else(|_| "examples".into());
    Some(
        Client::new(ClientOptions {
            url: Some(url),
            customer_id: Some(customer_id),
            ..Default::default()
        })
        .unwrap(),
    )
}

macro_rules! live {
    () => {
        match client() {
            Some(client) => client,
            None => {
                eprintln!("VISION_AGENTS_URL is not set; skipping");
                return;
            }
        }
    };
}

#[tokio::test]
async fn the_router_is_up() {
    let client = live!();

    // A degraded router answers 503 with the same body, naming the dependency that is down.
    match client.get_health().await {
        Ok(health) => assert!(!health.dependencies.is_empty()),
        Err(error) => {
            assert_eq!(error.status(), Some(503), "{error}");
            assert!(error.to_string().contains("degraded"), "{error}");
        }
    }
}

#[tokio::test]
async fn a_question_is_searched() {
    let client = live!();

    let answer = Router::new(client)
        .search("What is the capital of France?", None)
        .await
        .unwrap();

    assert!(!answer.provider.is_empty());
}

#[tokio::test]
async fn a_model_answers_over_its_socket() {
    let client = live!();
    let options = types::LlmOptions {
        target: Some("llm-fast".into()),
        ..Default::default()
    };
    let mut model = Router::new(client).completions(options).await.unwrap();

    let mut written = String::new();
    let complete = tokio::time::timeout(
        Duration::from_secs(60),
        model.respond(
            &Ask {
                max_tokens: Some(20),
                ..Ask::text("Say the word hello.")
            },
            |delta| written.push_str(delta),
        ),
    )
    .await
    .unwrap()
    .unwrap();
    model.close().await;

    assert!(
        !complete.text("text").is_empty() || !written.is_empty(),
        "{complete:?}"
    );
}

#[tokio::test]
async fn a_voice_speaks_over_its_socket() {
    let client = live!();
    let options = types::TtsOptions {
        target: Some("en-low-latency".into()),
        ..Default::default()
    };
    let mut voice = Router::new(client).voice(options).await.unwrap();

    let mut bytes = 0;
    tokio::time::timeout(
        Duration::from_secs(60),
        voice.speak("Hello there.", |audio| bytes += audio.pcm.len()),
    )
    .await
    .unwrap()
    .unwrap();
    voice.close().await;

    assert!(bytes > 0);
}

#[tokio::test]
async fn a_text_conversation_answers() {
    let client = live!();
    let agent = Agent::named("Rust Live")
        .client(client)
        .instructions("Answer in five words or fewer.");

    let session = agent
        .chat_with(types::CreateSessionRequest {
            incognito: Some(true),
            ..Default::default()
        })
        .await
        .unwrap();
    session.respond("What colour is the sky?").await.unwrap();
    let answered =
        tokio::time::timeout(Duration::from_secs(60), session.wait_for_event("responded"))
            .await
            .unwrap()
            .expect("the conversation ended before answering");
    session.close().await;

    assert!(!answered.text.is_empty(), "{answered:?}");
}

#[tokio::test]
async fn a_recorded_conversation_is_read_back_and_forked() {
    let client = live!();
    let agent = Agent::named("Rust Live")
        .client(client)
        .instructions("Answer in five words or fewer.");

    let session = agent.chat().await.unwrap();
    let turn = session
        .responses
        .create("What colour is grass?")
        .await
        .unwrap();
    tokio::time::timeout(Duration::from_secs(60), session.wait_for_event("responded"))
        .await
        .unwrap()
        .unwrap();
    let items = turn.items.all().await.unwrap();
    let fork = session
        .fork(&types::ForkSessionRequest {
            response_id: Some(turn.id().into()),
            ..Default::default()
        })
        .await
        .unwrap();
    fork.close().await;
    session.close().await;

    assert!(!items.is_empty());
}

#[tokio::test]
async fn an_agent_joins_a_stream_call() {
    let client = live!();
    let Ok(stream) = StreamApp::from_env() else {
        eprintln!("STREAM_API_KEY and STREAM_API_SECRET are not set; skipping");
        return;
    };
    let agent = Agent::named("Rust Live")
        .client(client)
        .stream(stream)
        .instructions("Say hello once.")
        .watch(WatchOptions::default());

    let session = agent.join("").await.unwrap();
    let url = agent.monitor_url(&session).unwrap();
    session.close().await;

    assert!(url.contains("/join/"));
}
