mod support;

use axum::http::Method;
use serde_json::json;
use support::Server;
use vision_agents::{Ask, Error, Recording, Router, types};

fn router(server: &Server) -> Router {
    Router::new(server.client())
        .config("fast")
        .tags([("env", "test")])
}

#[tokio::test]
async fn a_transcriber_opens_with_a_start_frame_and_hears_transcripts() {
    let server = Server::start().await;
    let router = router(&server);
    let opening = tokio::spawn(async move {
        router
            .transcriber(types::SttOptions {
                target: Some("deepgram/nova-3".into()),
                ..Default::default()
            })
            .await
    });
    let mut socket = server.accept().await;
    let mut transcriber = opening.await.unwrap().unwrap();

    assert_eq!(socket.path, "/v1/stt/stream");
    assert_eq!(
        socket.next().await.unwrap(),
        json!({"type": "start", "config_id": "fast", "target": "deepgram/nova-3", "tags": {"env": "test"}, "sample_rate": 16000,
               "stt": {"target": "deepgram/nova-3"}})
    );
    transcriber.send_audio(&[1, 2, 3, 4]).await.unwrap();
    assert_eq!(socket.next_binary().await, [1, 2, 3, 4]);

    socket
        .send(json!({"type": "transcript", "text": "hello", "final": true}))
        .await;
    let transcript = transcriber.next().await.unwrap().unwrap();
    assert_eq!(transcript.kind(), "transcript");
    assert_eq!(transcript.text("text"), "hello");
    assert!(transcript.flag("final"));
}

#[tokio::test]
async fn a_voice_speaks_and_reads_the_audio_header() {
    let server = Server::start().await;
    let router = router(&server);
    let opening = tokio::spawn(async move { router.voice(Default::default()).await });
    let mut socket = server.accept().await;
    let mut voice = opening.await.unwrap().unwrap();
    assert_eq!(socket.next().await.unwrap()["tts"], json!({}));

    let speaking = tokio::spawn(async move {
        let mut heard = Vec::new();
        voice
            .speak("hello there", |audio| heard.push(audio))
            .await
            .map(|_| heard)
    });
    let speak = socket.expect("speak").await;
    assert_eq!(speak["text"], "hello there");
    assert_eq!(speak["final"], true);
    let mut frame = 24_000u32.to_le_bytes().to_vec();
    frame.extend(1u16.to_le_bytes());
    frame.extend([0, 0, 9, 9]);
    socket.send_binary(frame).await;
    socket.send(json!({"type": "synthesis_complete"})).await;

    let heard = speaking.await.unwrap().unwrap();
    assert_eq!(heard.len(), 1);
    assert_eq!((heard[0].sample_rate, heard[0].channels), (24_000, 1));
    assert_eq!(heard[0].pcm, [9, 9]);
}

#[tokio::test]
async fn a_voice_that_fails_says_why() {
    let server = Server::start().await;
    let router = router(&server);
    let opening = tokio::spawn(async move { router.voice(Default::default()).await });
    let mut socket = server.accept().await;
    let mut voice = opening.await.unwrap().unwrap();

    let speaking = tokio::spawn(async move { voice.speak("hi", |_| {}).await });
    socket.expect("speak").await;
    socket
        .send(json!({"type": "error", "error": "no voice called that"}))
        .await;

    match speaking.await.unwrap().unwrap_err() {
        Error::Failed { message, .. } => assert_eq!(message, "no voice called that"),
        other => panic!("expected a failure, got {other:?}"),
    }
}

#[tokio::test]
async fn a_model_answers_as_it_writes() {
    let server = Server::start().await;
    let router = router(&server);
    let opening = tokio::spawn(async move { router.completions(Default::default()).await });
    let mut socket = server.accept().await;
    let mut model = opening.await.unwrap().unwrap();
    socket.expect("start").await;

    let asking = tokio::spawn(async move {
        let mut written = String::new();
        let complete = model
            .respond(
                &Ask {
                    instructions: "Be brief.".into(),
                    ..Ask::text("hi")
                },
                |delta| written.push_str(delta),
            )
            .await
            .unwrap();
        (written, complete)
    });
    let respond = socket.expect("respond").await;
    assert_eq!(respond["instructions"], "Be brief.");
    assert_eq!(
        respond["messages"],
        json!([{"role": "user", "content": "hi"}])
    );
    let id = respond["id"].clone();
    socket
        .send(json!({"type": "delta", "id": id, "text": "Hel"}))
        .await;
    socket
        .send(json!({"type": "delta", "id": id, "text": "lo"}))
        .await;
    socket
        .send(json!({"type": "complete", "id": id, "text": "Hello", "output_tokens": 2}))
        .await;

    let (written, complete) = asking.await.unwrap();
    assert_eq!(written, "Hello");
    assert_eq!(complete.number("output_tokens"), 2.0);
}

#[tokio::test]
async fn a_search_is_one_question_and_its_answer() {
    let server = Server::start().await;
    server.route(
        Method::POST,
        "/v1/search",
        200,
        json!({"model": "m", "provider": "exa", "results": []}),
    );

    let answer = router(&server).search("who won", None).await.unwrap();

    assert_eq!(answer.provider, "exa");
    assert_eq!(
        server.request(Method::POST, "/v1/search").body,
        json!({"query": "who won", "config_id": "fast", "tags": {"env": "test"}})
    );
}

fn job(status: &str) -> serde_json::Value {
    json!({"id": "job-1", "status": status, "created_at": "2026-09-24T10:00:00Z", "updated_at": "2026-09-24T10:00:00Z",
           "text": "hello", "error": "the file was empty"})
}

#[tokio::test]
async fn a_recording_is_transcribed_and_waited_for() {
    let server = Server::start().await;
    server.route(Method::POST, "/v1/stt/recordings", 202, job("queued"));
    server.route(
        Method::GET,
        "/v1/stt/recordings/job-1",
        200,
        job("completed"),
    );

    let done = router(&server)
        .transcribe(Recording::Audio(vec![0, 1, 2]), Default::default(), None)
        .await
        .unwrap();

    assert_eq!(done.text.as_deref(), Some("hello"));
    let sent = server.request(Method::POST, "/v1/stt/recordings").body;
    assert_eq!(sent["source"], json!({"audio": "AAEC"}));
    assert_eq!(sent["config_id"], "fast");
}

#[tokio::test]
async fn a_recording_that_failed_is_an_error() {
    let server = Server::start().await;
    server.route(Method::POST, "/v1/stt/recordings", 202, job("failed"));

    let failed = router(&server)
        .transcribe(
            Recording::Url("https://example.com/a.wav".into()),
            Default::default(),
            None,
        )
        .await
        .unwrap_err();

    assert!(
        failed.to_string().contains("the file was empty"),
        "{failed}"
    );
}
