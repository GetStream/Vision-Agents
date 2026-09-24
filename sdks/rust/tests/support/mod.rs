//! A real HTTP and WebSocket server, in process, standing in for the router and for Stream.
//!
//! It answers from routes a test registers and writes down every request it was sent, so a
//! test asserts on what arrived rather than on what was called. A socket is handed to the
//! test whole, which then plays the router's side of the conversation frame by frame.
#![allow(dead_code)]

use std::collections::{HashMap, VecDeque};
use std::sync::{Arc, Mutex};
use std::time::Duration;

use axum::Router;
use axum::body::Bytes;
use axum::extract::State;
use axum::extract::ws::rejection::WebSocketUpgradeRejection;
use axum::extract::ws::{Message, WebSocket, WebSocketUpgrade};
use axum::http::{HeaderMap, Method, StatusCode, Uri};
use axum::response::{IntoResponse, Response};
use serde_json::{Value, json};
use tokio::sync::mpsc;
use vision_agents::{Client, ClientOptions, StreamApp};

/// One request as it arrived.
#[derive(Debug, Clone)]
pub struct Seen {
    pub method: Method,
    pub path: String,
    pub query: String,
    pub headers: HeaderMap,
    pub body: Value,
}

impl Seen {
    pub fn header(&self, name: &str) -> &str {
        self.headers
            .get(name)
            .and_then(|value| value.to_str().ok())
            .unwrap_or("")
    }
}

/// A socket the SDK opened, with the handshake it opened it with.
pub struct Accepted {
    pub path: String,
    pub query: String,
    pub headers: HeaderMap,
    socket: WebSocket,
}

impl Accepted {
    pub fn header(&self, name: &str) -> &str {
        self.headers
            .get(name)
            .and_then(|value| value.to_str().ok())
            .unwrap_or("")
    }

    pub async fn send(&mut self, frame: Value) {
        self.socket
            .send(Message::text(frame.to_string()))
            .await
            .expect("sending a frame");
    }

    pub async fn send_binary(&mut self, bytes: Vec<u8>) {
        self.socket
            .send(Message::binary(bytes))
            .await
            .expect("sending audio");
    }

    /// The next JSON frame the SDK sent, or `None` once it has closed the socket.
    pub async fn next(&mut self) -> Option<Value> {
        loop {
            let message = tokio::time::timeout(Duration::from_secs(10), self.socket.recv())
                .await
                .expect("the SDK sent nothing for ten seconds");
            match message {
                Some(Ok(Message::Text(text))) => {
                    return Some(serde_json::from_str(text.as_str()).expect("a JSON frame"));
                }
                Some(Ok(Message::Close(_))) | None | Some(Err(_)) => return None,
                _ => continue,
            }
        }
    }

    /// The next binary frame the SDK sent.
    pub async fn next_binary(&mut self) -> Vec<u8> {
        loop {
            match self.socket.recv().await {
                Some(Ok(Message::Binary(bytes))) => return bytes.to_vec(),
                Some(Ok(Message::Text(_)))
                | Some(Ok(Message::Ping(_)))
                | Some(Ok(Message::Pong(_))) => continue,
                other => panic!("expected audio, got {other:?}"),
            }
        }
    }

    /// Reads frames until one of `kind` arrives.
    pub async fn expect(&mut self, kind: &str) -> Value {
        while let Some(frame) = self.next().await {
            if frame["type"] == kind {
                return frame;
            }
        }
        panic!("the socket closed before a {kind} frame arrived");
    }

    pub async fn close(mut self) {
        let _ = self.socket.send(Message::Close(None)).await;
    }
}

/// The answers queued for each route: a status and a body.
type Routes = HashMap<(Method, String), VecDeque<(u16, String)>>;

#[derive(Default)]
struct Shared {
    routes: Mutex<Routes>,
    seen: Mutex<Vec<Seen>>,
    sockets: Mutex<Option<mpsc::UnboundedSender<Accepted>>>,
}

pub struct Server {
    pub url: String,
    shared: Arc<Shared>,
    sockets: tokio::sync::Mutex<mpsc::UnboundedReceiver<Accepted>>,
}

impl Server {
    pub async fn start() -> Server {
        let shared = Arc::new(Shared::default());
        let (tx, rx) = mpsc::unbounded_channel();
        *shared.sockets.lock().unwrap() = Some(tx);

        let app = Router::new().fallback(handle).with_state(shared.clone());
        let listener = tokio::net::TcpListener::bind("127.0.0.1:0")
            .await
            .expect("binding");
        let url = format!("http://{}", listener.local_addr().unwrap());
        tokio::spawn(async move { axum::serve(listener, app).await.unwrap() });
        Server {
            url,
            shared,
            sockets: tokio::sync::Mutex::new(rx),
        }
    }

    /// Answers `method path` with `status` and `body`. A path ending in `*` answers every
    /// path it prefixes. Registering the same route again queues the answers in order; the
    /// last one keeps being given.
    pub fn route(&self, method: Method, path: &str, status: u16, body: Value) -> &Self {
        let body = if body.is_null() {
            String::new()
        } else {
            body.to_string()
        };
        self.shared
            .routes
            .lock()
            .unwrap()
            .entry((method, path.into()))
            .or_default()
            .push_back((status, body));
        self
    }

    /// Every request so far.
    pub fn seen(&self) -> Vec<Seen> {
        self.shared.seen.lock().unwrap().clone()
    }

    /// The requests to one route.
    pub fn requests(&self, method: Method, path: &str) -> Vec<Seen> {
        self.seen()
            .into_iter()
            .filter(|seen| seen.method == method && seen.path == path)
            .collect()
    }

    /// The single request to a route, failing the test if there was not exactly one.
    pub fn request(&self, method: Method, path: &str) -> Seen {
        let mut found = self.requests(method.clone(), path);
        assert_eq!(
            found.len(),
            1,
            "{method} {path} was sent {} times; everything sent: {:#?}",
            found.len(),
            self.seen()
        );
        found.remove(0)
    }

    /// The next socket the SDK opens.
    pub async fn accept(&self) -> Accepted {
        tokio::time::timeout(Duration::from_secs(10), self.sockets.lock().await.recv())
            .await
            .expect("no socket was opened within ten seconds")
            .expect("the server stopped")
    }

    /// A client for a router with nothing in front of it.
    pub fn client(&self) -> Client {
        Client::new(ClientOptions {
            url: Some(self.url.clone()),
            customer_id: Some("examples".into()),
            ..Default::default()
        })
        .unwrap()
    }

    /// A client holding a key and secret, as a backend does.
    pub fn backend(&self) -> Client {
        Client::new(ClientOptions {
            url: Some(self.url.clone()),
            api_key: Some("key".into()),
            api_secret: Some(SECRET.into()),
            ..Default::default()
        })
        .unwrap()
    }

    /// Stream, served from this same server.
    pub fn stream(&self) -> StreamApp {
        StreamApp::new("key", SECRET)
            .unwrap()
            .with_base_url(&self.url)
            .with_monitor_url("https://example.com/demo")
    }
}

pub const SECRET: &str = "a-secret-long-enough-to-sign-with";

async fn handle(
    State(shared): State<Arc<Shared>>,
    method: Method,
    uri: Uri,
    headers: HeaderMap,
    upgrade: Result<WebSocketUpgrade, WebSocketUpgradeRejection>,
    body: Bytes,
) -> Response {
    let path = uri.path().to_string();
    let query = uri.query().unwrap_or("").to_string();

    // A socket path with a route registered is refused with that route's answer instead.
    let refused = shared
        .routes
        .lock()
        .unwrap()
        .contains_key(&(method.clone(), path.clone()));
    if let (Ok(upgrade), false) = (upgrade, refused) {
        let sockets = shared.sockets.lock().unwrap().clone().expect("sockets");
        return upgrade.on_upgrade(move |socket| async move {
            let _ = sockets.send(Accepted {
                path,
                query,
                headers,
                socket,
            });
        });
    }

    let body_json = if body.is_empty() {
        Value::Null
    } else {
        serde_json::from_slice(&body).unwrap_or(json!(String::from_utf8_lossy(&body)))
    };
    shared.seen.lock().unwrap().push(Seen {
        method: method.clone(),
        path: path.clone(),
        query,
        headers,
        body: body_json,
    });

    let answer = {
        let mut routes = shared.routes.lock().unwrap();
        let key = routes
            .keys()
            .find(|(each, route)| {
                *each == method
                    && (*route == path
                        || route
                            .strip_suffix('*')
                            .is_some_and(|prefix| path.starts_with(prefix)))
            })
            .cloned();
        key.and_then(|key| routes.get_mut(&key)).map(|queue| {
            if queue.len() > 1 {
                queue.pop_front().unwrap()
            } else {
                queue.front().cloned().unwrap()
            }
        })
    };
    match answer {
        Some((status, body)) => {
            let status = StatusCode::from_u16(status).unwrap();
            if body.is_empty() {
                status.into_response()
            } else {
                (status, [("content-type", "application/json")], body).into_response()
            }
        }
        None => (
            StatusCode::NOT_FOUND,
            [("content-type", "application/json")],
            json!({"error": format!("nothing at {path}")}).to_string(),
        )
            .into_response(),
    }
}

/// A session as the router describes one.
pub fn session(id: &str) -> Value {
    json!({
        "id": id,
        "agent_id": "agent",
        "call_id": "call",
        "call_type": "agent",
        "created_at": "2026-09-24T10:00:00Z",
        "state": "live",
        "user_id": "user",
    })
}

pub fn response(id: &str, session: &str) -> Value {
    json!({"id": id, "session_id": session, "created_at": "2026-09-24T10:00:00Z", "status": "running"})
}

pub fn item(ordinal: i64, response: &str) -> Value {
    json!({"at": "2026-09-24T10:00:00Z", "kind": "said", "ordinal": ordinal, "response_id": response, "text": "hi"})
}

pub fn config(id: &str, name: &str) -> Value {
    json!({
        "id": id,
        "name": name,
        "mode": "voice",
        "created_at": "2026-09-24T10:00:00Z",
        "updated_at": "2026-09-24T10:00:00Z",
    })
}

/// The claims of a token this SDK signed, after checking the signature.
pub fn claims(token: &str) -> Value {
    use base64::Engine as _;
    use base64::engine::general_purpose::URL_SAFE_NO_PAD;
    use hmac::{Hmac, KeyInit, Mac};

    let token = token.strip_prefix("Bearer ").unwrap_or(token);
    let (signing, signature) = token.rsplit_once('.').expect("a signed token");
    let mut mac = Hmac::<sha2::Sha256>::new_from_slice(SECRET.as_bytes()).unwrap();
    mac.update(signing.as_bytes());
    mac.verify_slice(&URL_SAFE_NO_PAD.decode(signature).unwrap())
        .expect("the token is signed with the secret");
    let payload = signing.split('.').nth(1).unwrap();
    serde_json::from_slice(&URL_SAFE_NO_PAD.decode(payload).unwrap()).unwrap()
}
