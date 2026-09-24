//! The hand-written half of the transport. OpenAPI stops at the upgrade, so the session
//! events socket, the dispatch socket and the modality sockets are written rather than
//! generated.

use std::ops::Deref;
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, Ordering};

use futures_util::stream::{SplitSink, SplitStream};
use futures_util::{SinkExt, StreamExt};
use serde_json::{Map, Value};
use tokio::net::TcpStream;
use tokio::sync::Mutex;
use tokio_tungstenite::tungstenite::client::IntoClientRequest;
use tokio_tungstenite::tungstenite::http::{HeaderName, HeaderValue};
use tokio_tungstenite::tungstenite::{self, Message};
use tokio_tungstenite::{MaybeTlsStream, WebSocketStream};

use crate::error::{Error, Result};

type Stream = WebSocketStream<MaybeTlsStream<TcpStream>>;

/// One JSON frame, typed loosely on purpose.
///
/// A deployment that has learned a new event must reach a caller reading [`Frame::kind`]
/// rather than be dropped here, so frames are a map with readers for the fields most of them
/// carry, and the map itself for the rest.
#[derive(Debug, Clone, Default, PartialEq)]
pub struct Frame(pub Map<String, Value>);

impl Frame {
    /// The frame's `type`.
    pub fn kind(&self) -> &str {
        self.text("type")
    }

    /// A string field, or empty when it is missing or not a string.
    pub fn text(&self, key: &str) -> &str {
        self.0.get(key).and_then(Value::as_str).unwrap_or("")
    }

    pub fn flag(&self, key: &str) -> bool {
        self.0.get(key).and_then(Value::as_bool).unwrap_or(false)
    }

    pub fn number(&self, key: &str) -> f64 {
        self.0.get(key).and_then(Value::as_f64).unwrap_or(0.0)
    }

    /// An object field, as a frame of its own.
    pub fn nested(&self, key: &str) -> Option<Frame> {
        self.0
            .get(key)
            .and_then(Value::as_object)
            .map(|object| Frame(object.clone()))
    }
}

impl Deref for Frame {
    type Target = Map<String, Value>;

    fn deref(&self) -> &Self::Target {
        &self.0
    }
}

/// What arrives on a socket: a JSON frame, or audio.
#[derive(Debug, Clone, PartialEq)]
pub enum Incoming {
    Frame(Frame),
    Audio(Vec<u8>),
}

/// A socket on the router, split so one task can read while any number send.
///
/// There is no automatic reconnection. `respond` and `tool_result` are not idempotent and
/// the protocol has no sequence number to resume from, so replaying would duplicate turns.
pub struct Socket {
    pub sender: SocketSender,
    pub receiver: SocketReceiver,
}

impl Socket {
    /// Opens a socket, sending `headers` on the handshake. A refused upgrade is reported as
    /// the status and whatever the router said, the way a refused request is.
    pub async fn connect(url: &str, headers: Vec<(&'static str, String)>) -> Result<Socket> {
        let mut request = url.into_client_request()?;
        for (name, value) in headers {
            let name = HeaderName::from_bytes(name.as_bytes())
                .map_err(|error| Error::configuration(error.to_string()))?;
            let value = HeaderValue::from_str(&value)
                .map_err(|error| Error::configuration(error.to_string()))?;
            request.headers_mut().insert(name, value);
        }

        let path = request.uri().path().to_string();
        let (stream, _) = match tokio_tungstenite::connect_async(request).await {
            Ok(connected) => connected,
            Err(tungstenite::Error::Http(response)) => {
                let status = response.status().as_u16();
                let body = response
                    .body()
                    .as_deref()
                    .map(String::from_utf8_lossy)
                    .unwrap_or_default();
                let message = serde_json::from_str::<Value>(&body)
                    .ok()
                    .and_then(|said| {
                        said.get("error")
                            .and_then(Value::as_str)
                            .map(str::to_string)
                    })
                    .unwrap_or_else(|| body.trim().to_string());
                return Err(Error::Router {
                    status,
                    operation: format!("GET {path}"),
                    message,
                });
            }
            Err(error) => return Err(error.into()),
        };

        let (sink, stream) = stream.split();
        let open = Arc::new(AtomicBool::new(true));
        Ok(Socket {
            sender: SocketSender {
                sink: Arc::new(Mutex::new(sink)),
                open: open.clone(),
            },
            receiver: SocketReceiver { stream, open },
        })
    }
}

/// The sending half. Cheap to clone; clones send on the same socket.
#[derive(Clone)]
pub struct SocketSender {
    sink: Arc<Mutex<SplitSink<Stream, Message>>>,
    open: Arc<AtomicBool>,
}

impl SocketSender {
    pub fn open(&self) -> bool {
        self.open.load(Ordering::Acquire)
    }

    /// Sends one JSON frame.
    pub async fn send(&self, frame: &Value) -> Result<()> {
        self.write(Message::text(frame.to_string())).await
    }

    /// Sends audio, which the router reads as binary PCM.
    pub async fn send_audio(&self, pcm: &[u8]) -> Result<()> {
        self.write(Message::binary(pcm.to_vec())).await
    }

    /// Closes the socket. Safe to call more than once.
    pub async fn close(&self) {
        if self.open.swap(false, Ordering::AcqRel) {
            let _ = self.sink.lock().await.close().await;
        }
    }

    async fn write(&self, message: Message) -> Result<()> {
        if !self.open() {
            return Err(Error::Closed("the socket".into()));
        }
        self.sink
            .lock()
            .await
            .send(message)
            .await
            .map_err(Error::from)
    }
}

/// The receiving half. One reader: two loops over one socket would take half the frames each.
pub struct SocketReceiver {
    stream: SplitStream<Stream>,
    open: Arc<AtomicBool>,
}

impl SocketReceiver {
    /// The next frame or piece of audio, or `None` once the socket has closed.
    ///
    /// A text frame that is not JSON is skipped rather than fatal: ending the stream over it
    /// would lose everything said after it.
    pub async fn next(&mut self) -> Option<Result<Incoming>> {
        loop {
            let message = match self.stream.next().await {
                Some(Ok(message)) => message,
                Some(Err(error)) => {
                    self.open.store(false, Ordering::Release);
                    return Some(Err(error.into()));
                }
                None => {
                    self.open.store(false, Ordering::Release);
                    return None;
                }
            };
            match message {
                Message::Text(text) => {
                    if let Ok(Value::Object(object)) = serde_json::from_str(text.as_str()) {
                        return Some(Ok(Incoming::Frame(Frame(object))));
                    }
                }
                Message::Binary(audio) => return Some(Ok(Incoming::Audio(audio.to_vec()))),
                Message::Close(_) => {
                    self.open.store(false, Ordering::Release);
                    return None;
                }
                _ => {}
            }
        }
    }
}
