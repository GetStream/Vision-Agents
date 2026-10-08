use std::path::PathBuf;

use reqwest::StatusCode;
use reqwest::header::HeaderMap;
use serde_json::Value;

use crate::types;

/// What went wrong, by kind of failure.
///
/// A request that never arrived is [`Error::Transport`] and one the router refused is
/// [`Error::Router`], because a caller retrying a network failure and one retrying a 500 are
/// doing different things.
#[derive(Debug, thiserror::Error)]
#[non_exhaustive]
pub enum Error {
    /// The router, or Stream, answered and said no.
    ///
    /// Boxed, so that every `Result` in the crate does not carry seven fields to say it
    /// succeeded.
    #[error("{0}")]
    Router(Box<RouterFailure>),

    /// The request never got an answer.
    #[error("{operation}: {source}")]
    Transport {
        operation: String,
        #[source]
        source: reqwest::Error,
    },

    /// An answer arrived that is not what the spec says it should be.
    #[error("{operation}: the answer did not decode: {source}")]
    Decode {
        operation: String,
        #[source]
        source: serde_json::Error,
    },

    /// A socket could not be opened, or failed while open.
    #[error("socket: {0}")]
    Socket(#[from] Box<tokio_tungstenite::tungstenite::Error>),

    /// The router reported a failure: an `error` frame on a socket, or a job that failed.
    #[error("{operation}: {message}")]
    Failed { operation: String, message: String },

    /// Something was asked of a socket or a session that has already ended.
    #[error("{0} has already closed")]
    Closed(String),

    /// What was asked for contradicts itself, or is missing something, and was refused
    /// before anything was sent.
    #[error("{0}")]
    Configuration(String),

    /// An agent directory could not be read as one.
    #[error("{}: {message}", path.display())]
    Folder { path: PathBuf, message: String },

    #[error("{}: {source}", path.display())]
    Io {
        path: PathBuf,
        #[source]
        source: std::io::Error,
    },
}

/// What the router, or Stream, said when it refused.
///
/// The router answers every failure with an envelope, read into `message`, `kind`, `code`
/// and `doc_url`. A body that is not one (a proxy's page, an empty answer) leaves the last
/// three empty and is the message itself.
#[derive(Debug, Clone, PartialEq, Eq)]
#[non_exhaustive]
pub struct RouterFailure {
    pub status: u16,
    pub operation: String,
    /// What the far end said went wrong, or its status line when it said nothing.
    pub message: String,
    /// The envelope's `type`, which decides the status: `invalid_request`, `not_found`,
    /// `internal`, ... A string, so a type newer than this build is still read.
    pub kind: String,
    /// What went wrong, for a program to branch on: `not_configured`, `session_not_found`,
    /// ... More appear over time.
    pub code: String,
    /// Where `code` is explained.
    pub doc_url: String,
    /// The answer's `X-Request-Id`, which is what to quote to support: a 500 says only
    /// "something went wrong". Empty when there was none.
    pub request_id: String,
}

impl std::fmt::Display for RouterFailure {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            formatter,
            "{}: {}: {}",
            self.operation, self.status, self.message
        )
    }
}

impl Error {
    /// The HTTP status the far end answered with, when it answered.
    pub fn status(&self) -> Option<u16> {
        match self {
            Error::Router(failure) => Some(failure.status),
            _ => None,
        }
    }

    /// The refusal an answer outside 2xx is, whether to a request or to a socket upgrade.
    pub(crate) fn refused(
        operation: impl Into<String>,
        status: StatusCode,
        headers: &HeaderMap,
        body: &[u8],
    ) -> Self {
        let (message, kind, code, doc_url) = envelope(body).unwrap_or_else(|| {
            let text = String::from_utf8_lossy(body).trim().to_string();
            (text, String::new(), String::new(), String::new())
        });
        Error::Router(Box::new(RouterFailure {
            status: status.as_u16(),
            operation: operation.into(),
            message: if message.is_empty() {
                status.to_string()
            } else {
                message
            },
            kind,
            code,
            doc_url,
            request_id: headers
                .get("x-request-id")
                .and_then(|value| value.to_str().ok())
                .unwrap_or_default()
                .to_string(),
        }))
    }

    pub(crate) fn configuration(message: impl Into<String>) -> Self {
        Error::Configuration(message.into())
    }

    pub(crate) fn folder(path: impl Into<PathBuf>, message: impl Into<String>) -> Self {
        Error::Folder {
            path: path.into(),
            message: message.into(),
        }
    }

    pub(crate) fn io(path: impl Into<PathBuf>, source: std::io::Error) -> Self {
        Error::Io {
            path: path.into(),
            source,
        }
    }
}

/// The message, type, code and doc_url of the router's error envelope, or `None` for a body
/// that is not one.
///
/// The spec's `ErrorType` refuses a type it does not list, so an envelope carrying a newer one
/// is read field by field rather than taken for some other body.
fn envelope(body: &[u8]) -> Option<(String, String, String, String)> {
    if let Ok(types::ErrorResponse { error }) = serde_json::from_slice(body) {
        return Some((
            error.message,
            error.type_.to_string(),
            error.code,
            error.doc_url,
        ));
    }
    let said: Value = serde_json::from_slice(body).ok()?;
    let error = said.get("error")?.as_object()?;
    let text = |key: &str| {
        error
            .get(key)
            .and_then(Value::as_str)
            .unwrap_or_default()
            .to_string()
    };
    Some((text("message"), text("type"), text("code"), text("doc_url")))
}

impl From<tokio_tungstenite::tungstenite::Error> for Error {
    fn from(error: tokio_tungstenite::tungstenite::Error) -> Self {
        Error::Socket(Box::new(error))
    }
}

pub type Result<T, E = Error> = std::result::Result<T, E>;
