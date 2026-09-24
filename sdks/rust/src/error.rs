use std::path::PathBuf;

/// What went wrong, by kind of failure.
///
/// A request that never arrived is [`Error::Transport`] and one the router refused is
/// [`Error::Router`], because a caller retrying a network failure and one retrying a 500 are
/// doing different things.
#[derive(Debug, thiserror::Error)]
#[non_exhaustive]
pub enum Error {
    /// The router, or Stream, answered and said no.
    #[error("{operation}: {status}: {message}")]
    Router {
        status: u16,
        operation: String,
        /// What the far end said went wrong, or its status line when it said nothing.
        message: String,
    },

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

impl Error {
    /// The HTTP status the far end answered with, when it answered.
    pub fn status(&self) -> Option<u16> {
        match self {
            Error::Router { status, .. } => Some(*status),
            _ => None,
        }
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

impl From<tokio_tungstenite::tungstenite::Error> for Error {
    fn from(error: tokio_tungstenite::tungstenite::Error) -> Self {
        Error::Socket(Box::new(error))
    }
}

pub type Result<T, E = Error> = std::result::Result<T, E>;
