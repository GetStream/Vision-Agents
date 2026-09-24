//! The Rust server-side SDK for the Vision Agents acceleration backend.
//!
//! The backend joins the call, hears the caller, answers and speaks. What runs here is the
//! configuration of an agent, the caller's own functions, and whatever watches or steers
//! the conversation: [`Agent`] and [`Session`] for conversations, [`Dispatch`] for inbound
//! calls and messages, [`Router`] for one modality at a time, and [`Client`] for every
//! endpoint in the spec.

mod agent;
mod backend;
mod client;
mod dispatch;
mod error;
pub mod folder;
mod harness;
mod knowledge;
// Generated files keep the generator's formatting, so `generate --check` and `cargo fmt --check` agree.
#[rustfmt::skip]
mod operations;
mod responses;
mod router;
mod session;
mod sessions;
mod socket;
mod stream;
mod tools;
#[rustfmt::skip]
pub mod types;

pub use agent::{Agent, USER_KEY, user_id_of};
pub use backend::{ClientOptions, DEFAULT_URL, TOKEN_VALIDITY_SECONDS, sign, sign_for};
pub use client::{Client, segment};
pub use dispatch::{Dispatch, InboundCall, InboundMessage};
pub use error::{Error, Result};
pub use folder::Folder;
pub use harness::{Harness, Sandbox, Skill, daytona};
pub use knowledge::Knowledge;
pub use operations::*;
pub use responses::{AgentResponse, Items, ResponseRef, Responses};
pub use router::{Ask, Audio, Completions, Recording, Router, SAMPLE_RATE, Transcriber, Voice};
pub use session::{Participant, Session, SessionEvent, WatchOptions};
pub use sessions::{AgentRef, Sessions};
pub use socket::{Frame, Incoming, Socket, SocketReceiver, SocketSender};
pub use stream::{Call, DEFAULT_CALL_TYPE, STREAM_API, StreamApp};
pub use tools::Tools;
