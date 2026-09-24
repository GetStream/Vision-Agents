//! The two things this SDK needs from Stream itself: creating the call an agent joins, and
//! minting the tokens for joining it.
//!
//! Stream's official Rust crate, `getstream`, is a preview whose SFU and WebRTC stack is not
//! optional: it builds libvpx, openh264 and opus into every dependent. This SDK touches no
//! media, so the one endpoint it needs is called directly.

use serde_json::json;

use crate::backend::{API_KEY_ENV, API_SECRET_ENV, env, sign, sign_for};
use crate::client::segment;
use crate::error::{Error, Result};

/// Where Stream's API is. The video paths are served from the same host as chat's.
pub const STREAM_API: &str = "https://chat.stream-io-api.com";

/// The Stream call type used when none is named.
pub const DEFAULT_CALL_TYPE: &str = "agent";

/// Stream's hosted video demo, which a monitoring link opens.
pub const DEFAULT_MONITOR_URL: &str = "https://getstream.io/video/demos";

/// How long a monitoring token lasts. A call outliving it is a call nobody is still on.
const MONITOR_TOKEN_VALIDITY: u64 = 60 * 60;

/// One Stream call, named the way the backend needs it named.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct Call {
    /// Empty names a new call after a random id.
    pub id: String,
    /// Empty is [`DEFAULT_CALL_TYPE`].
    pub kind: String,
}

impl Call {
    pub fn new(id: impl Into<String>) -> Self {
        Call {
            id: id.into(),
            kind: String::new(),
        }
    }
}

impl From<&str> for Call {
    fn from(id: &str) -> Self {
        Call::new(id)
    }
}

impl From<String> for Call {
    fn from(id: String) -> Self {
        Call::new(id)
    }
}

/// A Stream app, as far as creating calls and minting tokens goes.
///
/// Server side only, because it holds the app secret.
#[derive(Debug, Clone)]
pub struct StreamApp {
    api_key: String,
    api_secret: String,
    base_url: String,
    monitor_url: String,
    http: reqwest::Client,
}

impl StreamApp {
    pub fn new(api_key: impl Into<String>, api_secret: impl Into<String>) -> Result<Self> {
        let (api_key, api_secret) = (api_key.into(), api_secret.into());
        if api_key.is_empty() || api_secret.is_empty() {
            return Err(Error::configuration(format!(
                "{API_KEY_ENV} and {API_SECRET_ENV} are required to create a call"
            )));
        }
        Ok(StreamApp {
            api_key,
            api_secret,
            base_url: STREAM_API.into(),
            monitor_url: env("EXAMPLE_BASE_URL").unwrap_or_else(|| DEFAULT_MONITOR_URL.into()),
            http: reqwest::Client::new(),
        })
    }

    /// The app named by `STREAM_API_KEY` and `STREAM_API_SECRET`.
    pub fn from_env() -> Result<Self> {
        StreamApp::new(
            env(API_KEY_ENV).unwrap_or_default(),
            env(API_SECRET_ENV).unwrap_or_default(),
        )
    }

    /// Points the app at another Stream API host.
    pub fn with_base_url(mut self, url: impl Into<String>) -> Self {
        self.base_url = url.into().trim_end_matches('/').to_string();
        self
    }

    /// Points monitoring links at another page. Falls back to `EXAMPLE_BASE_URL`.
    pub fn with_monitor_url(mut self, url: impl Into<String>) -> Self {
        self.monitor_url = url.into();
        self
    }

    pub fn api_key(&self) -> &str {
        &self.api_key
    }

    /// Creates the call the backend will join, or returns the one already under that id.
    pub async fn create_call(&self, call: &Call, created_by: &str) -> Result<Call> {
        if created_by.is_empty() {
            return Err(Error::configuration(
                "a call needs somebody to have created it",
            ));
        }
        let named = Call {
            id: if call.id.is_empty() {
                random_id()
            } else {
                call.id.clone()
            },
            kind: if call.kind.is_empty() {
                DEFAULT_CALL_TYPE.into()
            } else {
                call.kind.clone()
            },
        };

        let operation = format!("POST /api/v2/video/call/{}/{}", named.kind, named.id);
        let response = self
            .http
            .post(format!(
                "{}/api/v2/video/call/{}/{}",
                self.base_url,
                segment(&named.kind),
                segment(&named.id)
            ))
            .query(&[("api_key", &self.api_key)])
            .header(
                "Authorization",
                sign(json!({"server": true}), &self.api_secret)?,
            )
            .header("Stream-Auth-Type", "jwt")
            .json(&json!({"data": {"created_by_id": created_by}}))
            .send()
            .await
            .map_err(|source| Error::Transport {
                operation: operation.clone(),
                source,
            })?;

        let status = response.status();
        if !status.is_success() {
            let said = response.text().await.unwrap_or_default();
            return Err(Error::Router {
                status: status.as_u16(),
                operation,
                message: if said.trim().is_empty() {
                    status.to_string()
                } else {
                    said.trim().to_string()
                },
            });
        }
        Ok(named)
    }

    /// A token for `user_id` to connect to Stream's chat and video with.
    pub fn user_token(&self, user_id: &str, validity_seconds: u64) -> Result<String> {
        if user_id.is_empty() {
            return Err(Error::configuration("a token needs a user to name"));
        }
        sign_for(
            json!({"user_id": user_id}),
            &self.api_secret,
            validity_seconds,
        )
    }

    /// A link a person can open to join a call from a browser and hear the agent.
    ///
    /// They join as a listener of their own rather than as the agent, so opening it twice
    /// puts two people in the call instead of taking the first one's place.
    pub fn monitor_url(&self, call: &Call, user_id: &str, user_name: &str) -> Result<String> {
        if call.id.is_empty() {
            return Err(Error::configuration("there is no call to watch"));
        }
        let token = self.user_token(user_id, MONITOR_TOKEN_VALIDITY)?;
        let mut url = url::Url::parse(&format!(
            "{}/join/{}",
            self.monitor_url.trim_end_matches('/'),
            segment(&call.id)
        ))
        .map_err(|error| Error::configuration(format!("the monitoring url: {error}")))?;
        url.query_pairs_mut()
            .append_pair("api_key", &self.api_key)
            .append_pair("token", &token)
            .append_pair("skip_lobby", "true")
            .append_pair(
                "user_name",
                if user_name.is_empty() {
                    user_id
                } else {
                    user_name
                },
            );
        Ok(url.into())
    }
}

fn random_id() -> String {
    let bytes: [u8; 8] = rand::random();
    bytes.iter().map(|byte| format!("{byte:02x}")).collect()
}
