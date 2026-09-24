use std::time::{SystemTime, UNIX_EPOCH};

use base64::Engine as _;
use base64::engine::general_purpose::URL_SAFE_NO_PAD;
use hmac::{Hmac, KeyInit, Mac};
use serde_json::{Value, json};
use sha2::Sha256;

use crate::error::{Error, Result};

pub const URL_ENV: &str = "STREAM_ACCELERATION_URL";
pub const CUSTOMER_ENV: &str = "STREAM_ACCELERATION_CUSTOMER_ID";
pub const API_KEY_ENV: &str = "STREAM_API_KEY";
pub const API_SECRET_ENV: &str = "STREAM_API_SECRET";
pub const AUTHENTICATE_ENV: &str = "STREAM_ACCELERATION_AUTHENTICATE";

/// Where the router is when nothing says otherwise.
pub const DEFAULT_URL: &str = "http://localhost:8080";

/// How long a token minted here lasts. Short, because one is minted per request.
pub const TOKEN_VALIDITY_SECONDS: u64 = 60 * 60;

/// Where the router is and who is calling it.
///
/// Three ways to say who that is, and which one a deployment takes is a property of the
/// deployment rather than a choice: a customer id for a router with nothing in front of it,
/// a key and secret for a process the customer runs, and a key and a user token for anything
/// acting on one person's behalf. Every field left `None` falls back to the environment the
/// Go, Python and JavaScript SDKs read.
#[derive(Debug, Clone, Default)]
pub struct ClientOptions {
    /// The router's base URL. Falls back to `STREAM_ACCELERATION_URL`, then localhost.
    pub url: Option<String>,
    /// Who the work is billed to, taken at face value, for a router with no keys in front
    /// of it. Falls back to `STREAM_ACCELERATION_CUSTOMER_ID`.
    pub customer_id: Option<String>,
    /// The public half of a Stream credential. Falls back to `STREAM_API_KEY`, unless a
    /// customer id was named: naming one is choosing how the router is reached, and a key
    /// lying about in the environment does not overrule the choice.
    pub api_key: Option<String>,
    /// The secret belonging to that key, which is what makes this a backend. Falls back to
    /// `STREAM_API_SECRET`, unless a token was handed in.
    pub api_secret: Option<String>,
    /// A token minted for `user_id` to hold, in place of the secret.
    pub token: Option<String>,
    /// The end user this client acts for. With a secret it is sent as `X-Stream-User-Id`, so
    /// the sessions opened belong to that user and their own device can reach them.
    pub user_id: Option<String>,
    /// Whether the router is reached through Stream's authenticating proxy, which wants the
    /// credential spelled its own way. Falls back to `STREAM_ACCELERATION_AUTHENTICATE`.
    pub authenticate: Option<bool>,
    /// The HTTP client to send with, for a caller that wants its own proxy or timeouts.
    pub http: Option<reqwest::Client>,
}

#[derive(Debug, Clone)]
pub(crate) struct Backend {
    pub url: String,
    pub customer_id: String,
    pub api_key: String,
    pub api_secret: String,
    pub token: String,
    pub user_id: String,
    pub authenticate: bool,
}

impl Backend {
    pub fn new(options: &ClientOptions) -> Result<Self> {
        let url = options
            .url
            .clone()
            .or_else(|| env(URL_ENV))
            .unwrap_or_else(|| DEFAULT_URL.to_string())
            .trim_end_matches('/')
            .to_string();
        let customer_id = options
            .customer_id
            .clone()
            .or_else(|| env(CUSTOMER_ENV))
            .unwrap_or_default();
        let api_key = options.api_key.clone().unwrap_or_else(|| {
            if options.customer_id.is_some() {
                String::new()
            } else {
                env(API_KEY_ENV).unwrap_or_default()
            }
        });
        let token = options.token.clone().unwrap_or_default();
        // A token that was handed in is the caller's answer to who they are, so an ambient
        // secret does not overrule it: otherwise a client built the way a device builds one
        // turns into a backend in any process that happens to hold the secret.
        let api_secret = options.api_secret.clone().unwrap_or_else(|| {
            if token.is_empty() {
                env(API_SECRET_ENV).unwrap_or_default()
            } else {
                String::new()
            }
        });
        let authenticate = options
            .authenticate
            .unwrap_or_else(|| flag(env(AUTHENTICATE_ENV)));

        if authenticate && api_key.is_empty() {
            return Err(Error::configuration(format!(
                "a router behind the proxy is reached with a credential; pass api_key or set {API_KEY_ENV}"
            )));
        }
        if api_key.is_empty() && customer_id.is_empty() {
            return Err(Error::configuration(format!(
                "who is calling is not set; pass customer_id or set {CUSTOMER_ENV} for a router that \
                 trusts one, or api_key with either api_secret or token"
            )));
        }
        if !api_key.is_empty() && api_secret.is_empty() && token.is_empty() {
            return Err(Error::configuration(
                "api_key needs the secret it belongs to, or a token minted with it",
            ));
        }

        Ok(Backend {
            url,
            customer_id,
            api_key,
            api_secret,
            token,
            user_id: options.user_id.clone().unwrap_or_default(),
            authenticate,
        })
    }

    /// Whether this speaks for a process the customer runs rather than for a device.
    pub fn server_side(&self) -> bool {
        !self.api_secret.is_empty() || (self.api_key.is_empty() && !self.customer_id.is_empty())
    }

    /// What every request and socket handshake carries.
    ///
    /// Minted per call, so a client left idle longer than a token lasts does not wake up
    /// holding an expired one. A server socket sends headers rather than the query string a
    /// browser has to use.
    pub fn headers(&self) -> Result<Vec<(&'static str, String)>> {
        if self.api_key.is_empty() {
            return Ok(vec![("X-Customer-Id", self.customer_id.clone())]);
        }

        if self.authenticate {
            // `jwt` whoever the token is for: the proxy works out the caller from the token it
            // verified, and refuses a request that claims `server` for itself.
            return Ok(vec![
                ("api_key", self.api_key.clone()),
                ("stream-auth-type", "jwt".into()),
                ("Authorization", format!("Bearer {}", self.proxy_token()?)),
            ]);
        }

        let mut headers = vec![("X-Api-Key", self.api_key.clone())];
        if !self.api_secret.is_empty() {
            headers.push((
                "Authorization",
                format!(
                    "Bearer {}",
                    sign(json!({"server": true}), &self.api_secret)?
                ),
            ));
            headers.push(("Stream-Auth-Type", "server".into()));
            if !self.user_id.is_empty() {
                headers.push(("X-Stream-User-Id", self.user_id.clone()));
            }
            return Ok(headers);
        }
        headers.push(("Authorization", format!("Bearer {}", self.token)));
        headers.push(("Stream-Auth-Type", "jwt".into()));
        Ok(headers)
    }

    /// The WebSocket URL for a path on the router.
    pub fn socket_url(&self, path: &str) -> String {
        if let Some(rest) = self.url.strip_prefix("https://") {
            return format!("wss://{rest}{path}");
        }
        if let Some(rest) = self.url.strip_prefix("http://") {
            return format!("ws://{rest}{path}");
        }
        format!("{}{path}", self.url)
    }

    /// The token the proxy is given, which names a user where there is one: the proxy has
    /// no header to read a backend's choice of user from.
    fn proxy_token(&self) -> Result<String> {
        if !self.token.is_empty() {
            return Ok(self.token.clone());
        }
        if !self.user_id.is_empty() {
            return sign(json!({"user_id": self.user_id}), &self.api_secret);
        }
        sign(json!({"server": true}), &self.api_secret)
    }
}

/// Signs a Stream token, HS256, valid for [`TOKEN_VALIDITY_SECONDS`].
///
/// `{"server": true}` speaks for the app itself; `{"user_id": "..."}` is a token for that
/// user to hold.
pub fn sign(claims: Value, secret: &str) -> Result<String> {
    sign_for(claims, secret, TOKEN_VALIDITY_SECONDS)
}

/// [`sign`], valid for `validity_seconds`.
pub fn sign_for(claims: Value, secret: &str, validity_seconds: u64) -> Result<String> {
    if secret.is_empty() {
        return Err(Error::configuration(
            "a token cannot be signed without a secret",
        ));
    }
    let issued = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap_or_default()
        .as_secs();
    let mut payload = json!({"iat": issued, "exp": issued + validity_seconds});
    if let (Some(payload), Value::Object(claims)) = (payload.as_object_mut(), claims) {
        payload.extend(claims);
    }

    let signing = format!(
        "{}.{}",
        URL_SAFE_NO_PAD.encode(br#"{"alg":"HS256","typ":"JWT"}"#),
        URL_SAFE_NO_PAD.encode(payload.to_string())
    );
    let mut mac = Hmac::<Sha256>::new_from_slice(secret.as_bytes())
        .map_err(|_| Error::configuration("the secret cannot key an HMAC"))?;
    mac.update(signing.as_bytes());
    Ok(format!(
        "{signing}.{}",
        URL_SAFE_NO_PAD.encode(mac.finalize().into_bytes())
    ))
}

pub(crate) fn env(name: &str) -> Option<String> {
    std::env::var(name).ok().filter(|value| !value.is_empty())
}

fn flag(value: Option<String>) -> bool {
    value.is_some_and(|value| matches!(value.to_lowercase().as_str(), "1" | "true" | "yes" | "on"))
}
