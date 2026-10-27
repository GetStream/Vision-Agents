use std::time::{SystemTime, UNIX_EPOCH};

use base64::Engine as _;
use base64::engine::general_purpose::URL_SAFE_NO_PAD;
use hmac::{Hmac, KeyInit, Mac};
use serde_json::{Value, json};
use sha2::Sha256;

use crate::error::{Error, Result};

pub const URL_ENV: &str = "STREAM_ACCELERATION_URL";
/// Holds this SDK to the hosted router in one region, such as `us-east4`.
pub const REGION_ENV: &str = "STREAM_ACCELERATION_REGION";
pub const CUSTOMER_ENV: &str = "STREAM_ACCELERATION_CUSTOMER_ID";
pub const API_KEY_ENV: &str = "STREAM_API_KEY";
pub const API_SECRET_ENV: &str = "STREAM_API_SECRET";
pub const AUTHENTICATE_ENV: &str = "STREAM_ACCELERATION_AUTHENTICATE";

/// Stream's hosted router, which is where a client goes when nothing says otherwise.
pub const DEFAULT_URL: &str = "https://accelerate.gcp.stream-io-api.com";

/// How long a token minted here lasts. Short, because one is minted per request.
pub const TOKEN_VALIDITY_SECONDS: u64 = 60 * 60;

/// The hosted router held to one region: requests to it are served there and nowhere else,
/// even when that region is full or down.
///
/// Only the shape of the name is checked, such as `us-east4` or `europe-west4`: which
/// regions are served is the hosted router's to say, so offering another needs no release
/// of this SDK.
pub fn region_url(name: &str) -> Result<String> {
    if !is_region(name) {
        return Err(Error::configuration(format!(
            "{name:?} is not a region, such as us-east4"
        )));
    }
    Ok(format!("https://accelerate-{name}.gcp.stream-io-api.com"))
}

/// Whether `name` is one or more lowercase letters, a hyphen, one or more lowercase letters
/// and one or more digits, and nothing else.
fn is_region(name: &str) -> bool {
    let lowercase = |part: &str| !part.is_empty() && part.bytes().all(|b| b.is_ascii_lowercase());
    let Some((area, rest)) = name.split_once('-') else {
        return false;
    };
    let place = rest.trim_end_matches(|c: char| c.is_ascii_digit());
    lowercase(area) && lowercase(place) && place.len() < rest.len()
}

/// Whether `url` is the hosted router, in any region or none, which is always behind the
/// proxy.
fn hosted(url: &str) -> bool {
    url == DEFAULT_URL
        || url
            .strip_prefix("https://accelerate-")
            .and_then(|rest| rest.strip_suffix(".gcp.stream-io-api.com"))
            .is_some_and(is_region)
}

/// Where the router is and who is calling it.
///
/// Three ways to say who that is, and which one a deployment takes is a property of the
/// deployment rather than a choice: a customer id for a router with nothing in front of it,
/// a key and secret for a process the customer runs, and a key and a user token for anything
/// acting on one person's behalf. Every field left `None` falls back to the environment the
/// Go, Python and JavaScript SDKs read.
#[derive(Debug, Clone, Default)]
pub struct ClientOptions {
    /// The router's base URL. Falls back to `region`, then `STREAM_ACCELERATION_URL`, then
    /// `STREAM_ACCELERATION_REGION`, then Stream's hosted router, so only a self-hosted or
    /// local router needs it.
    pub url: Option<String>,
    /// Holds this client to Stream's hosted router in one region, such as `us-east4`. It
    /// cannot be set together with `url`.
    pub region: Option<String>,
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
    /// credential spelled its own way. On for the hosted router, in any region; anywhere else
    /// it falls back to `STREAM_ACCELERATION_AUTHENTICATE`.
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
    /// The end user a server credential speaks for. Unlike `user_id` it never becomes a user
    /// token, so what is written is still the server's and the model answers it.
    pub acting_for: String,
    pub authenticate: bool,
}

impl Backend {
    pub fn new(options: &ClientOptions) -> Result<Self> {
        Self::with_env(options, env)
    }

    /// [`Backend::new`], reading the environment through `env`, so a test can say what is in
    /// it without changing the process's own.
    fn with_env(options: &ClientOptions, env: impl Fn(&str) -> Option<String>) -> Result<Self> {
        let url = match (&options.url, &options.region) {
            (Some(_), Some(_)) => {
                return Err(Error::configuration("set a url or a region, not both"));
            }
            (Some(url), None) => url.clone(),
            (None, Some(region)) => region_url(region)?,
            (None, None) => match (env(URL_ENV), env(REGION_ENV)) {
                (Some(url), _) => url,
                (None, Some(region)) => region_url(&region)?,
                (None, None) => DEFAULT_URL.to_string(),
            },
        }
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
        // The hosted router, in any region, is always behind the proxy.
        let authenticate = options
            .authenticate
            .unwrap_or_else(|| flag(env(AUTHENTICATE_ENV)) || hosted(&url));

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
            acting_for: String::new(),
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
        let mut headers = self.credentials()?;
        if !self.acting_for.is_empty() {
            headers.push(("X-Stream-User-Id", self.acting_for.clone()));
        }
        Ok(headers)
    }

    fn credentials(&self) -> Result<Vec<(&'static str, String)>> {
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

#[cfg(test)]
mod tests {
    use super::*;

    /// An environment holding only `pairs`, read in place of the process's own so tests
    /// running side by side do not see each other's.
    fn environment(pairs: &[(&str, &str)]) -> impl Fn(&str) -> Option<String> {
        let pairs: Vec<(String, String)> = pairs
            .iter()
            .map(|(name, value)| (name.to_string(), value.to_string()))
            .collect();
        move |name| {
            pairs
                .iter()
                .find(|(key, value)| key == name && !value.is_empty())
                .map(|(_, value)| value.clone())
        }
    }

    const CREDENTIAL: [(&str, &str); 3] = [
        (CUSTOMER_ENV, "acme"),
        (API_KEY_ENV, "ntv9"),
        (API_SECRET_ENV, "shh"),
    ];

    #[test]
    fn a_region_is_the_hosted_router_there() {
        let backend = Backend::with_env(
            &ClientOptions {
                region: Some("us-east4".into()),
                ..Default::default()
            },
            environment(&CREDENTIAL),
        )
        .unwrap();

        assert_eq!(
            backend.url,
            "https://accelerate-us-east4.gcp.stream-io-api.com"
        );
        assert!(backend.authenticate);
    }

    #[test]
    fn a_region_in_the_environment_gives_way_to_a_url_there() {
        let local = Backend::with_env(
            &ClientOptions::default(),
            environment(&[
                (CUSTOMER_ENV, "acme"),
                (REGION_ENV, "us-east4"),
                (URL_ENV, "http://localhost:8080"),
            ]),
        )
        .unwrap();
        assert_eq!(local.url, "http://localhost:8080");

        let regional = Backend::with_env(
            &ClientOptions::default(),
            environment(&[
                (REGION_ENV, "us-east4"),
                CREDENTIAL[0],
                CREDENTIAL[1],
                CREDENTIAL[2],
            ]),
        )
        .unwrap();
        assert_eq!(
            regional.url,
            "https://accelerate-us-east4.gcp.stream-io-api.com"
        );
        assert!(regional.authenticate);
    }

    #[test]
    fn a_region_passed_in_beats_a_url_in_the_environment() {
        let backend = Backend::with_env(
            &ClientOptions {
                region: Some("europe-west4".into()),
                ..Default::default()
            },
            environment(&[
                (URL_ENV, "http://localhost:8080"),
                CREDENTIAL[0],
                CREDENTIAL[1],
                CREDENTIAL[2],
            ]),
        )
        .unwrap();

        assert_eq!(
            backend.url,
            "https://accelerate-europe-west4.gcp.stream-io-api.com"
        );
    }

    #[test]
    fn a_url_and_a_region_together_are_refused() {
        let refused = Backend::with_env(
            &ClientOptions {
                url: Some("https://acceleration.example.com".into()),
                region: Some("us-east4".into()),
                customer_id: Some("acme".into()),
                ..Default::default()
            },
            environment(&[]),
        );

        assert!(matches!(refused, Err(Error::Configuration(_))));
    }

    #[test]
    fn anything_but_a_region_name_is_refused() {
        for name in [
            "US-EAST4",
            "us-east4.evil.com/x",
            "us_east4",
            "useast4",
            "us-east",
        ] {
            let refused = Backend::with_env(
                &ClientOptions {
                    region: Some(name.into()),
                    customer_id: Some("acme".into()),
                    ..Default::default()
                },
                environment(&[]),
            );
            match refused {
                Err(Error::Configuration(message)) => assert!(message.contains(name), "{message}"),
                other => panic!("{name:?} resolved: {other:?}"),
            }
        }
    }
}
