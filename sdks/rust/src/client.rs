use std::sync::Arc;
use std::time::Duration;

use percent_encoding::{AsciiSet, NON_ALPHANUMERIC, utf8_percent_encode};
use reqwest::Method;
use serde::Serialize;
use serde::de::DeserializeOwned;

use crate::backend::{Backend, ClientOptions};
use crate::error::{Error, Result};
use crate::sessions::AgentRef;
use crate::socket::Socket;
use crate::types;

/// The router, held once: where it is and who is calling it.
///
/// Every operation in the spec that answers over plain HTTP is a method here, generated from
/// `acceleration/api/openapi.yaml`, so a new endpoint needs nothing written by hand to be
/// callable. Cheap to clone; clones share one connection pool.
#[derive(Debug, Clone)]
pub struct Client {
    inner: Arc<Inner>,
}

#[derive(Debug)]
struct Inner {
    backend: Backend,
    http: reqwest::Client,
}

impl Client {
    pub fn new(options: ClientOptions) -> Result<Self> {
        let backend = Backend::new(&options)?;
        let http = match options.http {
            Some(http) => http,
            None => reqwest::Client::builder()
                .connect_timeout(Duration::from_secs(10))
                .timeout(Duration::from_secs(120))
                .build()
                .map_err(|source| Error::Transport {
                    operation: "building the HTTP client".into(),
                    source,
                })?,
        };
        Ok(Client {
            inner: Arc::new(Inner { backend, http }),
        })
    }

    /// A client configured entirely from the environment.
    pub fn from_env() -> Result<Self> {
        Client::new(ClientOptions::default())
    }

    /// The router's base URL.
    pub fn url(&self) -> &str {
        &self.inner.backend.url
    }

    /// Whether this speaks for a process the customer runs rather than for a device.
    ///
    /// Only a server-side caller reaches the operations the spec does not mark client
    /// accessible. Worth asking before a call rather than reading a 403 afterwards.
    pub fn server_side(&self) -> bool {
        self.inner.backend.server_side()
    }

    /// A client acting for one end user, holding the token that proves it.
    ///
    /// A new client rather than a change to this one, because a process usually holds both:
    /// its own credential for what only a backend may do, and one per user for the
    /// conversations that belong to them.
    pub fn as_user(&self, user_id: &str, token: &str) -> Result<Client> {
        if user_id.is_empty() {
            return Err(Error::configuration("a user needs an id"));
        }
        if token.is_empty() {
            return Err(Error::configuration(format!(
                "there is no token for {user_id} to hold"
            )));
        }
        let backend = &self.inner.backend;
        Client::new(ClientOptions {
            url: Some(backend.url.clone()),
            customer_id: Some(backend.customer_id.clone()),
            api_key: Some(backend.api_key.clone()),
            // A token is the whole credential, so holding one stops this being a backend.
            api_secret: Some(String::new()),
            token: Some(token.to_string()),
            user_id: Some(user_id.to_string()),
            authenticate: Some(backend.authenticate),
            http: Some(self.inner.http.clone()),
        })
    }

    /// A client acting for a guest: a guest is a user with a token.
    pub fn as_guest(&self, guest: &types::GuestUser) -> Result<Client> {
        self.as_user(&guest.id, &guest.token)
    }

    /// An agent addressed by the name it is configured under, and its conversations.
    ///
    /// No request is made: a name that matches nothing is refused when a conversation is
    /// opened rather than here.
    pub fn agent(&self, name: &str) -> AgentRef {
        AgentRef::new(self.clone(), name)
    }

    /// Mints a guest so somebody can talk to an agent before they sign up.
    ///
    /// Nothing is remembered here: a server holding guests would hand one visitor another's
    /// conversations. Knowing which visitor is which is the caller's job.
    pub async fn guest_user(&self, request: &types::GuestUserRequest) -> Result<types::GuestUser> {
        self.create_guest_user(Some(request)).await
    }

    /// Moves a guest's conversations onto the account they turned out to be.
    ///
    /// Server side only: only the backend that just authenticated the account knows which
    /// guest it was. Refused here, with the reason, rather than left to a 403.
    pub async fn claim_guest(
        &self,
        guest_id: &str,
        user_id: &str,
    ) -> Result<types::ClaimGuestResult> {
        if !self.server_side() {
            return Err(Error::configuration(
                "claiming a guest is server side only: it is the backend that just authenticated \
                 the account that knows which guest it was",
            ));
        }
        if guest_id.is_empty() || user_id.is_empty() {
            return Err(Error::configuration(
                "claiming a guest needs the guest and the account",
            ));
        }
        self.claim_guest_user(&types::ClaimGuestRequest {
            guest_id: guest_id.into(),
            user_id: user_id.into(),
        })
        .await
    }

    /// Sends one request and decodes the answer.
    ///
    /// The generated methods are this with the path, types and operation id filled in; it is
    /// public for an operation newer than this build. An empty answer decodes as JSON `null`,
    /// which is `()` for a 204 and `None` for an `Option`.
    pub async fn send<Q, B, T>(
        &self,
        method: Method,
        path: &str,
        query: Option<&Q>,
        body: Option<&B>,
        operation: &str,
    ) -> Result<T>
    where
        Q: Serialize + ?Sized,
        B: Serialize + ?Sized,
        T: DeserializeOwned,
    {
        let mut request = self
            .inner
            .http
            .request(method, format!("{}{path}", self.url()));
        for (name, value) in self.inner.backend.headers()? {
            request = request.header(name, value);
        }
        if let Some(query) = query {
            request = request.query(query);
        }
        if let Some(body) = body {
            request = request.json(body);
        }

        let transport = |source| Error::Transport {
            operation: operation.to_string(),
            source,
        };
        let response = request.send().await.map_err(transport)?;
        let status = response.status();
        let bytes = response.bytes().await.map_err(transport)?;

        if !status.is_success() {
            let said = serde_json::from_slice::<types::Error>(&bytes)
                .map(|error| error.error)
                .unwrap_or_else(|_| String::from_utf8_lossy(&bytes).trim().to_string());
            return Err(Error::Router {
                status: status.as_u16(),
                operation: operation.to_string(),
                message: if said.is_empty() {
                    status.to_string()
                } else {
                    said
                },
            });
        }

        let answer: &[u8] = if bytes.is_empty() { b"null" } else { &bytes };
        serde_json::from_slice(answer).map_err(|source| Error::Decode {
            operation: operation.to_string(),
            source,
        })
    }

    /// Opens a socket on the router, carrying the same credentials a request does.
    pub async fn socket(&self, path: &str) -> Result<Socket> {
        let backend = &self.inner.backend;
        Socket::connect(&backend.socket_url(path), backend.headers()?).await
    }
}

/// Everything but the unreserved characters, so an id is always one path segment.
const SEGMENT: &AsciiSet = &NON_ALPHANUMERIC
    .remove(b'-')
    .remove(b'_')
    .remove(b'.')
    .remove(b'~');

/// Percent-encodes one path segment.
pub fn segment(value: &str) -> String {
    utf8_percent_encode(value, SEGMENT).to_string()
}
