//! A client held to a region reaches the hosted router there, for itself and for a user.
//!
//! The regional URL is kept as the SDK built it, TLS and all: only the connection is pointed
//! at a router in process, the way the Go SDK's regression test does it.

use std::net::SocketAddr;
use std::sync::{Arc, Mutex};

use axum::Router;
use axum::extract::State;
use axum::http::HeaderMap;
use axum::response::IntoResponse;
use axum::routing::get;
use serde_json::json;
use tokio::net::{TcpListener, TcpStream};
use tokio_rustls::TlsAcceptor;
use tokio_rustls::rustls::ServerConfig;
use tokio_rustls::rustls::crypto::aws_lc_rs;
use tokio_rustls::rustls::pki_types::{PrivateKeyDer, PrivatePkcs8KeyDer};
use tokio_rustls::server::TlsStream;
use vision_agents::{Client, ClientOptions, DEFAULT_URL, Error, REGION_ENV, URL_ENV};

const REGIONAL_HOST: &str = "accelerate-us-east4.gcp.stream-io-api.com";

/// The environment is the process's, and tests run side by side in it.
static ENVIRONMENT: tokio::sync::Mutex<()> = tokio::sync::Mutex::const_new(());

/// Leaves only a key and a secret in the environment, holding it until the guard is dropped.
async fn environment(pairs: &[(&str, &str)]) -> tokio::sync::MutexGuard<'static, ()> {
    let guard = ENVIRONMENT.lock().await;
    // SAFETY: every test here that reads or writes the environment holds the guard.
    unsafe {
        for name in [
            URL_ENV,
            REGION_ENV,
            "STREAM_ACCELERATION_CUSTOMER_ID",
            "STREAM_ACCELERATION_AUTHENTICATE",
        ] {
            std::env::remove_var(name);
        }
        std::env::set_var("STREAM_API_KEY", "test-key");
        std::env::set_var("STREAM_API_SECRET", "test-secret");
        for (name, value) in pairs {
            std::env::set_var(name, value);
        }
    }
    guard
}

/// One request to `GET /v1/settings/app` as it arrived.
#[derive(Debug, Clone)]
struct Seen {
    host: String,
    api_key: String,
    auth_type: String,
    authorization: String,
}

/// The hosted router in us-east4, served over TLS from this process.
struct Regional {
    addr: SocketAddr,
    seen: Arc<Mutex<Vec<Seen>>>,
}

impl Regional {
    async fn start() -> Regional {
        let cert = rcgen::generate_simple_self_signed(vec![REGIONAL_HOST.to_string()]).unwrap();
        let key = PrivateKeyDer::Pkcs8(PrivatePkcs8KeyDer::from(cert.signing_key.serialize_der()));
        let config = ServerConfig::builder_with_provider(Arc::new(aws_lc_rs::default_provider()))
            .with_safe_default_protocol_versions()
            .unwrap()
            .with_no_client_auth()
            .with_single_cert(vec![cert.cert.der().clone()], key)
            .unwrap();

        let seen = Arc::new(Mutex::new(Vec::new()));
        let app = Router::new()
            .route("/v1/settings/app", get(settings))
            .with_state(seen.clone());
        let listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
        let addr = listener.local_addr().unwrap();
        let listener = Tls {
            listener,
            acceptor: TlsAcceptor::from(Arc::new(config)),
        };
        tokio::spawn(async move { axum::serve(listener, app).await.unwrap() });
        Regional { addr, seen }
    }

    /// An HTTP client that sends the regional URL here, trusting this server's certificate.
    fn http(&self) -> reqwest::Client {
        reqwest::Client::builder()
            .resolve(REGIONAL_HOST, self.addr)
            .danger_accept_invalid_certs(true)
            .build()
            .unwrap()
    }

    /// The single request that arrived, failing the test if there was not exactly one.
    fn request(&self) -> Seen {
        let seen = self.seen.lock().unwrap().clone();
        assert_eq!(seen.len(), 1, "everything sent: {seen:#?}");
        seen[0].clone()
    }

    async fn read_settings(&self, options: ClientOptions) {
        let client = Client::new(ClientOptions {
            http: Some(self.http()),
            ..options
        })
        .unwrap();
        assert_eq!(client.url(), DEFAULT_URL);

        let settings = client.get_app_settings().await.unwrap();

        assert_eq!(settings.stream.writes_into.to_string(), "this_app");
        let seen = self.request();
        assert_eq!(seen.host, REGIONAL_HOST);
        assert_eq!(seen.api_key, "test-key");
        assert_eq!(seen.auth_type, "jwt");
        assert!(!seen.authorization.is_empty());
    }
}

async fn settings(
    State(seen): State<Arc<Mutex<Vec<Seen>>>>,
    headers: HeaderMap,
) -> impl IntoResponse {
    let header = |name: &str| {
        headers
            .get(name)
            .and_then(|value| value.to_str().ok())
            .unwrap_or("")
            .to_string()
    };
    seen.lock().unwrap().push(Seen {
        host: header("host"),
        api_key: header("api_key"),
        auth_type: header("stream-auth-type"),
        authorization: header("authorization"),
    });
    axum::Json(json!({"stream": {
        "tenancy": "app",
        "writes_into": "this_app",
        "channel_type": "present",
        "call_type": "present",
    }}))
}

/// Connections accepted over TLS.
struct Tls {
    listener: TcpListener,
    acceptor: TlsAcceptor,
}

impl axum::serve::Listener for Tls {
    type Io = TlsStream<TcpStream>;
    type Addr = SocketAddr;

    async fn accept(&mut self) -> (Self::Io, Self::Addr) {
        loop {
            let Ok((stream, addr)) = self.listener.accept().await else {
                continue;
            };
            if let Ok(stream) = self.acceptor.accept(stream).await {
                return (stream, addr);
            }
        }
    }

    fn local_addr(&self) -> std::io::Result<Self::Addr> {
        self.listener.local_addr()
    }
}

#[tokio::test]
async fn the_default_client_reads_app_settings() {
    let _environment = environment(&[]).await;
    let regional = Regional::start().await;

    regional.read_settings(ClientOptions::default()).await;
}

#[tokio::test]
async fn an_explicit_region_reads_app_settings() {
    let _environment = environment(&[]).await;
    let regional = Regional::start().await;

    regional
        .read_settings(ClientOptions {
            region: Some("us-east4".into()),
            ..Default::default()
        })
        .await;
}

#[tokio::test]
async fn a_region_from_the_environment_reads_app_settings() {
    let _environment = environment(&[(REGION_ENV, "us-east4")]).await;
    let regional = Regional::start().await;

    regional.read_settings(ClientOptions::default()).await;
}

#[tokio::test]
async fn a_conflicting_url_and_region_are_refused() {
    let _environment = environment(&[]).await;

    let refused = Client::new(ClientOptions {
        region: Some("europe-west4".into()),
        url: Some(DEFAULT_URL.into()),
        ..Default::default()
    });

    assert!(
        matches!(refused, Err(Error::Configuration(_))),
        "{refused:?}"
    );
}

#[tokio::test]
async fn an_invalid_region_with_a_url_is_refused() {
    let _environment = environment(&[]).await;

    let refused = Client::new(ClientOptions {
        region: Some("us-east4.evil.com/x".into()),
        url: Some(DEFAULT_URL.into()),
        ..Default::default()
    });

    assert!(
        matches!(refused, Err(Error::Configuration(_))),
        "{refused:?}"
    );
}

#[tokio::test]
async fn a_regional_client_can_act_for_a_user() {
    let _environment = environment(&[]).await;
    let regional = Regional::start().await;
    let client = Client::new(ClientOptions {
        region: Some("us-east4".into()),
        http: Some(regional.http()),
        ..Default::default()
    })
    .unwrap();

    let user = client.as_user("jean", "user-token").unwrap();

    assert_eq!(user.url(), DEFAULT_URL);
    user.get_app_settings().await.unwrap();
    let seen = regional.request();
    assert_eq!(seen.host, REGIONAL_HOST);
    assert_eq!(seen.api_key, "test-key");
    assert_eq!(seen.auth_type, "jwt");
    assert_eq!(seen.authorization, "Bearer user-token");
}
