use crate::client::Client;
use crate::error::{Error, Result};
use crate::operations::{DeleteConnectionQuery, ListConnectionsQuery};
use crate::types;

/// The page size [`Connections::list`] reads in, the most the router answers.
const PAGE: i64 = 200;

/// What [`Connections::create`] stores.
#[derive(Debug, Clone, Default, PartialEq)]
pub struct NewConnection {
    /// A built-in connector such as `linear`, or one of the app's own.
    pub connector: String,
    /// Tells the owner's connections to the same connector apart.
    pub label: Option<String>,
    /// The user the connection is for. `None` or empty is the app's own.
    pub user_id: Option<String>,
}

/// What [`Connections::authorize`] starts.
#[derive(Debug, Clone, Default, PartialEq)]
pub struct AuthorizeOptions {
    /// Where the browser goes back to when the consent ends. `None` goes back to the dashboard.
    pub return_url: Option<String>,
    /// The user the connection belongs to. `None` or empty is the app's own.
    pub user_id: Option<String>,
}

/// Which connections [`Connections::list`] reads. The default is the app's own, of every connector.
#[derive(Debug, Clone, Default, PartialEq)]
pub struct ConnectionFilter {
    /// Keeps one connector's connections.
    pub connector: Option<String>,
    /// Reads that user's connections, not the app's. `None` or empty reads the app's own.
    pub user_id: Option<String>,
}

/// The accounts the app and its users hold at connectors, such as a user's Linear: what an
/// agent's tools reach that account through. Server side only: a client holding a user's token
/// is refused naming a user, before anything is sent.
///
/// Every call names its user once, as `user_id`. A user goes to the router as the owner and as
/// `X-Stream-User-Id` alike, so the two never differ. A user's connection is reached only
/// naming that user; the app's is reached by every call, one naming a user included. `None` or
/// an empty id is the app's own connection, and then the client keeps acting for whoever it was
/// built for with `ClientOptions::user_id`, if anyone.
///
/// ```no_run
/// # async fn example(client: vision_agents::Client) -> vision_agents::Result<()> {
/// use vision_agents::{AuthorizeOptions, ConnectionFilter, NewConnection};
///
/// let connections = client.connections();
/// let user_id = Some("user-42".to_string());
/// let connection = connections
///     .create(&NewConnection { connector: "linear".into(), user_id: user_id.clone(), ..Default::default() })
///     .await?;
/// let consent = connections
///     .authorize(&connection.id, &AuthorizeOptions {
///         return_url: Some("https://app.example.com/connected".into()),
///         user_id: user_id.clone(),
///     })
///     .await?;
/// // Open consent.launch_url in a popup and post it consent.handoff_token.
/// let theirs = connections
///     .list(&ConnectionFilter { user_id: user_id.clone(), ..Default::default() })
///     .await?;
/// connections.delete(&connection.id, user_id.as_deref()).await?;
/// # Ok(())
/// # }
/// ```
#[derive(Debug, Clone)]
pub struct Connections {
    client: Client,
}

impl Connections {
    pub(crate) fn new(client: Client) -> Self {
        Connections { client }
    }

    /// Stores a pending connection to `new.connector`, owned by `new.user_id` or else by the app.
    pub async fn create(&self, new: &NewConnection) -> Result<types::Connection> {
        let user_id = named(new.user_id.as_deref());
        let owner = match user_id {
            Some(user_id) => types::ConnectionOwner {
                type_: types::ConnectionOwnerType::User,
                user_id: Some(user_id.into()),
            },
            None => types::ConnectionOwner {
                type_: types::ConnectionOwnerType::App,
                user_id: None,
            },
        };
        self.acting(user_id)?
            .create_connection(&types::ConnectionRequest {
                auth_scheme: None,
                connector_id: new.connector.clone(),
                inputs: Default::default(),
                label: new.label.clone(),
                owner,
            })
            .await
    }

    /// Starts the provider's consent for a connection: a first one for a pending connection, a
    /// reconnect for one connected before. Open the `launch_url` it answers in a popup from a
    /// page of `return_url`'s origin and post it the `handoff_token`; the attempt ends at
    /// `expires_at`. The browser goes back to `return_url` when the consent ends, with
    /// `connection_id` and `status` added to its query; `None` goes back to the dashboard.
    pub async fn authorize(
        &self,
        id: &str,
        options: &AuthorizeOptions,
    ) -> Result<types::Authorization> {
        self.reach(id, options.user_id.as_deref())?
            .create_authorization(
                id,
                Some(&types::AuthorizationRequest {
                    return_url: options.return_url.clone(),
                }),
            )
            .await
    }

    /// One connection of `user_id`'s, or of the app's. Another user's is not found, the same as
    /// one that does not exist; the app's is reached with a user named too.
    pub async fn get(&self, id: &str, user_id: Option<&str>) -> Result<types::Connection> {
        self.reach(id, user_id)?.get_connection(id).await
    }

    /// Every connection of `user_id`'s, or of the app's, newest first, read page by page.
    /// A user's list holds only that user's connections, not the app's.
    pub async fn list(&self, filter: &ConnectionFilter) -> Result<Vec<types::Connection>> {
        let user_id = named(filter.user_id.as_deref());
        let client = self.acting(user_id)?;
        let mut query = ListConnectionsQuery {
            owner_type: Some(if user_id.is_some() { "user" } else { "app" }.into()),
            connector_id: filter.connector.clone(),
            limit: Some(PAGE),
            cursor: None,
        };
        let mut connections = Vec::new();
        loop {
            let page = client.list_connections(&query).await?;
            connections.extend(page.items.unwrap_or_default());
            match page.next_cursor {
                Some(cursor) if page.has_more && !cursor.is_empty() => query.cursor = Some(cursor),
                _ => return Ok(connections),
            }
        }
    }

    /// Disconnects a connection of `user_id`'s, or of the app's, and drops its credentials. One
    /// an agent config binds as its fixed connection is refused with a 409.
    pub async fn delete(&self, id: &str, user_id: Option<&str>) -> Result<()> {
        self.reach(id, user_id)?
            .delete_connection(id, &DeleteConnectionQuery::default())
            .await
    }

    /// The client for one connection's requests. An empty id is refused before anything is
    /// sent: it would ask for `/v1/agents/connections/`, which names no connection.
    fn reach(&self, id: &str, user_id: Option<&str>) -> Result<Client> {
        if id.is_empty() {
            return Err(Error::configuration("a connection needs its id"));
        }
        self.acting(named(user_id))
    }

    /// This client acting for `user_id`; with none, as it was built, acting for whoever
    /// `ClientOptions::user_id` names, if anyone. A device holds its own user's token, which the
    /// router refuses here, so naming a user from one is refused before anything is sent.
    fn acting(&self, user_id: Option<&str>) -> Result<Client> {
        match user_id {
            Some(_) if !self.client.server_side() => Err(Error::configuration(
                "a user's connections keep the app's credential, which only a backend holds; a \
                 device holds its user's token (as_user) and acts only as that user, so leave \
                 user_id off",
            )),
            Some(user_id) => Ok(self.client.acting_for(user_id)),
            None => Ok(self.client.clone()),
        }
    }
}

/// The user a call names, with an empty id naming nobody.
fn named(user_id: Option<&str>) -> Option<&str> {
    user_id.filter(|user_id| !user_id.is_empty())
}
