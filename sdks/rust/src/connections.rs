use crate::client::Client;
use crate::error::{Error, Result};
use crate::operations::{DeleteConnectionQuery, ListConnectionsQuery};
use crate::types;

/// The page size [`Connections::list`] reads in, the most the router answers.
const PAGE: i64 = 200;

/// The accounts the app and its users hold at connectors, such as a user's Linear: what an
/// agent's tools reach that account through. Server side only.
///
/// Every method takes the user as its last parameter, named once per call. A user goes to the
/// router as the owner and as `X-Stream-User-Id` alike, so the two never differ. A user's
/// connection is reached only naming that user; the app's is reached by every call, one naming
/// a user included. `None` or an empty id is the app's own connection.
///
/// ```no_run
/// # async fn example(client: vision_agents::Client) -> vision_agents::Result<()> {
/// let connections = client.connections();
/// let connection = connections.create("linear", None, Some("user-42")).await?;
/// let consent = connections
///     .authorize(&connection.id, Some("https://app.example.com/connected"), Some("user-42"))
///     .await?;
/// // Open consent.launch_url in a popup and post it consent.handoff_token.
/// let theirs = connections.list(None, Some("user-42")).await?;
/// connections.delete(&connection.id, Some("user-42")).await?;
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

    /// Stores a pending connection to `connector` (a built-in such as `linear`, or one of the
    /// app's own), owned by `user_id` or else by the app. `label` tells the owner's connections
    /// to the same connector apart.
    pub async fn create(
        &self,
        connector: &str,
        label: Option<&str>,
        user_id: Option<&str>,
    ) -> Result<types::Connection> {
        let user_id = named(user_id);
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
        self.acting(user_id)
            .create_connection(&types::ConnectionRequest {
                auth_scheme: None,
                connector_id: connector.into(),
                inputs: Default::default(),
                label: label.map(Into::into),
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
        return_url: Option<&str>,
        user_id: Option<&str>,
    ) -> Result<types::Authorization> {
        self.reach(id, user_id)?
            .create_authorization(
                id,
                Some(&types::AuthorizationRequest {
                    return_url: return_url.map(Into::into),
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
    /// A user's list holds only that user's connections, not the app's. `connector` keeps one
    /// connector's.
    pub async fn list(
        &self,
        connector: Option<&str>,
        user_id: Option<&str>,
    ) -> Result<Vec<types::Connection>> {
        let user_id = named(user_id);
        let client = self.acting(user_id);
        let mut query = ListConnectionsQuery {
            owner_type: Some(if user_id.is_some() { "user" } else { "app" }.into()),
            connector_id: connector.map(Into::into),
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
        Ok(self.acting(named(user_id)))
    }

    /// This client acting for `user_id`, or for nobody, whoever it acts for otherwise.
    fn acting(&self, user_id: Option<&str>) -> Client {
        self.client.acting_for(user_id.unwrap_or_default())
    }
}

/// The user a call names, with an empty id naming nobody.
fn named(user_id: Option<&str>) -> Option<&str> {
    user_id.filter(|user_id| !user_id.is_empty())
}
