use std::time::Duration;

use tokio::time::Instant;

use crate::client::Client;
use crate::error::{Error, Result};
use crate::types;

/// How long a page is waited on. The router queues the read and retries one that fails, so
/// a page can take a while to settle; past this it is returned still pending.
const READ_TIMEOUT: Duration = Duration::from_secs(180);
const POLL_INTERVAL: Duration = Duration::from_millis(250);

/// An agent's knowledge base, as somewhere to put more of it.
///
/// The namespace is the agent's own name, which is where the knowledge in a synced
/// directory lands, so what is added here is found by the same lookup mid-answer.
#[derive(Debug, Clone)]
pub struct Knowledge {
    client: Client,
    namespace: String,
}

impl Knowledge {
    pub fn new(client: Client, namespace: impl Into<String>) -> Self {
        Knowledge {
            client,
            namespace: namespace.into(),
        }
    }

    pub fn namespace(&self) -> &str {
        &self.namespace
    }

    /// Keeps the knowledge base filled from a page published elsewhere.
    ///
    /// The router queues the read and cuts the page into passages the way it cuts a
    /// document; this waits for it, so what comes back already says whether it worked. Still
    /// pending if it was not read within three minutes. Empty `title` and `description` are
    /// left out.
    pub async fn add_url(
        &self,
        url: &str,
        title: &str,
        description: &str,
    ) -> Result<types::KnowledgeUrl> {
        if self.namespace.is_empty() {
            return Err(Error::configuration(
                "a knowledge base is named by the agent it belongs to",
            ));
        }
        let request = types::KnowledgeUrlRequest {
            namespace: self.namespace.clone(),
            url: url.into(),
            title: (!title.is_empty()).then(|| title.into()),
            description: (!description.is_empty()).then(|| description.into()),
        };
        let mut page = self.client.add_knowledge_url(&request).await?;

        let deadline = Instant::now() + READ_TIMEOUT;
        while page.state == types::KnowledgeUrlState::Pending && Instant::now() < deadline {
            tokio::time::sleep(POLL_INTERVAL).await;
            page = self.client.get_knowledge_url(&page.id).await?;
        }
        Ok(page)
    }
}
