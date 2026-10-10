use crate::client::Client;
use crate::error::{Error, Result};
use crate::operations::{ListResponseItemsQuery, ListResponsesQuery};
use crate::types;

/// How many items are read per request while reading a whole conversation.
const ITEM_PAGE: i64 = 200;

/// A session's turns.
///
/// `items` here is the whole conversation flattened, which is how a conversation reads: the
/// question, what the agent did about it, what it said, then the next question. One turn's
/// items come off the handle `create` returns.
#[derive(Debug, Clone)]
pub struct Responses {
    client: Client,
    session_id: String,
    pub items: Items,
}

impl Responses {
    pub(crate) fn new(client: Client, session_id: &str) -> Self {
        Responses {
            items: Items::new(client.clone(), session_id, ""),
            client,
            session_id: session_id.into(),
        }
    }

    /// Asks the agent something and names the turn it answers as.
    ///
    /// It returns as soon as the agent has started answering, not when it has finished: a
    /// model takes seconds, and the session's events are what watch it arrive.
    pub async fn create(&self, text: &str) -> Result<AgentResponse> {
        self.create_with(&types::CreateResponseRequest {
            text: text.into(),
            ..Default::default()
        })
        .await
    }

    /// [`Responses::create`], with images or anything else the request carries.
    ///
    /// The request's `request_id` is not the caller's to set: a question that is text only is
    /// sent with a fresh one, and one with images or videos with none.
    pub async fn create_with(
        &self,
        request: &types::CreateResponseRequest,
    ) -> Result<AgentResponse> {
        self.create_answering(request, None).await
    }

    /// `answering` is the request id of the inbound message being answered, sent in place of a
    /// fresh one.
    pub(crate) async fn create_answering(
        &self,
        request: &types::CreateResponseRequest,
        answering: Option<String>,
    ) -> Result<AgentResponse> {
        let text_only = request.images.as_ref().is_none_or(Vec::is_empty)
            && request.videos.as_ref().is_none_or(Vec::is_empty);
        let request = types::CreateResponseRequest {
            request_id: text_only.then(|| answering.unwrap_or_else(new_request_id)),
            ..request.clone()
        };
        let created = self
            .client
            .create_response(&self.session_id, &request)
            .await?;
        Ok(AgentResponse {
            items: Items::new(self.client.clone(), &created.session_id, &created.id),
            created,
        })
    }

    /// One page of the turns so far, oldest first. `None` for the cursor is the first page,
    /// and the page's `next_cursor` the next.
    pub async fn list(
        &self,
        limit: Option<i64>,
        cursor: Option<String>,
    ) -> Result<types::AgentResponsePage> {
        self.client
            .list_responses(&self.session_id, &ListResponsesQuery { limit, cursor })
            .await
    }

    /// Goes back to a response and carries on from there, as though nothing after it was said.
    ///
    /// The model forgets the later turns and they drop out of `list` and `items`. A
    /// conversation kept in Stream Chat is refused, because the channel would still hold the
    /// later turns: fork it at the response instead.
    pub async fn rewind(&self, to: impl ResponseRef) -> Result<()> {
        let response_id = to.response_id();
        if response_id.is_empty() {
            return Err(Error::configuration(
                "a response that was never recorded cannot be rewound to",
            ));
        }
        self.client
            .rewind_session(
                &self.session_id,
                &types::RewindSessionRequest {
                    response_id: response_id.into(),
                },
            )
            .await
    }
}

/// One turn, and a way to read what it was made of.
#[derive(Debug, Clone)]
pub struct AgentResponse {
    pub created: types::AgentResponse,
    pub items: Items,
}

impl AgentResponse {
    /// The backend's id for this turn. Not the `turn_id` socket events carry.
    pub fn id(&self) -> &str {
        &self.created.id
    }
}

/// The things a conversation, or one turn of it, was made of, in the order they happened.
///
/// Read rather than watched: this is what the backend wrote down. Deltas are not here; a
/// hundred fragments of one sentence are the sentence.
#[derive(Debug, Clone)]
pub struct Items {
    client: Client,
    session_id: String,
    response_id: String,
}

impl Items {
    fn new(client: Client, session_id: &str, response_id: &str) -> Self {
        Items {
            client,
            session_id: session_id.into(),
            response_id: response_id.into(),
        }
    }

    /// One page, for a caller doing its own paging. `None` for the cursor is the first page,
    /// and the page's `next_cursor` the next.
    pub async fn list(
        &self,
        limit: Option<i64>,
        cursor: Option<String>,
    ) -> Result<types::AgentResponseItemPage> {
        let query = ListResponseItemsQuery {
            response_id: (!self.response_id.is_empty()).then(|| self.response_id.clone()),
            limit,
            cursor,
        };
        self.client
            .list_response_items(&self.session_id, &query)
            .await
    }

    /// Every item, oldest first, a page at a time.
    pub async fn all(&self) -> Result<Vec<types::AgentResponseItem>> {
        let mut collected = Vec::new();
        let mut cursor = None;
        loop {
            let page = self.list(Some(ITEM_PAGE), cursor).await?;
            collected.extend(page.items);
            match page.next_cursor {
                Some(next) if page.has_more => cursor = Some(next),
                _ => return Ok(collected),
            }
        }
    }
}

fn new_request_id() -> String {
    let bytes: [u8; 16] = rand::random();
    bytes.iter().map(|byte| format!("{byte:02x}")).collect()
}

/// Anything that names a response: its id, the response, or any item of one.
pub trait ResponseRef {
    fn response_id(&self) -> &str;
}

impl ResponseRef for &str {
    fn response_id(&self) -> &str {
        self
    }
}

impl ResponseRef for &String {
    fn response_id(&self) -> &str {
        self
    }
}

impl ResponseRef for &AgentResponse {
    fn response_id(&self) -> &str {
        &self.created.id
    }
}

impl ResponseRef for &types::AgentResponse {
    fn response_id(&self) -> &str {
        &self.id
    }
}

impl ResponseRef for &types::AgentResponseItem {
    fn response_id(&self) -> &str {
        &self.response_id
    }
}
