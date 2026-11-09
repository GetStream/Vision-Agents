use crate::backend::{check_category, check_team};
use crate::client::Client;
use crate::error::{Error, Result};
use crate::operations::{GetLimitUsageQuery, ListLimitsQuery};
use crate::types::{self, LimitInterval, LimitLevel, LimitSubject};
use crate::usd::Usd;

/// The page size lists are read in, the most the router answers.
const PAGE: i64 = 200;

/// The customer's spend limits: on the organization, its apps, keys, end users, teams and
/// usage categories. Server side only: the router refuses a user's token, 403
/// `server_side_only`.
///
/// Every limit here is the customer's own. Stream's spend limit for the organization is never
/// among them, and nothing here lifts it. Work a limit refuses fails with `Error::Router` of
/// status 402 and code `spend_limit_reached`, or 429 `user_limit_reached` for one end user's
/// limit. Stream's own limit refuses with 402 `system_limit_reached`.
#[derive(Debug, Clone)]
pub struct SpendLimits {
    client: Client,
}

/// A limit to create: what one subject may spend in each window of `interval`.
///
/// The level is the most specific one the ids name, which the router works out: `User` with
/// a user id is that user's own limit, `User` with only an app id the default for every user
/// of the app, and `User` alone the default for every user of the organization. Keys go the
/// same way. `level` overrides it. `Team` with an app id is the team within that app.
#[derive(Debug, Clone, PartialEq)]
pub struct Limit {
    pub subject: LimitSubject,
    pub interval: LimitInterval,
    pub amount: Usd,
    pub app_id: Option<String>,
    pub api_key: Option<String>,
    pub user_id: Option<String>,
    pub team: Option<String>,
    pub category: Option<String>,
    pub level: Option<LimitLevel>,
    pub note: Option<String>,
}

impl Limit {
    pub fn new(subject: LimitSubject, interval: LimitInterval, amount: Usd) -> Self {
        Limit {
            subject,
            interval,
            amount,
            app_id: None,
            api_key: None,
            user_id: None,
            team: None,
            category: None,
            level: None,
            note: None,
        }
    }

    pub fn level(mut self, level: LimitLevel) -> Self {
        self.level = Some(level);
        self
    }

    pub fn note(mut self, note: impl Into<String>) -> Self {
        self.note = Some(note.into());
        self
    }
}

/// Which limits [`SpendLimits::list`] reads. The default is every limit.
#[derive(Debug, Clone, Default, PartialEq)]
pub struct LimitFilter {
    /// `Team` lists the teams' limits, `Category` every other category's. With an app id they
    /// are the ones within that app.
    pub subject: Option<LimitSubject>,
    pub level: Option<LimitLevel>,
    pub interval: Option<LimitInterval>,
    pub app_id: Option<String>,
    pub api_key: Option<String>,
    pub user_id: Option<String>,
    pub team: Option<String>,
    pub category: Option<String>,
}

/// One subject, for what it spent.
#[derive(Debug, Clone, PartialEq)]
pub struct Target {
    pub subject: LimitSubject,
    pub app_id: Option<String>,
    pub api_key: Option<String>,
    pub user_id: Option<String>,
    pub team: Option<String>,
    pub category: Option<String>,
}

impl Target {
    pub fn new(subject: LimitSubject) -> Self {
        Target {
            subject,
            app_id: None,
            api_key: None,
            user_id: None,
            team: None,
            category: None,
        }
    }
}

/// Counts a subject's limits from now until each window ends: every interval's, or only those
/// in `intervals`. Nothing is refunded: what it spent stays spent.
#[derive(Debug, Clone, PartialEq)]
pub struct Reset {
    pub subject: LimitSubject,
    /// Why, as the audit log records it. Required.
    pub reason: String,
    /// Empty is every interval.
    pub intervals: Vec<LimitInterval>,
    pub app_id: Option<String>,
    pub api_key: Option<String>,
    pub user_id: Option<String>,
    pub team: Option<String>,
    pub category: Option<String>,
}

impl Reset {
    pub fn new(subject: LimitSubject, reason: impl Into<String>) -> Self {
        Reset {
            subject,
            reason: reason.into(),
            intervals: Vec::new(),
            app_id: None,
            api_key: None,
            user_id: None,
            team: None,
            category: None,
        }
    }

    pub fn intervals(mut self, intervals: impl IntoIterator<Item = LimitInterval>) -> Self {
        self.intervals = intervals.into_iter().collect();
        self
    }
}

/// The setters naming a subject's ids, which every limit body and filter has.
macro_rules! subject_ids {
    ($($kind:ty),*) => {$(
        impl $kind {
            pub fn app_id(mut self, app_id: impl Into<String>) -> Self {
                self.app_id = Some(app_id.into());
                self
            }

            pub fn api_key(mut self, api_key: impl Into<String>) -> Self {
                self.api_key = Some(api_key.into());
                self
            }

            pub fn user_id(mut self, user_id: impl Into<String>) -> Self {
                self.user_id = Some(user_id.into());
                self
            }

            pub fn team(mut self, team: impl Into<String>) -> Self {
                self.team = Some(team.into());
                self
            }

            pub fn category(mut self, category: impl Into<String>) -> Self {
                self.category = Some(category.into());
                self
            }

            fn ids(&self) -> Ids<'_> {
                Ids {
                    app_id: given(&self.app_id),
                    api_key: given(&self.api_key),
                    user_id: given(&self.user_id),
                    team: given(&self.team),
                    category: given(&self.category),
                }
            }
        }
    )*};
}

subject_ids!(Limit, LimitFilter, Target, Reset);

impl SpendLimits {
    pub(crate) fn new(client: Client) -> Self {
        SpendLimits { client }
    }

    pub(crate) fn client(&self) -> &Client {
        &self.client
    }

    /// Every limit the filter matches, read page by page.
    pub async fn list(&self, filter: &LimitFilter) -> Result<Vec<types::Limit>> {
        let ids = filter.ids();
        let subject = match &filter.subject {
            Some(subject) => Some(ids.target(subject)?),
            None => {
                ids.team.map(check_team).transpose()?;
                ids.category.map(check_category).transpose()?;
                None
            }
        };
        let mut query = ListLimitsQuery {
            subject: subject.as_ref().map(ToString::to_string),
            level: filter.level.as_ref().map(ToString::to_string),
            app_id: ids.app_id.map(Into::into),
            api_key_id: ids.api_key.map(Into::into),
            user_id: ids.user_id.map(Into::into),
            team: ids.team.map(Into::into),
            category: ids.category.map(Into::into),
            interval: filter.interval.as_ref().map(ToString::to_string),
            limit: Some(PAGE),
            cursor: None,
        };
        let mut limits = Vec::new();
        loop {
            let page = self.client.list_limits(&query).await?;
            limits.extend(page.items.unwrap_or_default());
            match page.next_cursor {
                Some(cursor) if page.has_more && !cursor.is_empty() => query.cursor = Some(cursor),
                _ => return Ok(limits),
            }
        }
    }

    /// One limit by its id.
    pub async fn get(&self, limit_id: i64) -> Result<types::Limit> {
        self.client.get_limit(&limit_id.to_string()).await
    }

    /// Creates a limit.
    ///
    /// The router refuses with code `limit_above_parent` (the limit is above one of a subject
    /// that contains it), `limit_exists` (the subject has one at that level and interval:
    /// change it with [`SpendLimits::set_amount`]) or `use_team`.
    pub async fn create(&self, limit: &Limit) -> Result<types::Limit> {
        let ids = limit.ids();
        let subject = ids.target(&limit.subject)?;
        not_negative(limit.amount)?;
        self.client
            .create_limit(&types::LimitInput {
                subject,
                interval: limit.interval.clone(),
                limit_micros: limit.amount.micros(),
                app_id: ids.app_id.map(Into::into),
                api_key: ids.api_key.map(Into::into),
                user_id: ids.user_id.map(Into::into),
                team: ids.team.map(Into::into),
                category: ids.category.map(Into::into),
                level: limit.level.clone(),
                note: limit.note.clone().filter(|note| !note.is_empty()),
            })
            .await
    }

    /// Changes what a limit allows. What it is on never changes.
    pub async fn set_amount(&self, limit_id: i64, amount: Usd) -> Result<types::Limit> {
        not_negative(amount)?;
        self.update(
            limit_id,
            &types::LimitChange {
                limit_micros: Some(amount.micros()),
                note: None,
            },
        )
        .await
    }

    /// Changes a limit's note. An empty one clears it.
    pub async fn set_note(&self, limit_id: i64, note: &str) -> Result<types::Limit> {
        self.update(
            limit_id,
            &types::LimitChange {
                limit_micros: None,
                note: Some(note.into()),
            },
        )
        .await
    }

    pub(crate) async fn update(
        &self,
        limit_id: i64,
        change: &types::LimitChange,
    ) -> Result<types::Limit> {
        self.client
            .update_limit(&limit_id.to_string(), change)
            .await
    }

    /// Deletes a limit. Its subject's spend from then on is held only to its parents'.
    pub async fn delete(&self, limit_id: i64) -> Result<()> {
        self.client.delete_limit(&limit_id.to_string()).await
    }

    /// What one subject spent in its current windows, against its limits and grants.
    pub async fn usage(&self, target: &Target) -> Result<types::SubjectUsage> {
        let ids = target.ids();
        let subject = ids.target(&target.subject)?;
        let mut answered = self
            .client
            .get_limit_usage(&GetLimitUsageQuery {
                subject: Some(subject.to_string()),
                app_id: ids.app_id.map(Into::into),
                api_key_id: ids.api_key.map(Into::into),
                user_id: ids.user_id.map(Into::into),
                team: ids.team.map(Into::into),
                category: ids.category.map(Into::into),
            })
            .await?;
        if answered.len() != 1 {
            return Err(Error::Failed {
                operation: "getLimitUsage".into(),
                message: format!(
                    "reading the spend of one {subject}: the router answered {}",
                    answered.len()
                ),
            });
        }
        Ok(answered.remove(0))
    }

    /// Counts a subject's limits from now. The router refuses with code `no_limit` a subject
    /// with no limit of the customer's.
    pub async fn reset(&self, reset: &Reset) -> Result<types::SubjectUsage> {
        if reset.reason.trim().is_empty() {
            return Err(Error::configuration("a reset needs a reason"));
        }
        let ids = reset.ids();
        let subject = ids.target(&reset.subject)?;
        self.client
            .reset_limit_usage(&types::LimitResetInput {
                subject,
                reason: reset.reason.clone(),
                intervals: (!reset.intervals.is_empty()).then(|| reset.intervals.clone()),
                app_id: ids.app_id.map(Into::into),
                api_key: ids.api_key.map(Into::into),
                user_id: ids.user_id.map(Into::into),
                team: ids.team.map(Into::into),
                category: ids.category.map(Into::into),
            })
            .await
    }
}

/// The ids that name one subject, an empty one taken as not given.
struct Ids<'a> {
    app_id: Option<&'a str>,
    api_key: Option<&'a str>,
    user_id: Option<&'a str>,
    team: Option<&'a str>,
    category: Option<&'a str>,
}

impl Ids<'_> {
    /// The subject as the router names it: a team or category within an app is `AppTeam` or
    /// `AppCategory`. It refuses what the router would, before anything is sent.
    fn target(&self, subject: &LimitSubject) -> Result<LimitSubject> {
        if let Some(category) = self.category {
            check_category(category)?;
        }
        if let Some(team) = self.team {
            check_team(team)?;
        }
        let in_app = self.app_id.is_some();
        let subject = match subject {
            LimitSubject::Team | LimitSubject::AppTeam if self.team.is_none() => {
                return Err(Error::configuration(format!(
                    "a {subject} subject names its team"
                )));
            }
            LimitSubject::Category | LimitSubject::AppCategory if self.category.is_none() => {
                return Err(Error::configuration(format!(
                    "a {subject} subject names its category"
                )));
            }
            LimitSubject::Team if in_app => LimitSubject::AppTeam,
            LimitSubject::Category if in_app => LimitSubject::AppCategory,
            other => other.clone(),
        };
        let team_subject = matches!(subject, LimitSubject::Team | LimitSubject::AppTeam);
        let category_subject =
            matches!(subject, LimitSubject::Category | LimitSubject::AppCategory);
        if self.team.is_some() && !team_subject {
            return Err(Error::configuration(format!(
                "a team is for the team subject, not {subject}"
            )));
        }
        if self.category.is_some() && !category_subject {
            return Err(Error::configuration(format!(
                "a category is for the category subject, not {subject}"
            )));
        }
        if matches!(subject, LimitSubject::AppTeam | LimitSubject::AppCategory) && !in_app {
            return Err(Error::configuration(format!(
                "an {subject} subject names its app"
            )));
        }
        Ok(subject)
    }
}

fn given(id: &Option<String>) -> Option<&str> {
    id.as_deref().filter(|id| !id.is_empty())
}

fn not_negative(amount: Usd) -> Result<()> {
    if amount.micros() < 0 {
        return Err(Error::configuration(format!(
            "a spend limit of {amount} is below zero"
        )));
    }
    Ok(())
}
