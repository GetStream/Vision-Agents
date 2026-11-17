use crate::backend::check_team;
use crate::error::{Error, Result};
use crate::limits::{Limit, LimitFilter, SpendLimits, Target};
use crate::operations::ListTeamsQuery;
use crate::types::{self, LimitInterval, LimitSubject};
use crate::usd::Usd;

/// The organization's teams. Server side only.
///
/// A team has no members list: a user is in the team their token names. A team's spend is the
/// spend of the work those tokens ran, across the organization or within one app.
#[derive(Debug, Clone)]
pub struct Teams {
    limits: SpendLimits,
}

/// What one team may spend in each window of `interval`: across the organization, or within
/// one app.
#[derive(Debug, Clone, PartialEq)]
pub struct TeamLimit {
    pub team: String,
    pub interval: LimitInterval,
    pub amount: Usd,
    pub app_id: Option<String>,
    /// `None` keeps the note a limit already has.
    pub note: Option<String>,
}

impl TeamLimit {
    pub fn new(team: impl Into<String>, interval: LimitInterval, amount: Usd) -> Self {
        TeamLimit {
            team: team.into(),
            interval,
            amount,
            app_id: None,
            note: None,
        }
    }

    pub fn app_id(mut self, app_id: impl Into<String>) -> Self {
        self.app_id = Some(app_id.into());
        self
    }

    pub fn note(mut self, note: impl Into<String>) -> Self {
        self.note = Some(note.into());
        self
    }
}

impl Teams {
    pub(crate) fn new(limits: SpendLimits) -> Self {
        Teams { limits }
    }

    /// Every team with a limit or with spend this month, each with its apps, its spend in
    /// the current windows and its limits. With an app id, only within that app.
    pub async fn list(&self, app_id: Option<&str>) -> Result<Vec<types::Team>> {
        self.limits
            .client()
            .list_teams(&ListTeamsQuery {
                app_id: app_id.map(Into::into),
            })
            .await
    }

    /// Makes the team's limit for the interval the amount given: created when it has none,
    /// changed when it has one.
    pub async fn set_limit(&self, limit: &TeamLimit) -> Result<types::Limit> {
        if limit.amount.micros() < 0 {
            return Err(Error::configuration(format!(
                "a spend limit of {} is below zero",
                limit.amount
            )));
        }
        let app_id = limit.app_id.as_deref();
        let existing = match self.limit(&limit.team, app_id, &limit.interval).await? {
            Some(existing) => existing,
            None => {
                let mut create =
                    Limit::new(LimitSubject::Team, limit.interval.clone(), limit.amount)
                        .team(&limit.team);
                create.app_id = limit.app_id.clone();
                create.note = limit.note.clone();
                match self.limits.create(&create).await {
                    // Somebody else created it since it was looked up.
                    Err(Error::Router(refused)) if refused.code == "limit_exists" => self
                        .limit(&limit.team, app_id, &limit.interval)
                        .await?
                        .ok_or_else(|| {
                            Error::configuration(format!(
                                "the {} limit on team {} exists and could not be found",
                                limit.interval, limit.team
                            ))
                        })?,
                    created => return created,
                }
            }
        };
        self.limits
            .update(
                existing.id,
                &types::LimitChange {
                    limit_micros: Some(limit.amount.micros()),
                    note: limit.note.clone(),
                },
            )
            .await
    }

    /// Deletes the team's limit for the interval, across the organization or within one app.
    /// Nothing happens when it has none.
    pub async fn remove_limit(
        &self,
        team: &str,
        interval: LimitInterval,
        app_id: Option<&str>,
    ) -> Result<()> {
        let Some(existing) = self.limit(team, app_id, &interval).await? else {
            return Ok(());
        };
        match self.limits.delete(existing.id).await {
            // Somebody else deleted it since it was looked up.
            Err(Error::Router(refused)) if refused.status == 404 => Ok(()),
            deleted => deleted,
        }
    }

    /// What the team spent in its current windows, against its limits and grants, across
    /// the organization or within one app.
    pub async fn usage(&self, team: &str, app_id: Option<&str>) -> Result<types::SubjectUsage> {
        let mut target = Target::new(LimitSubject::Team).team(team);
        target.app_id = app_id.map(Into::into);
        self.limits.usage(&target).await
    }

    /// The team's own limit in the interval, across the organization or within the app.
    async fn limit(
        &self,
        team: &str,
        app_id: Option<&str>,
        interval: &LimitInterval,
    ) -> Result<Option<types::Limit>> {
        check_team(team)?;
        let limits = self
            .limits
            .list(&LimitFilter {
                subject: Some(LimitSubject::Team),
                interval: Some(interval.clone()),
                app_id: app_id.map(Into::into),
                team: Some(team.into()),
                ..Default::default()
            })
            .await?;
        Ok(limits.into_iter().find(|limit| {
            limit.team.as_deref() == Some(team)
                && limit.interval == *interval
                && limit.app_id.as_deref().filter(|id| !id.is_empty())
                    == app_id.filter(|id| !id.is_empty())
        }))
    }
}
