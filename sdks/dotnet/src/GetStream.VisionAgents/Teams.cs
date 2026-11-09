using GetStream.VisionAgents.Models;

namespace GetStream.VisionAgents;

/// <summary>
/// What one team may spend in each window of <see cref="Interval"/>: across the organization,
/// or within <see cref="AppId"/>.
/// </summary>
public sealed record TeamLimit
{
    /// <summary>The team, 1 to 59 characters of A-Z, a-z, 0-9, <c>_ . -</c>.</summary>
    public required string Team { get; init; }

    /// <summary>The window it counts in.</summary>
    public required LimitInterval Interval { get; init; }

    /// <summary>What the team may spend in each window.</summary>
    public required Usd Amount { get; init; }

    /// <summary>The app the limit is within. Null is across the organization.</summary>
    public string? AppId { get; init; }

    /// <summary>Why the limit is there. Null leaves an existing limit's note as it is.</summary>
    public string? Note { get; init; }
}

/// <summary>The organization's teams and their spend limits. Server side only.</summary>
/// <remarks>
/// A team has no members list: a user is in the team their token names. A team's spend is the
/// spend of the work those tokens ran, across the organization or within one app.
/// </remarks>
public sealed class Teams
{
    private readonly VisionAgentsClient _client;
    private readonly SpendLimits _limits;

    internal Teams(VisionAgentsClient client)
    {
        _client = client;
        _limits = new SpendLimits(client);
    }

    /// <summary>
    /// Every team with a limit or with spend this month, each with its apps, its spend in the
    /// current windows and its limits. With an app id, only within that app.
    /// </summary>
    public Task<List<Team>> ListAsync(string? appId = null, CancellationToken cancellationToken = default) =>
        _client.GetAsync<List<Team>>("/v1/teams", new Dictionary<string, string?> { ["app_id"] = appId }, cancellationToken);

    /// <summary>
    /// Makes the team's limit for the interval the amount given: created when it has none,
    /// changed when it has one.
    /// </summary>
    /// <exception cref="ConfigurationException">The team is no team name, or the amount is below zero.</exception>
    public async Task<Limit> SetLimitAsync(TeamLimit limit, CancellationToken cancellationToken = default)
    {
        SpendLimits.NotBelowZero(limit.Amount);
        var existing = await FindAsync(limit.Team, limit.Interval, limit.AppId, cancellationToken).ConfigureAwait(false);
        if (existing is null)
        {
            try
            {
                return await _limits.CreateAsync(new LimitOptions
                {
                    Subject = LimitSubject.Team,
                    Team = limit.Team,
                    AppId = limit.AppId,
                    Interval = limit.Interval,
                    Amount = limit.Amount,
                    Note = limit.Note,
                }, cancellationToken).ConfigureAwait(false);
            }
            catch (RouterException refused) when (refused.Code == "limit_exists")
            {
                // Somebody else created it since it was looked up.
            }
            existing = await FindAsync(limit.Team, limit.Interval, limit.AppId, cancellationToken).ConfigureAwait(false)
                ?? throw new RouterException(409, "POST /v1/spend/limits",
                    $"the {SpendLimits.Wire(limit.Interval)} limit on team {limit.Team} exists and could not be found");
        }
        return await _limits.UpdateAsync(existing.Id, new LimitChange
        {
            LimitMicros = limit.Amount.Micros,
            Note = VisionAgentsClient.Blank(limit.Note),
        }, cancellationToken).ConfigureAwait(false);
    }

    /// <summary>
    /// Deletes the team's limit for the interval, across the organization or within an app.
    /// Nothing happens when it has none.
    /// </summary>
    public async Task RemoveLimitAsync(string team, LimitInterval interval, string? appId = null, CancellationToken cancellationToken = default)
    {
        if (await FindAsync(team, interval, appId, cancellationToken).ConfigureAwait(false) is not { } existing)
        {
            return;
        }
        try
        {
            await _limits.DeleteAsync(existing.Id, cancellationToken).ConfigureAwait(false);
        }
        catch (RouterException refused) when (refused.Status == 404)
        {
            // Somebody else deleted it since it was looked up.
        }
    }

    /// <summary>The team's own limit in the interval, across the organization or within the app.</summary>
    private async Task<Limit?> FindAsync(string team, LimitInterval interval, string? appId, CancellationToken cancellationToken)
    {
        UsageCategories.CheckTeam(team);
        appId = VisionAgentsClient.Blank(appId);
        var wire = SpendLimits.Wire(interval);
        var limits = await _limits.ListAsync(new LimitQuery
        {
            Subject = LimitSubject.Team,
            Team = team,
            AppId = appId,
            Interval = interval,
        }, cancellationToken).ConfigureAwait(false);
        return limits.FirstOrDefault(limit =>
            limit.Team == team && limit.Interval == wire && VisionAgentsClient.Blank(limit.AppId) == appId);
    }
}
