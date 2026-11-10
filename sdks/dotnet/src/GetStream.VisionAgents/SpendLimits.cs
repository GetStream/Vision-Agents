using System.Globalization;
using GetStream.VisionAgents.Models;

namespace GetStream.VisionAgents;

/// <summary>What a limit or a reset is about.</summary>
public enum LimitSubject
{
    /// <summary>The whole organization.</summary>
    Organization,

    /// <summary>One app.</summary>
    App,

    /// <summary>A key, or every key of an app or the organization.</summary>
    ApiKey,

    /// <summary>An end user, or every end user of an app or the organization.</summary>
    User,

    /// <summary>A team across the organization, or within one app when an app id is set.</summary>
    Team,

    /// <summary>A team within one app. <see cref="Team"/> with an app id is the same subject.</summary>
    AppTeam,

    /// <summary>
    /// A usage category across the organization, or within one app when an app id is set. A
    /// team is not a category: use <see cref="Team"/>.
    /// </summary>
    Category,

    /// <summary>A usage category within one app. <see cref="Category"/> with an app id is the same.</summary>
    AppCategory,
}

/// <summary>The window a limit counts spend in, in UTC.</summary>
public enum LimitInterval
{
    /// <summary>The day.</summary>
    Daily,

    /// <summary>The week from Monday.</summary>
    Weekly,

    /// <summary>14 days counted from Monday 2024-01-01.</summary>
    Biweekly,

    /// <summary>The calendar month.</summary>
    Monthly,
}

/// <summary>
/// Whose ids a limit names. A key's or a user's limit at organization or app level is the
/// default for every key or user there, each counted on its own.
/// </summary>
public enum LimitLevel
{
    /// <summary>The organization.</summary>
    Organization,

    /// <summary>One app.</summary>
    App,

    /// <summary>One key.</summary>
    ApiKey,

    /// <summary>One end user.</summary>
    User,
}

/// <summary>A limit to create: what one subject may spend in each window of <see cref="Interval"/>.</summary>
/// <remarks>
/// The level is the most specific one the ids name, which the router works out:
/// <see cref="LimitSubject.User"/> with a <see cref="UserId"/> is that user's own limit, with
/// only an <see cref="AppId"/> the default for every user of the app, and alone the default for
/// every user of the organization. Keys go the same way. <see cref="Level"/> overrides it.
/// </remarks>
public sealed record LimitOptions
{
    /// <summary>What the limit is on.</summary>
    public required LimitSubject Subject { get; init; }

    /// <summary>The window it counts in.</summary>
    public required LimitInterval Interval { get; init; }

    /// <summary>What the subject may spend in each window. Zero refuses every billable request.</summary>
    public required Usd Amount { get; init; }

    /// <summary>The app, for an app's limit or one within it.</summary>
    public string? AppId { get; init; }

    /// <summary>The key, for one key's limit.</summary>
    public string? ApiKey { get; init; }

    /// <summary>The end user, for one user's limit.</summary>
    public string? UserId { get; init; }

    /// <summary>The team, for <see cref="LimitSubject.Team"/>.</summary>
    public string? Team { get; init; }

    /// <summary>The usage category, for <see cref="LimitSubject.Category"/>.</summary>
    public string? Category { get; init; }

    /// <summary>The level, in place of the one the ids name.</summary>
    public LimitLevel? Level { get; init; }

    /// <summary>Why the limit is there, for whoever reads it next.</summary>
    public string? Note { get; init; }
}

/// <summary>Which limits to list. Every field narrows; none lists them all.</summary>
public sealed record LimitQuery
{
    /// <summary>
    /// Only limits on this subject. <see cref="LimitSubject.Team"/> lists the teams' limits,
    /// <see cref="LimitSubject.Category"/> every other category's.
    /// </summary>
    public LimitSubject? Subject { get; init; }

    /// <summary>Only limits at this level.</summary>
    public LimitLevel? Level { get; init; }

    /// <summary>Only limits counting in this window.</summary>
    public LimitInterval? Interval { get; init; }

    /// <summary>Only limits on this app or within it.</summary>
    public string? AppId { get; init; }

    /// <summary>Only limits on this key.</summary>
    public string? ApiKey { get; init; }

    /// <summary>Only limits on this end user.</summary>
    public string? UserId { get; init; }

    /// <summary>Only limits on this team.</summary>
    public string? Team { get; init; }

    /// <summary>Only limits on this usage category.</summary>
    public string? Category { get; init; }
}

/// <summary>
/// Counts a subject's limits from now until each window ends: every interval's, or only those
/// in <see cref="Intervals"/>. Nothing is refunded: what it spent stays spent.
/// </summary>
public sealed record LimitReset
{
    /// <summary>Whose spend is counted again.</summary>
    public required LimitSubject Subject { get; init; }

    /// <summary>Why, for the audit log.</summary>
    public required string Reason { get; init; }

    /// <summary>The windows to reset. Null resets every one.</summary>
    public IReadOnlyList<LimitInterval>? Intervals { get; init; }

    /// <summary>The app, for an app or a subject within one.</summary>
    public string? AppId { get; init; }

    /// <summary>The key.</summary>
    public string? ApiKey { get; init; }

    /// <summary>The end user.</summary>
    public string? UserId { get; init; }

    /// <summary>The team, for <see cref="LimitSubject.Team"/>.</summary>
    public string? Team { get; init; }

    /// <summary>The usage category, for <see cref="LimitSubject.Category"/>.</summary>
    public string? Category { get; init; }
}

/// <summary>
/// The customer's spend limits: on the organization, its apps, keys, end users, teams and usage
/// categories.
/// </summary>
/// <remarks>
/// <para>
/// Server side only: the router refuses a user's token, 403 <c>server_side_only</c>. Every limit
/// here is the customer's own. Stream's spend limit for the organization is never among them,
/// and nothing here lifts it.
/// </para>
/// <para>
/// Work a limit refuses fails with a <see cref="RouterException"/> of 402
/// <c>spend_limit_reached</c>, or 429 <c>user_limit_reached</c> for one end user's limit.
/// Stream's own limit refuses with 402 <c>system_limit_reached</c>.
/// </para>
/// </remarks>
public sealed class SpendLimits
{
    private const string Path = "/v1/spend/limits";

    // The most the router answers in one page.
    private const int Page = 200;

    private readonly VisionAgentsClient _client;

    internal SpendLimits(VisionAgentsClient client) => _client = client;

    /// <summary>Every limit the query matches, read page by page.</summary>
    public async Task<List<Limit>> ListAsync(LimitQuery? query = null, CancellationToken cancellationToken = default)
    {
        query ??= new LimitQuery();
        var ids = new SubjectIds(query.AppId, query.ApiKey, query.UserId, query.Team, query.Category);
        var filter = new Dictionary<string, string?>
        {
            ["subject"] = ids.Filtering(query.Subject) is { } subject ? Wire(subject) : null,
            ["level"] = query.Level is { } level ? Wire(level) : null,
            ["interval"] = query.Interval is { } interval ? Wire(interval) : null,
            ["app_id"] = ids.AppId,
            // api_key in a query is the credential Stream's proxy reads.
            ["api_key_id"] = ids.ApiKey,
            ["user_id"] = ids.UserId,
            ["team"] = ids.Team,
            ["category"] = ids.Category,
            ["limit"] = Page.ToString(CultureInfo.InvariantCulture),
        };
        var limits = new List<Limit>();
        while (true)
        {
            var page = await _client.GetAsync<LimitPage>(Path, filter, cancellationToken).ConfigureAwait(false);
            limits.AddRange(page.Items ?? []);
            if (!page.HasMore || string.IsNullOrEmpty(page.NextCursor))
            {
                return limits;
            }
            filter["cursor"] = page.NextCursor;
        }
    }

    /// <summary>One limit, by its id.</summary>
    public Task<Limit> GetAsync(long limitId, CancellationToken cancellationToken = default) =>
        _client.GetAsync<Limit>(LimitPath(limitId), cancellationToken: cancellationToken);

    /// <summary>Creates a limit.</summary>
    /// <remarks>
    /// The router refuses with <see cref="RouterException.Code"/> <c>limit_above_parent</c> (the
    /// limit is above one of a subject that contains it), <c>limit_exists</c> (the subject has one
    /// at that level and interval: change it with <see cref="SetAmountAsync"/>) or <c>use_team</c>.
    /// </remarks>
    /// <exception cref="ConfigurationException">The ids do not name the subject, or the amount is below zero.</exception>
    public Task<Limit> CreateAsync(LimitOptions limit, CancellationToken cancellationToken = default)
    {
        var ids = new SubjectIds(limit.AppId, limit.ApiKey, limit.UserId, limit.Team, limit.Category);
        var kind = ids.Target(limit.Subject);
        NotBelowZero(limit.Amount);
        return _client.PostAsync<Limit>(Path, new LimitInput
        {
            Subject = Wire(kind),
            Interval = Wire(limit.Interval),
            LimitMicros = limit.Amount.Micros,
            AppId = ids.AppId,
            ApiKey = ids.ApiKey,
            UserId = ids.UserId,
            Team = ids.Team,
            Category = ids.Category,
            Level = limit.Level is { } level ? Wire(level) : null,
            Note = VisionAgentsClient.Blank(limit.Note),
        }, cancellationToken);
    }

    /// <summary>Changes what a limit allows. What it is on never changes.</summary>
    /// <exception cref="ConfigurationException">The amount is below zero.</exception>
    public Task<Limit> SetAmountAsync(long limitId, Usd amount, CancellationToken cancellationToken = default)
    {
        NotBelowZero(amount);
        return UpdateAsync(limitId, new LimitChange { LimitMicros = amount.Micros }, cancellationToken);
    }

    /// <summary>Changes a limit's note. An empty one clears it.</summary>
    public Task<Limit> SetNoteAsync(long limitId, string note, CancellationToken cancellationToken = default) =>
        UpdateAsync(limitId, new LimitChange { Note = note ?? "" }, cancellationToken);

    /// <summary>Deletes a limit. Its subject's spend from then on is held only to its parents'.</summary>
    public Task DeleteAsync(long limitId, CancellationToken cancellationToken = default) =>
        _client.DeleteAsync(LimitPath(limitId), cancellationToken);

    /// <summary>
    /// Counts a subject's limits from now, and answers what it spent against them. The router
    /// refuses with <see cref="RouterException.Code"/> <c>no_limit</c> a subject with no limit of
    /// the customer's.
    /// </summary>
    /// <exception cref="ConfigurationException">There is no reason, or the ids do not name the subject.</exception>
    public Task<SubjectUsage> ResetAsync(LimitReset reset, CancellationToken cancellationToken = default)
    {
        if (string.IsNullOrWhiteSpace(reset.Reason))
        {
            throw new ConfigurationException("a reset needs a reason");
        }
        var ids = new SubjectIds(reset.AppId, reset.ApiKey, reset.UserId, reset.Team, reset.Category);
        var kind = ids.Target(reset.Subject);
        return _client.PostAsync<SubjectUsage>($"{Path}/usage/reset", new LimitResetInput
        {
            Subject = Wire(kind),
            Reason = reset.Reason,
            Intervals = reset.Intervals is { Count: > 0 } intervals ? [.. intervals.Select(Wire)] : null,
            AppId = ids.AppId,
            ApiKey = ids.ApiKey,
            UserId = ids.UserId,
            Team = ids.Team,
            Category = ids.Category,
        }, cancellationToken);
    }

    internal Task<Limit> UpdateAsync(long limitId, LimitChange change, CancellationToken cancellationToken) =>
        _client.PatchAsync<Limit>(LimitPath(limitId), change, cancellationToken);

    internal static void NotBelowZero(Usd amount)
    {
        if (amount < Usd.Zero)
        {
            throw new ConfigurationException($"a spend limit of {amount} is below zero");
        }
    }

    internal static string Wire(LimitSubject subject) => subject switch
    {
        LimitSubject.Organization => "organization",
        LimitSubject.App => "app",
        LimitSubject.ApiKey => "api_key",
        LimitSubject.User => "user",
        LimitSubject.Team => "team",
        LimitSubject.AppTeam => "app_team",
        LimitSubject.Category => "category",
        LimitSubject.AppCategory => "app_category",
        _ => throw new ConfigurationException($"{subject} is not a subject a limit can be on"),
    };

    internal static string Wire(LimitInterval interval) => interval switch
    {
        LimitInterval.Daily => "daily",
        LimitInterval.Weekly => "weekly",
        LimitInterval.Biweekly => "biweekly",
        LimitInterval.Monthly => "monthly",
        _ => throw new ConfigurationException($"{interval} is not an interval a limit counts in"),
    };

    internal static string Wire(LimitLevel level) => level switch
    {
        LimitLevel.Organization => "organization",
        LimitLevel.App => "app",
        LimitLevel.ApiKey => "api_key",
        LimitLevel.User => "user",
        _ => throw new ConfigurationException($"{level} is not a level a limit can be at"),
    };

    private static string LimitPath(long limitId) => $"{Path}/{limitId.ToString(CultureInfo.InvariantCulture)}";

    /// <summary>The ids that name one subject, which every limit body and filter has. Empty ones are null.</summary>
    private sealed record SubjectIds
    {
        public SubjectIds(string? appId, string? apiKey, string? userId, string? team, string? category)
        {
            AppId = VisionAgentsClient.Blank(appId);
            ApiKey = VisionAgentsClient.Blank(apiKey);
            UserId = VisionAgentsClient.Blank(userId);
            Team = VisionAgentsClient.Blank(team);
            Category = VisionAgentsClient.Blank(category);
        }

        public string? AppId { get; }
        public string? ApiKey { get; }
        public string? UserId { get; }
        public string? Team { get; }
        public string? Category { get; }

        /// <summary>
        /// The subject a write is about, as the router names it: a team or category within an app
        /// is <see cref="LimitSubject.AppTeam"/> or <see cref="LimitSubject.AppCategory"/>. It
        /// refuses what the router would, before anything is sent.
        /// </summary>
        public LimitSubject Target(LimitSubject subject)
        {
            if (Category is not null)
            {
                UsageCategories.CheckCategory(Category);
            }
            if (Team is not null)
            {
                UsageCategories.CheckTeam(Team);
            }
            var inApp = AppId is not null;
            switch (subject)
            {
                case LimitSubject.Team or LimitSubject.AppTeam:
                    if (Team is null)
                    {
                        throw new ConfigurationException($"a {Wire(subject)} subject names its team in Team");
                    }
                    subject = inApp ? LimitSubject.AppTeam : subject;
                    break;
                case LimitSubject.Category or LimitSubject.AppCategory:
                    if (Category is null)
                    {
                        throw new ConfigurationException($"a {Wire(subject)} subject names its category in Category");
                    }
                    subject = inApp ? LimitSubject.AppCategory : subject;
                    break;
            }
            if (Team is not null && subject is not (LimitSubject.Team or LimitSubject.AppTeam))
            {
                throw new ConfigurationException($"Team is for the team subject, not {Wire(subject)}");
            }
            if (Category is not null && subject is not (LimitSubject.Category or LimitSubject.AppCategory))
            {
                throw new ConfigurationException($"Category is for the category subject, not {Wire(subject)}");
            }
            if (!inApp && subject is LimitSubject.AppTeam or LimitSubject.AppCategory)
            {
                throw new ConfigurationException($"an {Wire(subject)} subject names its app in AppId");
            }
            return subject;
        }

        /// <summary>A list's subject, as the router names it, with its names checked.</summary>
        public LimitSubject? Filtering(LimitSubject? subject)
        {
            if (subject is { } named)
            {
                return Target(named);
            }
            if (Team is not null)
            {
                UsageCategories.CheckTeam(Team);
            }
            if (Category is not null)
            {
                UsageCategories.CheckCategory(Category);
            }
            return null;
        }
    }
}
