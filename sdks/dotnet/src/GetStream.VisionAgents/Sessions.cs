using System.Globalization;
using System.Text.Json;
using GetStream.VisionAgents.Models;

namespace GetStream.VisionAgents;

/// <summary>Which sessions to list. Every field narrows; none lists them all.</summary>
public sealed record SessionQuery
{
    /// <summary>Sessions of the agent config of this name.</summary>
    public string? Agent { get; init; }

    /// <summary>Sessions of this agent config.</summary>
    public string? ConfigId { get; init; }

    /// <summary>Sessions belonging to this user. A server-side caller may name anybody.</summary>
    public string? UserId { get; init; }

    /// <summary>Sessions filed under this project.</summary>
    public string? Project { get; init; }

    /// <summary>Sessions in this state.</summary>
    public string? State { get; init; }

    /// <summary>Sessions whose custom data holds all of these.</summary>
    public Dictionary<string, object?>? Custom { get; init; }

    /// <summary>Sessions created after this.</summary>
    public DateTimeOffset? CreatedAfter { get; init; }

    /// <summary>Sessions created before this.</summary>
    public DateTimeOffset? CreatedBefore { get; init; }

    /// <summary>How many to return.</summary>
    public int? Limit { get; init; }

    /// <summary>How many to skip.</summary>
    public int? Offset { get; init; }
}

/// <summary>
/// The sessions already held: listed, searched and read back.
/// </summary>
/// <remarks>
/// What these return are records. Opening a conversation somebody talks in is an agent's
/// job, since it is the agent that runs the tools.
/// </remarks>
public sealed class Sessions(VisionAgentsClient client)
{
    /// <summary>Sessions, newest first. A page shorter than the limit is the last.</summary>
    public async Task<List<Session>> ListAsync(SessionQuery? query = null, CancellationToken cancellationToken = default)
    {
        var listed = await client.GetAsync<List<Models.Session>>("/v1/agents/sessions", Query(query ?? new SessionQuery()), cancellationToken)
            .ConfigureAwait(false);
        return listed.ConvertAll(session => Session.Read(client, session));
    }

    /// <summary>Sessions by what they were called, best match first. What was said is not searched.</summary>
    public async Task<List<Session>> SearchAsync(string text, SessionQuery? query = null, CancellationToken cancellationToken = default)
    {
        var parameters = Query(query ?? new SessionQuery());
        parameters["q"] = text;
        var found = await client.GetAsync<List<Models.Session>>("/v1/agents/sessions/search", parameters, cancellationToken)
            .ConfigureAwait(false);
        return found.ConvertAll(session => Session.Read(client, session));
    }

    /// <summary>One session.</summary>
    public async Task<Session> GetAsync(string id, CancellationToken cancellationToken = default) =>
        Session.Read(client, await client.GetAsync<Models.Session>(
            $"/v1/agents/sessions/{VisionAgentsClient.Escape(id)}", cancellationToken: cancellationToken).ConfigureAwait(false));

    /// <summary>A session's turns, by its id, without reading the session first.</summary>
    public Responses Responses(string id) => new(client, id);

    private static Dictionary<string, string?> Query(SessionQuery query) => new()
    {
        ["agent"] = VisionAgentsClient.Blank(query.Agent),
        ["config_id"] = VisionAgentsClient.Blank(query.ConfigId),
        ["user_id"] = VisionAgentsClient.Blank(query.UserId),
        ["project"] = VisionAgentsClient.Blank(query.Project),
        ["state"] = VisionAgentsClient.Blank(query.State),
        ["custom"] = query.Custom is { Count: > 0 } ? JsonSerializer.Serialize(query.Custom, Json.Options) : null,
        ["created_after"] = query.CreatedAfter?.ToString("O", CultureInfo.InvariantCulture),
        ["created_before"] = query.CreatedBefore?.ToString("O", CultureInfo.InvariantCulture),
        ["limit"] = VisionAgentsClient.Number(query.Limit),
        ["offset"] = VisionAgentsClient.Number(query.Offset),
    };
}
