using GetStream.VisionAgents.Models;

namespace GetStream.VisionAgents;

/// <summary>Which sessions to query. Every field narrows; none lists them all.</summary>
public sealed record SessionQuery
{
    /// <summary>Sessions of the agent config of this name.</summary>
    public string? Agent { get; init; }

    /// <summary>Sessions created with this agent id.</summary>
    public string? AgentId { get; init; }

    /// <summary>Sessions belonging to this user. A server-side caller may name anybody.</summary>
    public string? UserId { get; init; }

    /// <summary>Sessions filed under this project. A search covers every project, so it refuses this.</summary>
    public string? ProjectId { get; init; }

    /// <summary>How the user took part: <c>text</c>, <c>voice</c> or <c>video</c>.</summary>
    public string? Modality { get; init; }

    /// <summary>The sessions still running, <c>live</c>, or the ones over, <c>ended</c>.</summary>
    public string? State { get; init; }

    /// <summary>How many to return, up to 200. Null is the router's 25.</summary>
    public int? Limit { get; init; }

    /// <summary>The <c>NextCursor</c> of the page before, with the same filters. Null is the first page.</summary>
    public string? Cursor { get; init; }
}

/// <summary>
/// The sessions already held: queried, searched, read back, changed and deleted.
/// </summary>
/// <remarks>
/// What these return are records. Opening a conversation somebody talks in is an agent's
/// job, since it is the agent that runs the tools.
/// </remarks>
public sealed class Sessions(VisionAgentsClient client, string? agent = null)
{
    /// <summary>
    /// A page of sessions, most recently updated first, the ones that ended included. Pass
    /// the page's <c>NextCursor</c> as <see cref="SessionQuery.Cursor"/> for the next one.
    /// </summary>
    public Task<SessionPage> QueryAsync(SessionQuery? query = null, CancellationToken cancellationToken = default) =>
        client.PostAsync<SessionPage>("/v1/agents/sessions/query", Body(null, query ?? new SessionQuery()), cancellationToken);

    /// <summary>Sessions by what they were called, best match first. What was said is not searched. It pages the way <see cref="QueryAsync"/> does.</summary>
    public Task<SessionPage> SearchAsync(string text, SessionQuery? query = null, CancellationToken cancellationToken = default) =>
        client.PostAsync<SessionPage>("/v1/agents/sessions/query", Body(text, query ?? new SessionQuery()), cancellationToken);

    /// <summary>One session.</summary>
    public async Task<Session> GetAsync(string id, CancellationToken cancellationToken = default) =>
        Session.Read(client, await client.GetAsync<Models.Session>(
            $"/v1/agents/sessions/{VisionAgentsClient.Escape(id)}", cancellationToken: cancellationToken).ConfigureAwait(false));

    /// <summary>
    /// Changes one session, whether or not it is still being held, and returns it as it now is.
    /// A field left null is left as it is.
    /// </summary>
    /// <remarks>
    /// One that ended can still be renamed and relabelled; instructions, models and voice
    /// need it running, and take over from its next turn. Only a backend may ask.
    /// </remarks>
    public Task<Models.Session> UpdateAsync(string id, UpdateSessionRequest update, CancellationToken cancellationToken = default) =>
        client.PatchAsync<Models.Session>($"/v1/agents/sessions/{VisionAgentsClient.Escape(id)}", update, cancellationToken);

    /// <summary>
    /// Deletes a session, running or ended: it is stopped, and its turns and what it
    /// remembered are deleted with it. The user's other memories are kept.
    /// </summary>
    public Task DeleteAsync(string id, CancellationToken cancellationToken = default) =>
        client.DeleteAsync($"/v1/agents/sessions/{VisionAgentsClient.Escape(id)}", cancellationToken);

    /// <summary>Deletes what one session remembered, running or ended, and leaves the rest of the user's memories alone. Only a backend may ask.</summary>
    public Task DeleteMemoriesAsync(string id, CancellationToken cancellationToken = default) =>
        client.DeleteAsync($"/v1/agents/sessions/{VisionAgentsClient.Escape(id)}/memories", cancellationToken);

    /// <summary>A session's turns, by its id, without reading the session first.</summary>
    public Responses Responses(string id) => new(client, id);

    /// <summary>
    /// The query the listing and the search share, so the two cannot drift apart in what they
    /// honour. Text makes it a search.
    /// </summary>
    private Models.SessionQuery Body(string? text, SessionQuery query) => new()
    {
        Filter = new SessionFilter
        {
            Agent = VisionAgentsClient.Blank(query.Agent) ?? VisionAgentsClient.Blank(agent),
            AgentId = VisionAgentsClient.Blank(query.AgentId),
            UserId = VisionAgentsClient.Blank(query.UserId),
            ProjectId = VisionAgentsClient.Blank(query.ProjectId),
            Modality = VisionAgentsClient.Blank(query.Modality),
            State = VisionAgentsClient.Blank(query.State),
            Text = text is { Length: > 0 } ? new TextMatch { Q = text } : null,
        },
        Limit = query.Limit is > 0 ? query.Limit : null,
        Cursor = VisionAgentsClient.Blank(query.Cursor),
    };
}
