using System.Text.Json.Serialization;
using GetStream.VisionAgents.Models;

namespace GetStream.VisionAgents;

/// <summary>
/// One key of a session query's order: a field, and which way it runs.
/// </summary>
/// <remarks>
/// The fields are <c>last_message_at</c>; <c>pinned</c>, alone or followed by
/// <c>last_message_at</c>, which lists the pinned sessions first; <c>review_score</c>, for a
/// server-side caller; and <c>relevance</c>, for a text search and nothing else.
/// </remarks>
/// <param name="Field">The field to order by, spelled as the router spells it.</param>
/// <param name="Direction">-1 is descending and 1 ascending. Null is the router's own, descending. <c>pinned</c> and <c>relevance</c> only sort descending.</param>
public sealed record SessionSort(
    [property: JsonPropertyName("field")] string Field,
    [property: JsonPropertyName("direction")] int? Direction = null);

/// <summary>Which sessions to query, in what order, and which page of them.</summary>
public sealed record SessionQuery
{
    /// <summary>
    /// Which sessions, as the router reads it: the keys are the wire's own and the whole of it
    /// is sent as it is. Null lists every session the caller may see.
    /// </summary>
    /// <remarks>
    /// <para>
    /// The keys of one object are ANDed. <c>$and</c> and <c>$or</c> each take 1 to 10 filters
    /// and nest up to 3 deep. The fields, and nothing else:
    /// </para>
    /// <list type="bullet">
    /// <item><c>project_id</c>: a string, <c>{"$eq": s}</c> or <c>{"$in": [up to 50 strings]}</c>.</item>
    /// <item><c>agent</c>: a string or <c>{"$eq": s}</c>. A query made through an agent sets it at the top level.</item>
    /// <item><c>text</c>: <c>{"$q": "words"}</c>, once, at the top level or directly in the top-level <c>$and</c>, never inside <c>$or</c>. <see cref="Sessions.SearchAsync"/> sets it.</item>
    /// <item><c>review_score</c>: 1 to 5, or <c>{"$eq"|"$gt"|"$gte"|"$lt"|"$lte": n}</c>, several of them ANDed. Server side only.</item>
    /// <item><c>user_id</c>: a string or <c>{"$eq": s}</c>, at the top level or directly in the top-level <c>$and</c>. Server side only.</item>
    /// </list>
    /// <para>Any other key or operator is refused by the router with a 400.</para>
    /// </remarks>
    public Dictionary<string, object?>? Filter { get; init; }

    /// <summary>
    /// The order. Null is the router's own: the most recent message first, or the best match
    /// first for a text search. A cursor only carries on the order it came from.
    /// </summary>
    public IReadOnlyList<SessionSort>? Sort { get; init; }

    /// <summary>How many to return, 1 to 200. Null is the router's 25.</summary>
    public int? Limit { get; init; }

    /// <summary>The <c>NextCursor</c> of the page before, with the same filter and sort. Null is the first page.</summary>
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
    /// A page of sessions, the most recent message first unless <see cref="SessionQuery.Sort"/>
    /// says otherwise, the ones that ended included. Pass the page's <c>NextCursor</c> as
    /// <see cref="SessionQuery.Cursor"/> for the next one.
    /// </summary>
    /// <remarks>Through an agent, only that agent's sessions: the filter's top-level <c>agent</c> is set to its name.</remarks>
    public Task<SessionPage> QueryAsync(SessionQuery? query = null, CancellationToken cancellationToken = default)
    {
        query ??= new SessionQuery();
        return client.PostAsync<SessionPage>("/v1/agents/sessions/query",
            new QueryBody(Scoped(query.Filter), query.Sort is { Count: > 0 } ? query.Sort : null, query.Limit, VisionAgentsClient.Blank(query.Cursor)),
            cancellationToken);
    }

    /// <summary>
    /// Sessions by their title, description, project and agent name, best match first. What
    /// was said is not searched. It takes the filter <see cref="QueryAsync"/> does and pages
    /// the same way, and has no order of its own to choose.
    /// </summary>
    /// <exception cref="ArgumentException">The filter already has a top-level <c>text</c>, or there is nothing to search for.</exception>
    public Task<SessionPage> SearchAsync(
        string text,
        Dictionary<string, object?>? filter = null,
        int? limit = null,
        string? cursor = null,
        CancellationToken cancellationToken = default)
    {
        ArgumentException.ThrowIfNullOrWhiteSpace(text);
        if (filter?.ContainsKey("text") == true)
        {
            throw new ArgumentException("the filter already has a text; a search sets it", nameof(filter));
        }
        var sent = Scoped(filter) ?? new Dictionary<string, object?>();
        sent["text"] = new Dictionary<string, object?> { ["$q"] = text };
        return client.PostAsync<SessionPage>("/v1/agents/sessions/query",
            new QueryBody(sent, null, limit, VisionAgentsClient.Blank(cursor)), cancellationToken);
    }

    /// <summary>One session.</summary>
    public async Task<Session> GetAsync(string id, CancellationToken cancellationToken = default) =>
        Session.Read(client, await client.GetAsync<Models.Session>(
            $"/v1/agents/sessions/{VisionAgentsClient.Escape(id)}", cancellationToken: cancellationToken).ConfigureAwait(false));

    /// <summary>
    /// Changes one session, whether or not it is still being held, and returns it as it now is.
    /// A field left null is left as it is.
    /// </summary>
    /// <remarks>
    /// One that ended can still be renamed, relabelled and pinned; models and voice need it running, and take over from its next turn.
    /// A device may change the title, description, custom labels and pin of its own sessions; the rest only a backend may ask.
    /// </remarks>
    public Task<Models.Session> UpdateAsync(string id, UpdateSessionRequest update, CancellationToken cancellationToken = default) =>
        client.PatchAsync<Models.Session>($"/v1/agents/sessions/{VisionAgentsClient.Escape(id)}", update, cancellationToken);

    /// <summary>
    /// Pins a session, running or ended, and returns it as it now is, with its <c>PinnedAt</c>.
    /// One already pinned keeps the time it was first pinned.
    /// </summary>
    /// <remarks>A pin changes nothing else, the session's place in a <c>last_message_at</c> order included; sort by <c>pinned</c> to list them first.</remarks>
    public Task<Models.Session> PinAsync(string id, CancellationToken cancellationToken = default) =>
        UpdateAsync(id, new UpdateSessionRequest { Pinned = true }, cancellationToken);

    /// <summary>Unpins a session and returns it as it now is, with no <c>PinnedAt</c>.</summary>
    public Task<Models.Session> UnpinAsync(string id, CancellationToken cancellationToken = default) =>
        UpdateAsync(id, new UpdateSessionRequest { Pinned = false }, cancellationToken);

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
    /// The filter as it is sent: the caller's top-level keys, with <c>agent</c> set when this
    /// is one agent's sessions. Null when there is nothing to send.
    /// </summary>
    /// <remarks>
    /// The caller's filter is not wrapped in a <c>$and</c>: <c>text</c> and <c>user_id</c> are
    /// only read at the top level or in the top-level <c>$and</c>, and a wrap would push them
    /// one deeper.
    /// </remarks>
    private Dictionary<string, object?>? Scoped(Dictionary<string, object?>? filter)
    {
        var name = VisionAgentsClient.Blank(agent);
        if (name is null && filter is not { Count: > 0 })
        {
            return null;
        }
        var sent = filter is null ? new Dictionary<string, object?>() : new Dictionary<string, object?>(filter);
        if (name is not null)
        {
            sent["agent"] = name;
        }
        return sent;
    }

    /// <summary>What a query posts. A null is left out, which is the router's own.</summary>
    private sealed record QueryBody(
        [property: JsonPropertyName("filter")] Dictionary<string, object?>? Filter,
        [property: JsonPropertyName("sort")] IReadOnlyList<SessionSort>? Sort,
        [property: JsonPropertyName("limit")] int? Limit,
        [property: JsonPropertyName("cursor")] string? Cursor);
}
