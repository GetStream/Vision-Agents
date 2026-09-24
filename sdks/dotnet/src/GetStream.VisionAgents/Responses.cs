using System.Runtime.CompilerServices;
using GetStream.VisionAgents.Models;

namespace GetStream.VisionAgents;

/// <summary>
/// One turn, and a way to read what it was made of.
/// </summary>
/// <remarks>
/// <see cref="Responses.CreateAsync"/> returns once the agent has started answering rather
/// than when it has finished, because a model takes seconds and a request that waited them
/// out would time out on anything worth asking. So this is a handle on an answer in
/// progress: <see cref="Items"/> reads what has been written down so far, and the session's
/// events are what watch it arrive.
/// </remarks>
public sealed class Response
{
    internal Response(VisionAgentsClient client, AgentResponse created)
    {
        Created = created;
        Items = new Items(client, created.SessionId, created.Id);
    }

    /// <summary>What the router said when it accepted the turn.</summary>
    public AgentResponse Created { get; }

    /// <summary>
    /// The backend's id for this turn, empty for a session that records nothing. It is not
    /// the <c>turn_id</c> socket events carry.
    /// </summary>
    public string Id => Created.Id;

    /// <summary>Where the turn had got to when it was accepted.</summary>
    public string Status => Created.Status;

    /// <summary>What this one turn was made of.</summary>
    public Items Items { get; }
}

/// <summary>
/// The things one or more turns were made of, in the order they happened.
/// </summary>
/// <remarks>
/// Read rather than watched: this is what the backend wrote down, so it reads the same
/// whether the conversation is still going or ended last week. Deltas are not here; a
/// hundred fragments of one sentence are the sentence.
/// </remarks>
public sealed class Items
{
    /// <summary>How many items are read per request while unwinding.</summary>
    public const int Page = 200;

    // As many as the router hands over at once.
    private const int Ceiling = 1000;

    private readonly VisionAgentsClient _client;
    private readonly string _sessionId;
    private readonly string _responseId;

    internal Items(VisionAgentsClient client, string sessionId, string responseId = "")
    {
        _client = client;
        _sessionId = sessionId;
        _responseId = responseId;
    }

    /// <summary>Every item, oldest first, fetched a page at a time.</summary>
    public async IAsyncEnumerable<AgentResponseItem> UnwindAsync(
        int page = Page,
        [EnumeratorCancellation] CancellationToken cancellationToken = default)
    {
        page = Math.Clamp(page <= 0 ? Page : page, 1, Ceiling);
        var offset = 0;
        while (true)
        {
            var read = await ListAsync(page, offset, cancellationToken).ConfigureAwait(false);
            foreach (var item in read)
            {
                yield return item;
            }
            // A short page is the last one; asking again to see an empty one would double
            // the requests for every conversation that is an exact multiple of the page.
            if (read.Count < page)
            {
                yield break;
            }
            offset += read.Count;
        }
    }

    /// <summary>One page of items, for a caller doing its own paging.</summary>
    public Task<List<AgentResponseItem>> ListAsync(int? limit = null, int? offset = null, CancellationToken cancellationToken = default) =>
        _client.GetAsync<List<AgentResponseItem>>(
            $"/v1/agents/sessions/{VisionAgentsClient.Escape(_sessionId)}/responses/items",
            new Dictionary<string, string?>
            {
                ["response_id"] = VisionAgentsClient.Blank(_responseId),
                ["limit"] = VisionAgentsClient.Number(limit),
                ["offset"] = VisionAgentsClient.Number(offset),
            },
            cancellationToken);

    /// <summary>Everything in one list, for a conversation short enough to hold.</summary>
    public async Task<List<AgentResponseItem>> AllAsync(CancellationToken cancellationToken = default)
    {
        var all = new List<AgentResponseItem>();
        await foreach (var item in UnwindAsync(cancellationToken: cancellationToken).ConfigureAwait(false))
        {
            all.Add(item);
        }
        return all;
    }
}

/// <summary>
/// A session's turns.
/// </summary>
/// <remarks>
/// <see cref="Items"/> here is the whole conversation flattened, which is how it reads and
/// how it gets rendered. A single turn's items come off the <see cref="Response"/> that
/// <see cref="CreateAsync"/> returns.
/// </remarks>
public sealed class Responses
{
    private readonly VisionAgentsClient _client;
    private readonly string _sessionId;

    internal Responses(VisionAgentsClient client, string sessionId)
    {
        _client = client;
        _sessionId = sessionId;
        Items = new Items(client, sessionId);
    }

    /// <summary>Everything the conversation was made of.</summary>
    public Items Items { get; }

    /// <summary>
    /// Asks the agent something and names the turn it answers as.
    /// </summary>
    /// <remarks>
    /// An incognito session records nothing, so the turn it hands back has no id: there is
    /// nothing to read back afterwards, which is what incognito means.
    /// </remarks>
    public async Task<Response> CreateAsync(string text, IReadOnlyList<ImageSource>? images = null, CancellationToken cancellationToken = default)
    {
        var request = new CreateResponseRequest { Text = text, Images = images is { Count: > 0 } ? [.. images] : null };
        var created = await _client.PostAsync<AgentResponse>(
            $"/v1/agents/sessions/{VisionAgentsClient.Escape(_sessionId)}/responses", request, cancellationToken).ConfigureAwait(false);
        return new Response(_client, created);
    }

    /// <summary>The turns so far, oldest first.</summary>
    public Task<List<AgentResponse>> ListAsync(int? limit = null, int? offset = null, CancellationToken cancellationToken = default) =>
        _client.GetAsync<List<AgentResponse>>(
            $"/v1/agents/sessions/{VisionAgentsClient.Escape(_sessionId)}/responses",
            new Dictionary<string, string?>
            {
                ["limit"] = VisionAgentsClient.Number(limit),
                ["offset"] = VisionAgentsClient.Number(offset),
            },
            cancellationToken);

    /// <summary>Goes back to a response and carries on from there.</summary>
    public Task RewindAsync(Response to, CancellationToken cancellationToken = default) =>
        RewindAsync(to.Id, cancellationToken);

    /// <summary>Goes back to the response an item belongs to.</summary>
    public Task RewindAsync(AgentResponseItem to, CancellationToken cancellationToken = default) =>
        RewindAsync(to.ResponseId, cancellationToken);

    /// <summary>
    /// Goes back to a response, by its id, and carries on from there.
    /// </summary>
    /// <remarks>
    /// The reply being spoken is abandoned and the conversation continues as though nothing
    /// after that response had been said. The response itself is kept. A conversation kept
    /// in Stream Chat cannot be rewound, because its transcript lives there; fork it at the
    /// response instead, with <see cref="ForkOptions.ResponseId"/>.
    /// </remarks>
    public Task RewindAsync(string responseId, CancellationToken cancellationToken = default)
    {
        if (string.IsNullOrEmpty(responseId))
        {
            throw new ConfigurationException(
                "that response has no id, which is what a session that records nothing hands back; there is nothing to rewind to");
        }
        return _client.PostAsync($"/v1/agents/sessions/{VisionAgentsClient.Escape(_sessionId)}/rewind",
            new RewindSessionRequest { ResponseId = responseId }, cancellationToken);
    }
}
