using GetStream.VisionAgents.Models;

namespace GetStream.VisionAgents;

/// <summary>Where code the agent writes gets run.</summary>
/// <remarks>
/// Code execution never happens on the live speech path: a sandbox is offered to the slower
/// model doing delegated work, not to the one holding the conversation.
/// </remarks>
/// <param name="Provider">The sandbox provider's name, as the backend knows it.</param>
public sealed record Sandbox(string Provider)
{
    /// <summary>A Daytona sandbox. The backend needs <c>DAYTONA_API_KEY</c> for it to do anything.</summary>
    public static Sandbox Daytona() => new("daytona");
}

/// <summary>
/// What stands between what a caller said and the model that answers them.
/// </summary>
/// <remarks>
/// The loop runs in the backend and is part of the agent's stored config, never of a
/// session: <see cref="Agent.SyncAsync"/> writes it, and every session opened under that
/// config runs it.
/// </remarks>
public sealed record Harness
{
    /// <summary>Which harness the backend runs. Null is <c>default</c>, the only one there is.</summary>
    public string? Name { get; init; }

    /// <summary>
    /// Model targets for the work handed over, keyed by name. The entry under
    /// <c>default</c>, or the only entry, is the model that runs skills.
    /// </summary>
    public IReadOnlyDictionary<string, string>? Subagents { get; init; }

    /// <summary>Where delegated code runs.</summary>
    public Sandbox? Sandbox { get; init; }

    /// <summary>Skills of your own, stored and named by the config in place of the built-in set.</summary>
    public IReadOnlyList<Skill>? Skills { get; init; }

    /// <summary>The model that runs delegated work, or empty when nothing is delegated.</summary>
    public string Subagent => Subagents switch
    {
        null or { Count: 0 } => "",
        _ when Subagents.TryGetValue("default", out var named) => named,
        { Count: 1 } => Subagents.Values.First(),
        _ => "",
    };

    internal void Validate()
    {
        if (Name is { Length: > 0 } && Name != "default")
        {
            throw new ConfigurationException($"there is no harness called {Name}");
        }
        if (Subagents is { Count: > 1 } && !Subagents.ContainsKey("default"))
        {
            throw new ConfigurationException("several subagents and no \"default\", so which one runs skills is undecided");
        }
        foreach (var skill in Skills ?? [])
        {
            if (skill.Name == "" || skill.Description == "" || skill.Instructions == "")
            {
                throw new ConfigurationException($"skill \"{skill.Name}\" needs a name, a description and instructions");
            }
        }
        if (Sandbox is { Provider: "" })
        {
            throw new ConfigurationException("a sandbox needs a provider");
        }
    }
}

/// <summary>
/// The models the backend runs. Every target is a provider/model name or a capability
/// shortcut such as <c>llm-fast</c>; one left null takes the stored config's, or the backend's default.
/// </summary>
public sealed record Pipeline
{
    /// <summary>The model that answers.</summary>
    public string? Llm { get; init; }

    /// <summary>The model that transcribes.</summary>
    public string? Stt { get; init; }

    /// <summary>The model that speaks.</summary>
    public string? Tts { get; init; }

    /// <summary>A provider-specific voice id.</summary>
    public string? Voice { get; init; }

    /// <summary>A hint, which narrows the candidates in every modality.</summary>
    public string? Language { get; init; }

    /// <summary>Said on joining, without going through the model.</summary>
    public string? Greeting { get; init; }

    /// <summary>
    /// <c>exact</c>, the default, or <c>variation</c>, which has the model say its own
    /// variation of <see cref="Greeting"/> on every call.
    /// </summary>
    public string? GreetingMode { get; init; }

    /// <summary>Murmurs while a caller is still talking, the way a person does.</summary>
    public bool? Backchannel { get; init; }

    /// <summary>A ceiling on a reply.</summary>
    public int? MaxTokens { get; init; }

    /// <summary>How long the model waits for one of your tools before carrying on without it.</summary>
    public TimeSpan? ToolTimeout { get; init; }

    /// <summary>Which video a skill that captures it sees.</summary>
    public SessionVideo? Video { get; init; }
}

/// <summary>What one conversation is about, as opposed to the agent behind it.</summary>
public sealed record SessionOptions
{
    /// <summary>
    /// The id to hold the session by, up to 64 of <c>A-Za-z0-9_-</c>, for a caller that wants
    /// to know it before the session exists. Null lets the router generate one; one already
    /// taken is refused with a 409.
    /// </summary>
    public string? Id { get; init; }

    /// <summary>
    /// The conversation being answered, which names the channel replies are written into.
    /// A worker answering several conversations has to set it.
    /// </summary>
    public string? AgentId { get; init; }

    /// <summary>What a person finds the conversation by afterwards.</summary>
    public string? Title { get; init; }

    /// <summary>What it is about. Searched, like the title.</summary>
    public string? Description { get; init; }

    /// <summary>Groups conversations, and is carried as a cost label too.</summary>
    public string? ProjectId { get; init; }

    /// <summary>Anything of the caller's own, which a later query can match on.</summary>
    public Dictionary<string, object?>? Custom { get; init; }

    /// <summary>Holds the conversation and keeps nothing, not even the Stream Chat channel a text session is otherwise kept in.</summary>
    public bool Incognito { get; init; }

    /// <summary>Models for this conversation alone, over the agent's own.</summary>
    public ModelOverwrites? ModelOverwrites { get; init; }
}

/// <summary>How an agent is built.</summary>
public sealed record AgentOptions
{
    /// <summary>
    /// The agent to be: a directory holding agent.yaml, found the way <see cref="Folder.Find"/>
    /// finds one, or else the name of a config stored on the router.
    /// </summary>
    public string? Config { get; init; }

    /// <summary>
    /// A stored config by its id rather than its name, which is what a dispatched message
    /// carries as <see cref="InboundMessage.ConfigId"/>.
    /// </summary>
    public string? ConfigId { get; init; }

    /// <summary>What the agent is called. Defaults to the directory's or the config's name.</summary>
    public string? Name { get; init; }

    /// <summary>
    /// The system prompt. Wins over instructions.md. It reaches the backend with
    /// <see cref="Agent.SyncAsync"/>: a session runs on the stored config's.
    /// </summary>
    public string? Instructions { get; init; }

    /// <summary>The policy screening what may be asked of the agent. Wins over guardrail.md.</summary>
    public string? Guardrail { get; init; }

    /// <summary>The models the backend runs.</summary>
    public Pipeline? Pipeline { get; init; }

    /// <summary>
    /// Skills, subagents and the sandbox, written onto the stored config by
    /// <see cref="Agent.SyncAsync"/>. Null sends nothing, which leaves what is stored.
    /// </summary>
    public Harness? Harness { get; init; }

    /// <summary>Cost labels, carried onto every request the agent's sessions make.</summary>
    public IReadOnlyDictionary<string, string>? CostTracking { get; init; }

    /// <summary>
    /// Who the agent's memories are about, under <c>user_id</c>, and what else narrows recall.
    /// </summary>
    public IReadOnlyDictionary<string, string>? MemoryFilter { get; init; }

    /// <summary>Who the agent joins calls as. Defaults to one derived from its name.</summary>
    public string? UserId { get; init; }

    /// <summary>The caller's own functions, which the model is offered. Defaults to an empty set.</summary>
    public Tools? Tools { get; init; }

    /// <summary>The router. Defaults to one read from the environment.</summary>
    public VisionAgentsClient? Client { get; init; }

    /// <summary>What creates Stream calls. Defaults to one read from the environment when a call is first needed.</summary>
    public Edge? Edge { get; init; }
}

/// <summary>A call somebody placed to one of the customer's numbers, as dispatch hands it over.</summary>
public sealed record InboundCall
{
    /// <summary>The Stream call the caller is on.</summary>
    public required string CallId { get; init; }

    /// <summary>Its type.</summary>
    public string CallType { get; init; } = Edge.DefaultCallType;

    /// <summary>
    /// The session to open for the call, with voice: the call is named for it. Empty for a
    /// number attached before calls were, which has to be attached again.
    /// </summary>
    public string SessionId { get; init; } = "";

    /// <summary>The number that was rung.</summary>
    public string CalledNumber { get; init; } = "";

    /// <summary>Who rang, when the carrier says.</summary>
    public string CallerNumber { get; init; } = "";

    /// <summary>What the routing rule attached.</summary>
    public IReadOnlyDictionary<string, string> Custom { get; init; } = new Dictionary<string, string>();

    /// <summary>When it arrived.</summary>
    public DateTimeOffset? At { get; init; }
}

/// <summary>
/// A message written to an agent that is not running, or to a running session whose agent
/// leaves text to dispatch, as dispatch hands it over.
/// </summary>
public sealed record InboundMessage
{
    /// <summary>The channel it was written in, empty for a session that keeps no channel.</summary>
    public required string ChannelId { get; init; }

    /// <summary>The channel's type.</summary>
    public string ChannelType { get; init; } = "";

    /// <summary>Who to answer as, which names the channel replies are written into.</summary>
    public string AgentId { get; init; } = "";

    /// <summary>The config the message was routed to.</summary>
    public string ConfigId { get; init; } = "";

    /// <summary>
    /// The running session it was written to, set when its agent leaves text to dispatch.
    /// Nothing has answered it: <see cref="Dispatch.AnswerAsync"/> has the model do so.
    /// </summary>
    public string SessionId { get; init; } = "";

    /// <summary>The request it was sent as, which the answer lands on. May be empty.</summary>
    public string RequestId { get; init; } = "";

    /// <summary>What was written.</summary>
    public string Text { get; init; } = "";

    /// <summary>The message's id.</summary>
    public string MessageId { get; init; } = "";

    /// <summary>Who wrote it.</summary>
    public string UserId { get; init; } = "";

    /// <summary>What they are called.</summary>
    public string UserName { get; init; } = "";

    /// <summary>What the routing rule attached.</summary>
    public IReadOnlyDictionary<string, string> Custom { get; init; } = new Dictionary<string, string>();

    /// <summary>When it arrived.</summary>
    public DateTimeOffset? At { get; init; }
}

/// <summary>
/// An agent's knowledge base, as somewhere to put more of it.
/// </summary>
/// <remarks>
/// The namespace is the agent's own name, which is where a synced directory's knowledge
/// lands, so what is added here is found by the same lookup mid-answer.
/// </remarks>
public sealed class Knowledge
{
    // The router queues the read and retries one that fails, so a page can take a while to
    // settle. Past this it is returned still pending rather than waited on forever.
    internal static readonly TimeSpan ReadTimeout = TimeSpan.FromSeconds(180);
    internal static readonly TimeSpan PollInterval = TimeSpan.FromMilliseconds(250);

    private readonly VisionAgentsClient _client;

    internal Knowledge(VisionAgentsClient client, string @namespace)
    {
        _client = client;
        Namespace = @namespace;
    }

    /// <summary>The knowledge base this fills.</summary>
    public string Namespace { get; }

    /// <summary>
    /// Keeps the knowledge base filled from a page published elsewhere, and waits for it to be read.
    /// </summary>
    /// <remarks>
    /// It stays a subscription rather than a one-off: the passages are keyed by the url, and
    /// reading it again replaces them. What comes back says whether it worked, or is still
    /// pending if it was not read within three minutes. Adding a page again replaces it, so
    /// adding it without <paramref name="refreshHours"/> turns its schedule off.
    /// </remarks>
    /// <param name="url">The page.</param>
    /// <param name="title">What the page is, for a reader of the subscription.</param>
    /// <param name="description">What it holds.</param>
    /// <param name="refreshHours">How many hours between reads; null is never.</param>
    /// <param name="cancellationToken">Stops waiting.</param>
    public async Task<KnowledgeUrl> AddUrlAsync(string url, string? title = null, string? description = null, int? refreshHours = null, CancellationToken cancellationToken = default)
    {
        if (refreshHours is < 1)
        {
            throw new ConfigurationException("refresh_hours is how many hours between reads, so it is at least 1; leave it out for never");
        }
        var page = await _client.PostAsync<KnowledgeUrl>("/v1/agents/knowledge/urls", new KnowledgeUrlRequest
        {
            Namespace = Namespace,
            Url = url,
            Title = VisionAgentsClient.Blank(title),
            Description = VisionAgentsClient.Blank(description),
            RefreshHours = refreshHours,
        }, cancellationToken).ConfigureAwait(false);

        var deadline = DateTimeOffset.UtcNow + ReadTimeout;
        while (page.State == "pending" && DateTimeOffset.UtcNow < deadline)
        {
            await Task.Delay(PollInterval, cancellationToken).ConfigureAwait(false);
            page = await _client.GetAsync<KnowledgeUrl>(
                $"/v1/agents/knowledge/urls/{VisionAgentsClient.Escape(page.Id)}", cancellationToken: cancellationToken).ConfigureAwait(false);
        }
        return page;
    }
}

/// <summary>
/// An agent: what it is, and the conversations it holds.
/// </summary>
/// <remarks>
/// <para>
/// The model runs in the backend. What runs here is the agent's configuration, rendered
/// into each session it opens, and its <see cref="Tools"/>, which the model calls back into.
/// </para>
/// <para>
/// An agent built from a directory syncs it before its first session, so the session opens
/// under the stored config; <c>.agent_sync</c> makes a second sync of an unchanged
/// directory a read.
/// </para>
/// </remarks>
public sealed class Agent : IAsyncDisposable
{
    /// <summary>The key in <see cref="AgentOptions.MemoryFilter"/> naming who the memories are about.</summary>
    public const string UserKey = "user_id";

    private readonly AgentOptions _options;
    private readonly bool _ownsClient;
    private readonly string _config;
    private readonly SemaphoreSlim _syncing = new(1, 1);
    private Edge? _edge;
    private string _configId = "";
    private Session? _session;

    /// <summary>Validates an agent's configuration and reads its directory, if it has one.</summary>
    /// <exception cref="ConfigurationException">The configuration is not one an agent can run.</exception>
    public Agent(AgentOptions options)
    {
        if (options.Config is { Length: > 0 } config && Folder.Find(config) is { } path)
        {
            Folder = Folder.Load(path);
        }
        _config = Folder is null ? options.Config ?? "" : "";
        _configId = options.ConfigId ?? "";
        if (_configId != "" && (Folder is not null || _config != ""))
        {
            throw new ConfigurationException("name the config or its id, not both");
        }

        var harness = options.Harness;
        if (Folder is { Skills.Count: > 0 } && harness?.Skills is not { Count: > 0 })
        {
            harness = (harness ?? new Harness()) with { Skills = Folder.Skills };
        }
        harness?.Validate();

        Name = options.Name is { Length: > 0 } name ? name
            : Folder?.Name ?? (_config != "" ? _config : _configId != "" ? _configId
                : throw new ConfigurationException("an agent needs a name, a directory or a config"));
        _options = options with
        {
            Instructions = options.Instructions is { Length: > 0 } ? options.Instructions : Folder?.Instructions,
            Guardrail = options.Guardrail is { Length: > 0 } ? options.Guardrail : Folder?.Guardrail,
            Harness = harness,
        };
        UserId = options.UserId is { Length: > 0 } userId ? userId : UserIdOf(Name);
        Tools = options.Tools ?? new Tools();
        _ownsClient = options.Client is null;
        Client = options.Client ?? new VisionAgentsClient();
        _edge = options.Edge;
        Knowledge = new Knowledge(Client, Name);
    }

    /// <summary>What the agent is called.</summary>
    public string Name { get; }

    /// <summary>Who the agent joins calls as.</summary>
    public string UserId { get; }

    /// <summary>The system prompt it joins with.</summary>
    public string Instructions => _options.Instructions ?? "";

    /// <summary>The agent directory it was read from, or null.</summary>
    public Folder? Folder { get; }

    /// <summary>The caller's own functions, which the model is offered and this process runs.</summary>
    public Tools Tools { get; }

    /// <summary>The router this agent talks to.</summary>
    public VisionAgentsClient Client { get; }

    /// <summary>The agent's knowledge base.</summary>
    public Knowledge Knowledge { get; }

    /// <summary>The conversations held under the config of the agent's name: queried, updated and deleted.</summary>
    public Sessions Sessions => new(Client, Name);

    /// <summary>The conversation the agent opened last, or null before the first.</summary>
    public Session? Session => Volatile.Read(ref _session);

    /// <summary>The turns of the conversation the agent opened last.</summary>
    /// <exception cref="InvalidOperationException">The agent has not joined anything yet.</exception>
    public Responses Responses => Session?.Responses
        ?? throw new InvalidOperationException("the agent is in no conversation yet; join a call or start a chat first");

    /// <summary>What creates Stream calls, built from the environment on first use.</summary>
    public Edge Edge => _edge ??= new Edge();

    /// <summary>
    /// Stores the agent's configuration on the router, with its skills and knowledge, in one request.
    /// </summary>
    /// <remarks>
    /// What the code set wins over what the directory says, and is part of the fingerprint,
    /// so changing either syncs again. A directory unchanged since its last sync, according
    /// to <c>.agent_sync</c>, is only read back. The sessions the agent opens afterwards run
    /// under the stored config, which is where the harness is.
    /// </remarks>
    public async Task<AgentConfig> SyncAsync(CancellationToken cancellationToken = default)
    {
        var skills = _options.Harness?.Skills ?? [];
        var harness = _options.Harness?.Name ?? "";
        var subagent = _options.Harness?.Subagent ?? "";
        var sandbox = _options.Harness?.Sandbox?.Provider ?? "";
        var hash = Folder.Fingerprint(Folder?.Source ?? "", Instructions, _options.Guardrail ?? "",
            skills, Folder?.Knowledge ?? [], Folder?.KnowledgeUrls ?? [], Folder?.Simulations);
        if (harness != "" || subagent != "" || sandbox != "" || _options.CostTracking is { Count: > 0 })
        {
            hash = Folder.Fingerprint(hash, harness + subagent + sandbox, Folder.GoMap(_options.CostTracking), [], [], [], null);
        }

        if (Folder is not null && Folder.ReadStamp() == hash && await StoredAsync(cancellationToken).ConfigureAwait(false) is { } stored)
        {
            _configId = stored.Id;
            return stored;
        }

        var request = new SyncAgentRequest
        {
            Name = Name,
            Hash = hash,
            Instructions = VisionAgentsClient.Blank(Instructions),
            Guardrail = VisionAgentsClient.Blank(_options.Guardrail),
            Skills = skills.Count > 0 ? [.. skills.Select(SkillRequestOf)] : null,
        };
        if (Folder is not null)
        {
            Declare(request, Folder);
        }
        request.Harness = VisionAgentsClient.Blank(harness) ?? request.Harness;
        request.Subagent = VisionAgentsClient.Blank(subagent) ?? request.Subagent;
        request.Sandbox = VisionAgentsClient.Blank(sandbox) ?? request.Sandbox;
        if (_options.CostTracking is { Count: > 0 } costs)
        {
            request.Tags ??= [];
            foreach (var (key, value) in costs)
            {
                request.Tags[key] = value;
            }
        }

        var synced = await Client.PostAsync<SyncAgentResult>("/v1/agents/sync", request, cancellationToken).ConfigureAwait(false);
        Folder?.WriteStamp(hash);
        _configId = synced.Config.Id;
        return synced.Config;
    }

    /// <summary>
    /// Changes some of how the agent is configured and returns the config as it now is. A
    /// field left null keeps what is stored, so setting a guardrail leaves the instructions,
    /// skills and models alone. Server side only.
    /// </summary>
    /// <exception cref="ConfigurationException">Nothing is stored under the agent's name.</exception>
    public async Task<AgentConfig> UpdateConfigAsync(AgentConfigPatch patch, CancellationToken cancellationToken = default)
    {
        var id = _configId != "" ? _configId
            : (await StoredAsync(cancellationToken).ConfigureAwait(false))?.Id
                ?? throw new ConfigurationException($"there is no agent called {Name} to update");
        return await Client.PatchAsync<AgentConfig>($"/v1/agents/configs/{VisionAgentsClient.Escape(id)}", patch, cancellationToken)
            .ConfigureAwait(false);
    }

    /// <summary>
    /// Holds a conversation on the session's own call, <c>agent:&lt;session id&gt;</c>.
    /// </summary>
    /// <remarks>
    /// It returns once the backend is in the call. <see cref="Session.Voice"/> stops voice
    /// to carry the conversation on in writing, and starts it again.
    /// </remarks>
    public Task<Session> JoinAsync(SessionOptions? options = null, CancellationToken cancellationToken = default) =>
        OpenAsync(null, true, options, null, false, cancellationToken);

    /// <summary>
    /// Answers a call dispatch handed over, on the call the router routed the caller into,
    /// which is named for the session opened here.
    /// </summary>
    /// <exception cref="ConfigurationException">The call names no session.</exception>
    public Task<Session> JoinAsync(InboundCall call, SessionOptions? options = null, CancellationToken cancellationToken = default) =>
        call.SessionId == ""
            ? throw new ConfigurationException("the call names no session; attach its number again")
            : OpenAsync(call.SessionId, true, options,
                call.CalledNumber != "" ? new SessionPhone { Number = call.CalledNumber } : null, false, cancellationToken);

    /// <summary>
    /// Holds the conversation in writing rather than on a call. Nothing is transcribed or spoken
    /// until <see cref="Session.Voice"/> starts it; everything between hearing and answering is unchanged.
    /// </summary>
    public Task<Session> ChatAsync(SessionOptions? options = null, CancellationToken cancellationToken = default) =>
        OpenAsync(null, false, options, null, false, cancellationToken);

    /// <summary>
    /// Carries on a conversation by the id of the session it was held in, and watches it. One
    /// that ended is reopened with what was said in it.
    /// </summary>
    public async Task<Session> ResumeAsync(string id, CancellationToken cancellationToken = default)
    {
        var got = await Client.GetAsync<Models.Session>(
            $"/v1/agents/sessions/{VisionAgentsClient.Escape(id)}", cancellationToken: cancellationToken).ConfigureAwait(false);
        var session = await Session.OpenAsync(Client, got, Tools, UserId, new Session.Watch(), cancellationToken).ConfigureAwait(false);
        Volatile.Write(ref _session, session);
        return session;
    }

    /// <summary>
    /// Rings somebody from one of the customer's numbers and holds the conversation when they answer.
    /// </summary>
    /// <remarks>
    /// The agent placed this call, so it is told it is navigating: recordings are let finish
    /// and menus are answered rather than talked over.
    /// </remarks>
    public async Task<Session> OutboundCallAsync(string from, string to, SessionOptions? options = null, CancellationToken cancellationToken = default)
    {
        if (string.IsNullOrEmpty(from) || string.IsNullOrEmpty(to))
        {
            throw new ConfigurationException("a call needs a number to ring from and one to ring");
        }
        // Placing the call makes its own routing rule, pinned to the call of the session it
        // names, so the answered leg arrives in the call this agent is about to join.
        var placed = await Client.PostAsync<PlacedCall>("/v1/phone/calls", new PlaceCallRequest
        {
            From = from,
            To = to,
            Tags = _options.CostTracking is { Count: > 0 } costs ? new Dictionary<string, string>(costs) : null,
        }, cancellationToken).ConfigureAwait(false);
        if (placed.SessionId is not { Length: > 0 } sessionId)
        {
            throw new RouterException(0, "POST /v1/phone/calls", "the router placed the call for no session");
        }
        return await OpenAsync(sessionId, true, options, new SessionPhone { Number = from, VendorCallId = placed.VendorCallId }, true, cancellationToken)
            .ConfigureAwait(false);
    }

    /// <summary>
    /// Answers the next call to one of the customer's numbers.
    /// </summary>
    /// <remarks>
    /// The number is attached, so every caller lands in a call of their own, and this waits
    /// for the router to hand the next one over, joins it, and waits until the caller says
    /// something. What they said is still on the session's events. A worker answering many
    /// calls uses <see cref="Dispatch.WaitForCall"/> and <see cref="JoinAsync(InboundCall, SessionOptions?, CancellationToken)"/> instead.
    /// </remarks>
    public async Task<Session> WaitForCallAsync(string number, SessionOptions? options = null, CancellationToken cancellationToken = default)
    {
        if (string.IsNullOrEmpty(number))
        {
            throw new ConfigurationException("there is no number to answer on");
        }
        await Client.PostAsync<PhoneNumber>($"/v1/phone/numbers/{VisionAgentsClient.Escape(number)}/attach",
            new AttachNumberRequest(), cancellationToken).ConfigureAwait(false);

        var arrived = new TaskCompletionSource<InboundCall>(TaskCreationOptions.RunContinuationsAsynchronously);
        var dispatch = new Dispatch(new DispatchOptions { Client = Client, Capacity = 1 }).WaitForCall(call =>
        {
            if (call.CalledNumber != number)
            {
                throw new InvalidOperationException($"this agent is waiting on {number}, not {call.CalledNumber}");
            }
            if (!arrived.TrySetResult(call))
            {
                throw new InvalidOperationException("this agent is already answering a call");
            }
            return Task.CompletedTask;
        });
        using var waiting = CancellationTokenSource.CreateLinkedTokenSource(cancellationToken);
        var running = dispatch.RunAsync(waiting.Token);
        if (await Task.WhenAny(arrived.Task, running).ConfigureAwait(false) != arrived.Task)
        {
            await running.ConfigureAwait(false);
            cancellationToken.ThrowIfCancellationRequested();
            throw new InvalidOperationException("stopped waiting before anybody rang");
        }
        await waiting.CancelAsync().ConfigureAwait(false);
        await running.ConfigureAwait(false);

        var session = await JoinAsync(await arrived.Task.ConfigureAwait(false), options, cancellationToken).ConfigureAwait(false);
        try
        {
            await session.HeardAsync(cancellationToken).ConfigureAwait(false);
        }
        catch
        {
            await session.CloseAsync(CancellationToken.None).ConfigureAwait(false);
            throw;
        }
        return session;
    }

    /// <summary>A link a person can open to join the session's call from a browser and hear the agent.</summary>
    public string MonitorUrl(Session session) =>
        !session.Voice.Started
            ? throw new ConfigurationException("a conversation held in writing has no call to watch")
            : Edge.MonitorUrl(session.Call, "monitor-" + session.Id, "Monitor");

    /// <summary>Closes the conversation the agent opened last, and the client if the agent made it.</summary>
    public async ValueTask DisposeAsync()
    {
        if (Session is { } session)
        {
            await session.DisposeAsync().ConfigureAwait(false);
        }
        if (_ownsClient)
        {
            Client.Dispose();
        }
        _syncing.Dispose();
    }

    /// <summary>The session request, rendered from the agent and the conversation.</summary>
    internal CreateSessionRequest Request(string? sessionId, bool voice, SessionOptions? options, SessionPhone? phone, bool navigating)
    {
        var pipeline = _options.Pipeline ?? new Pipeline();
        options ??= new SessionOptions();
        return new CreateSessionRequest
        {
            Id = VisionAgentsClient.Blank(sessionId) ?? VisionAgentsClient.Blank(options.Id),
            StartVoice = voice ? true : null,
            Agent = _configId == "" ? VisionAgentsClient.Blank(_config) : null,
            ConfigId = VisionAgentsClient.Blank(_configId),
            UserId = UserId,
            UserName = Name,
            AgentId = VisionAgentsClient.Blank(options.AgentId) ?? UserId,
            Incognito = options.Incognito ? true : null,
            Title = VisionAgentsClient.Blank(options.Title),
            Description = VisionAgentsClient.Blank(options.Description),
            ProjectId = VisionAgentsClient.Blank(options.ProjectId),
            Custom = options.Custom is { Count: > 0 } custom ? custom.ToDictionary(pair => pair.Key, pair => pair.Value!) : null,
            ModelOverwrites = options.ModelOverwrites,
            Llm = VisionAgentsClient.Blank(pipeline.Llm),
            Stt = VisionAgentsClient.Blank(pipeline.Stt),
            Tts = VisionAgentsClient.Blank(pipeline.Tts),
            Voice = VisionAgentsClient.Blank(pipeline.Voice),
            Greeting = pipeline.Greeting is { Length: > 0 } greeting
                ? new Greeting { Text = greeting, Mode = VisionAgentsClient.Blank(pipeline.GreetingMode) }
                : null,
            Languages = pipeline.Language is { Length: > 0 } language ? [language] : null,
            Backchannel = pipeline.Backchannel,
            MaxTokens = pipeline.MaxTokens is > 0 ? pipeline.MaxTokens : null,
            ToolTimeoutMs = pipeline.ToolTimeout is { } timeout && timeout > TimeSpan.Zero ? (int)timeout.TotalMilliseconds : null,
            Video = pipeline.Video,
            Tags = _options.CostTracking is { Count: > 0 } costs ? new Dictionary<string, string>(costs) : null,
            Memory = MemoryOf(_options.MemoryFilter),
            Phone = phone,
            Navigating = navigating ? true : null,
            Tools = Tools.Count > 0 ? Tools.Declared() : null,
        };
    }

    /// <summary>Turns a name into something a call can be joined under.</summary>
    internal static string UserIdOf(string name)
    {
        var id = new string(name.Select(letter => letter switch
        {
            >= 'a' and <= 'z' or >= '0' and <= '9' or '-' or '_' => letter,
            >= 'A' and <= 'Z' => char.ToLowerInvariant(letter),
            _ => '-',
        }).ToArray()).Trim('-');
        return id == "" ? "vision-agent" : id;
    }

    /// <summary>Splits the filter into who the memories are about and what narrows them.</summary>
    internal static SessionMemory? MemoryOf(IReadOnlyDictionary<string, string>? filter)
    {
        if (filter is not { Count: > 0 })
        {
            return null;
        }
        var narrowing = filter.Where(pair => pair.Key != UserKey).ToDictionary(pair => pair.Key, pair => pair.Value);
        return new SessionMemory
        {
            UserId = filter.TryGetValue(UserKey, out var user) ? user : null,
            Filter = narrowing.Count > 0 ? narrowing : null,
        };
    }

    private async Task<Session> OpenAsync(string? sessionId, bool voice, SessionOptions? options, SessionPhone? phone, bool navigating, CancellationToken cancellationToken)
    {
        if (Folder is not null && _configId == "")
        {
            await _syncing.WaitAsync(cancellationToken).ConfigureAwait(false);
            try
            {
                if (_configId == "")
                {
                    await SyncAsync(cancellationToken).ConfigureAwait(false);
                }
            }
            finally
            {
                _syncing.Release();
            }
        }
        var created = await Client.PostAsync<Models.Session>("/v1/agents/sessions",
            Request(sessionId, voice, options, phone, navigating), cancellationToken).ConfigureAwait(false);
        var session = await Session.OpenAsync(Client, created, Tools, UserId, new Session.Watch(), cancellationToken).ConfigureAwait(false);
        Volatile.Write(ref _session, session);
        return session;
    }

    /// <summary>The config stored under the agent's name, or null when there is none.</summary>
    private async Task<AgentConfig?> StoredAsync(CancellationToken cancellationToken)
    {
        var listed = await Client.GetAsync<List<AgentConfig>>("/v1/agents/configs",
            new Dictionary<string, string?> { ["name"] = Name }, cancellationToken).ConfigureAwait(false);
        return listed.Find(config => config.Name == Name);
    }

    private static void Declare(SyncAgentRequest request, Folder folder)
    {
        var declared = folder.Declaration;
        request.Knowledge = folder.Knowledge.Count > 0
            ? [.. folder.Knowledge.Select(document => new KnowledgeDocument { Source = document.Source, Text = document.Text })]
            : null;
        request.KnowledgeUrls = folder.KnowledgeUrls.Count > 0
            ? [.. folder.KnowledgeUrls.Select(page => new KnowledgeUrlDeclaration
            {
                Url = page.Url,
                Title = VisionAgentsClient.Blank(page.Title),
                Description = VisionAgentsClient.Blank(page.Description),
                RefreshHours = page.RefreshHours > 0 ? page.RefreshHours : null,
            })]
            : null;
        // Sent only when there is a simulations/, and empty when it holds none: what is sent
        // is the whole of the agent's simulations, and leaving it out leaves the stored ones.
        request.Simulations = folder.Simulations is { } simulations ? [.. simulations] : null;
        // Only what the file names is sent, so the router leaves whatever is stored for the rest.
        request.Mode = VisionAgentsClient.Blank(declared.Mode);
        request.Stt = VisionAgentsClient.Blank(declared.Stt);
        request.Tts = VisionAgentsClient.Blank(declared.Tts);
        request.Sts = declared.Sts;
        request.Voice = VisionAgentsClient.Blank(declared.Voice);
        request.Llm = VisionAgentsClient.Blank(declared.Llm);
        request.Harness = VisionAgentsClient.Blank(declared.Harness);
        request.Subagent = VisionAgentsClient.Blank(declared.Subagent);
        request.Search = VisionAgentsClient.Blank(declared.Search);
        request.Greeting = declared.Greeting is { } greeting
            ? new Greeting { Text = greeting.Text, Mode = VisionAgentsClient.Blank(greeting.Mode) }
            : null;
        request.Sandbox = VisionAgentsClient.Blank(declared.Sandbox);
        // Deprecated for connectors, but still what the router reads until plugins are removed.
#pragma warning disable CS0612
        request.Plugins = declared.Plugins.Count > 0 ? [.. declared.Plugins] : null;
#pragma warning restore CS0612
        request.Keyterms = declared.Keyterms.Count > 0 ? [.. declared.Keyterms] : null;
        request.Tags = declared.Tags.Count > 0 ? new Dictionary<string, string>(declared.Tags) : null;
        request.Video = declared.Video is { } video
            ? new SessionVideo { Source = VisionAgentsClient.Blank(video.Source), MaxFrames = video.MaxFrames }
            : null;
        request.Dispatch = declared.Dispatch is { } dispatch
            ? new AgentDispatch { IncomingCall = VisionAgentsClient.Blank(dispatch.IncomingCall), Text = VisionAgentsClient.Blank(dispatch.Text) }
            : null;
    }

    private static SkillRequest SkillRequestOf(Skill skill) => new()
    {
        ConfigId = "",
        Name = skill.Name,
        Description = skill.Description,
        Instructions = skill.Instructions,
        CaptureVideo = skill.CaptureVideo,
        DeadlineMs = skill.Deadline is { } deadline && deadline > TimeSpan.Zero ? (long)deadline.TotalMilliseconds : null,
    };
}
