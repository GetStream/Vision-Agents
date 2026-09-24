using System.Collections.Concurrent;
using System.Runtime.CompilerServices;
using System.Text.Json.Nodes;
using System.Threading.Channels;
using GetStream.VisionAgents.Models;

namespace GetStream.VisionAgents;

/// <summary>Somebody on the call.</summary>
public sealed record Participant(string Id, string UserId, string Name);

/// <summary>
/// One thing that happened in a session, as the router said it.
/// </summary>
/// <remarks>
/// <see cref="Kind"/> is the frame's <c>type</c>: joined, participant_joined,
/// participant_left, hearing, heard, decision, responding, response_delta, responded, error
/// or left. The fields are read off whichever frame it was, and <see cref="Frame"/> is the
/// frame whole, for anything this SDK does not name.
/// </remarks>
public sealed record SessionEvent
{
    internal SessionEvent(Frame frame)
    {
        Frame = frame;
        Kind = frame.Type;
        Text = frame.Text("text");
        TurnId = frame.Text("turn_id");
        Language = frame.Text("language");
        PendingWork = frame.Flag("pending_work");
        Error = frame.Text("error") is { Length: > 0 } error ? error : frame.Text("message");
        if (frame.Nested("participant") is { } participant)
        {
            Participant = new Participant(participant.Text("id"), participant.Text("user_id"), participant.Text("name"));
        }
    }

    /// <summary>What happened.</summary>
    public string Kind { get; }

    /// <summary>What was heard or said, for the events that carry text.</summary>
    public string Text { get; }

    /// <summary>The turn the event belongs to.</summary>
    public string TurnId { get; }

    /// <summary>The language something was heard in, when the transcriber says.</summary>
    public string Language { get; }

    /// <summary>On responded: the answer promised work still running, whose result follows as another turn.</summary>
    public bool PendingWork { get; }

    /// <summary>On error: what went wrong.</summary>
    public string Error { get; }

    /// <summary>Who it was about, for the events that name somebody.</summary>
    public Participant? Participant { get; }

    /// <summary>The frame as it arrived.</summary>
    public Frame Frame { get; }
}

/// <summary>
/// One conversation: a call joined or a chat held.
/// </summary>
/// <remarks>
/// <para>
/// A session opened by an <see cref="Agent"/> is watched: a socket is held open that runs
/// the agent's tools when the model calls them and carries everything else to
/// <see cref="EventsAsync"/>. One read from the router's list is a record only; it can be
/// asked for responses, forked or closed, but it has no events.
/// </para>
/// <para>
/// Events are buffered, the most recent 256 of them, so a caller that is not reading does
/// not hold up tools; when nobody reads, the oldest are dropped. There is one reader.
/// </para>
/// </remarks>
public sealed class Session : IAsyncDisposable
{
    // The model can ask for many things at once; more than this in flight is a runaway.
    private const int MaxRunningTools = 16;

    private readonly VisionAgentsClient _client;
    private readonly Tools? _tools;
    private readonly string _agentUserId;
    private readonly Socket? _socket;
    private readonly Channel<SessionEvent> _events = Channel.CreateBounded<SessionEvent>(
        new BoundedChannelOptions(256) { FullMode = BoundedChannelFullMode.DropOldest, SingleReader = true });
    private readonly ConcurrentDictionary<string, CancellationTokenSource> _running = new();
    private readonly CancellationTokenSource _stop = new();
    private readonly TaskCompletionSource _ended = new(TaskCreationOptions.RunContinuationsAsynchronously);
    private readonly TaskCompletionSource<Participant> _joined = new(TaskCreationOptions.RunContinuationsAsynchronously);
    private readonly TaskCompletionSource<SessionEvent> _heard = new(TaskCreationOptions.RunContinuationsAsynchronously);
    private readonly Task _watching;
    private int _reading;
    private int _closed;

    private Session(VisionAgentsClient client, Models.Session created, Tools? tools, string agentUserId, Socket? socket)
    {
        _client = client;
        _tools = tools;
        _agentUserId = agentUserId;
        _socket = socket;
        Created = created;
        Responses = new Responses(client, created.Id);
        if (socket is null)
        {
            _ended.TrySetResult();
            _events.Writer.TryComplete();
            _watching = Task.CompletedTask;
        }
        else
        {
            _watching = Task.Run(WatchAsync);
        }
    }

    /// <summary>What the router said when it opened the session.</summary>
    public Models.Session Created { get; }

    /// <summary>The router's id for the session.</summary>
    public string Id => Created.Id;

    /// <summary>The call joined, empty for a chat.</summary>
    public string CallId => Created.CallId;

    /// <summary>The call's type.</summary>
    public string CallType => Created.CallType;

    /// <summary>The Stream Chat channel a persisted conversation is kept in.</summary>
    public string ConversationId => Created.ConversationId ?? "";

    /// <summary>Whether the session's socket is still open.</summary>
    public bool Live => _socket is { Open: true };

    /// <summary>The session's turns.</summary>
    public Responses Responses { get; }

    /// <summary>
    /// What happens in the session, until it ends.
    /// </summary>
    /// <remarks>Tool calls are not here: they are answered before anything else sees them.</remarks>
    public async IAsyncEnumerable<SessionEvent> EventsAsync([EnumeratorCancellation] CancellationToken cancellationToken = default)
    {
        if (_socket is null)
        {
            throw new ConfigurationException("this session was read, not opened, so nothing is watching it; open one through an agent");
        }
        if (Interlocked.Exchange(ref _reading, 1) == 1)
        {
            throw new InvalidOperationException("a session's events have one reader");
        }
        try
        {
            await foreach (var happened in _events.Reader.ReadAllAsync(cancellationToken).ConfigureAwait(false))
            {
                yield return happened;
            }
        }
        finally
        {
            Volatile.Write(ref _reading, 0);
        }
    }

    /// <summary>Says something as it is, without going through the model.</summary>
    public Task SayAsync(string text, CancellationToken cancellationToken = default) =>
        SendAsync(Frames.Of("say", ("text", text)), cancellationToken);

    /// <summary>Hands the model a turn, as though somebody had said it.</summary>
    public Task RespondAsync(string text, IReadOnlyList<ImageSource>? images = null, CancellationToken cancellationToken = default) =>
        SendAsync(Frames.Of("respond", ("text", text), ("images", images is { Count: > 0 } ? images : null)), cancellationToken);

    /// <summary>Stops whatever the agent is saying.</summary>
    public Task InterruptAsync(CancellationToken cancellationToken = default) =>
        SendAsync(Frames.Of("interrupt"), cancellationToken);

    /// <summary>Replaces the system prompt from the next turn on.</summary>
    public Task SetInstructionsAsync(string instructions, CancellationToken cancellationToken = default) =>
        SendAsync(Frames.Of("instructions", ("instructions", instructions)), cancellationToken);

    /// <summary>
    /// Carries the conversation on in a new session, leaving this one as it was.
    /// </summary>
    /// <remarks>
    /// The fork is watched the way this one is, with the same tools. Name a
    /// <see cref="ForkOptions.ResponseId"/> to branch from a response rather than from the end.
    /// </remarks>
    public async Task<Session> ForkAsync(ForkOptions? options = null, CancellationToken cancellationToken = default)
    {
        options ??= new ForkOptions();
        var request = new ForkSessionRequest
        {
            Agent = VisionAgentsClient.Blank(options.Agent),
            Title = VisionAgentsClient.Blank(options.Title),
            Description = VisionAgentsClient.Blank(options.Description),
            Project = VisionAgentsClient.Blank(options.Project),
            Custom = options.Custom,
            ModelOverwrites = options.ModelOverwrites,
            Instructions = VisionAgentsClient.Blank(options.Instructions),
            Incognito = options.Incognito,
            Messages = options.Messages,
            ResponseId = VisionAgentsClient.Blank(options.ResponseId),
            CallId = VisionAgentsClient.Blank(options.CallId),
        };
        var forked = await _client.PostAsync<Models.Session>(
            $"/v1/agents/sessions/{VisionAgentsClient.Escape(Id)}/fork", request, cancellationToken).ConfigureAwait(false);
        return _socket is null
            ? Read(_client, forked)
            : await OpenAsync(_client, forked, _tools, _agentUserId, new Watch(), cancellationToken).ConfigureAwait(false);
    }

    /// <summary>Waits for the session to end.</summary>
    public Task WaitAsync(CancellationToken cancellationToken = default) => _ended.Task.WaitAsync(cancellationToken);

    /// <summary>
    /// Ends the session: the agent leaves the call and its socket closes. Safe to call twice.
    /// </summary>
    /// <remarks>
    /// Over the socket when it is open, since that is what the router is listening to;
    /// otherwise by deleting the session, which is how a read session is ended.
    /// </remarks>
    public async Task CloseAsync(CancellationToken cancellationToken = default)
    {
        if (Interlocked.Exchange(ref _closed, 1) == 1)
        {
            await _watching.ConfigureAwait(false);
            return;
        }
        if (_socket is { Open: true })
        {
            try
            {
                await _socket.SendAsync(Frames.Of("close"), cancellationToken).ConfigureAwait(false);
            }
            catch (SocketClosedException)
            {
                // It closed on its own first, which is the same outcome.
            }
            await _socket.CloseAsync().ConfigureAwait(false);
        }
        else
        {
            try
            {
                await _client.DeleteAsync($"/v1/agents/sessions/{VisionAgentsClient.Escape(Id)}", cancellationToken).ConfigureAwait(false);
            }
            catch (RouterException failure) when (failure.Status == 404)
            {
                // Already gone.
            }
        }
        await _watching.ConfigureAwait(false);
    }

    /// <inheritdoc />
    public async ValueTask DisposeAsync()
    {
        await CloseAsync().ConfigureAwait(false);
        if (_socket is not null)
        {
            await _socket.DisposeAsync().ConfigureAwait(false);
        }
        _stop.Dispose();
    }

    /// <summary>A session read from the router, which nothing watches.</summary>
    internal static Session Read(VisionAgentsClient client, Models.Session created) => new(client, created, null, "", null);

    /// <summary>
    /// Opens a created session's socket and starts watching it. A socket that cannot be
    /// opened deletes the session, which would otherwise hold a call with nobody running
    /// its tools.
    /// </summary>
    internal static async Task<Session> OpenAsync(
        VisionAgentsClient client,
        Models.Session created,
        Tools? tools,
        string agentUserId,
        Watch watch,
        CancellationToken cancellationToken)
    {
        var query = new Dictionary<string, string> { ["decisions"] = watch.Decisions ? "true" : "false" };
        if (watch.Interim)
        {
            query["interim"] = "true";
        }
        Socket socket;
        try
        {
            socket = await Socket.ConnectAsync(client.Backend,
                client.Backend.SocketUrl($"/v1/agents/sessions/{VisionAgentsClient.Escape(created.Id)}/events", query),
                cancellationToken).ConfigureAwait(false);
        }
        catch
        {
            try
            {
                await client.DeleteAsync($"/v1/agents/sessions/{VisionAgentsClient.Escape(created.Id)}", CancellationToken.None).ConfigureAwait(false);
            }
            catch (RouterException)
            {
                // The socket's failure is the one worth reporting.
            }
            throw;
        }
        return new Session(client, created, tools, agentUserId, socket);
    }

    /// <summary>
    /// Waits for somebody other than the agent to be on the call: on an answered phone call,
    /// the caller. Nothing is taken off <see cref="EventsAsync"/> to find out.
    /// </summary>
    /// <exception cref="SocketClosedException">The session ended first.</exception>
    public Task<Participant> WaitForParticipantAsync(CancellationToken cancellationToken = default) =>
        _socket is null
            ? throw new ConfigurationException("this session was read, not opened, so nothing is watching it")
            : _joined.Task.WaitAsync(cancellationToken);

    /// <summary>The first thing heard.</summary>
    internal Task<SessionEvent> HeardAsync(CancellationToken cancellationToken) => _heard.Task.WaitAsync(cancellationToken);

    private Task SendAsync(JsonObject frame, CancellationToken cancellationToken)
    {
        if (_socket is null)
        {
            throw new SocketClosedException("this session was read, not opened, so there is no socket to say anything over");
        }
        return _socket.SendAsync(frame, cancellationToken);
    }

    private async Task WatchAsync()
    {
        try
        {
            while (await _socket!.ReceiveAsync(_stop.Token).ConfigureAwait(false) is { } message)
            {
                if (message.Frame is not { } frame)
                {
                    continue;
                }
                switch (frame.Type)
                {
                    case "tool_call":
                        Call(frame);
                        continue;
                    case "tool_cancel":
                        if (_running.TryRemove(frame.Text("id"), out var cancelled))
                        {
                            await cancelled.CancelAsync().ConfigureAwait(false);
                        }
                        continue;
                }

                var happened = new SessionEvent(frame);
                if (happened is { Kind: "participant_joined", Participant: { } participant } && participant.UserId != _agentUserId)
                {
                    _joined.TrySetResult(participant);
                }
                else if (happened.Kind == "heard")
                {
                    _heard.TrySetResult(happened);
                }
                _events.Writer.TryWrite(happened);
            }
        }
        catch (OperationCanceledException)
        {
            // Stopped, which is how a close ends the loop.
        }
        finally
        {
            foreach (var running in _running.Values)
            {
                await running.CancelAsync().ConfigureAwait(false);
            }
            var ended = new SocketClosedException("the session ended first");
            _joined.TrySetException(ended);
            _heard.TrySetException(ended);
            _ = _joined.Task.Exception;
            _ = _heard.Task.Exception;
            _events.Writer.TryComplete();
            _ended.TrySetResult();
        }
    }

    /// <summary>
    /// Runs a tool the model called, off the read loop so a slow one does not hold up the
    /// conversation, and answers with what it returned or what it threw.
    /// </summary>
    private void Call(Frame frame)
    {
        var id = frame.Text("id");
        var answer = (string Key, object? Value) => Frames.Of("tool_result",
            ("tool_call_id", id),
            ("command_id", frame.Text("command_id")),
            ("turn_id", frame.Text("turn_id")),
            (Key, Value));
        if (_tools is null || _running.Count >= MaxRunningTools)
        {
            _ = TrySendAsync(answer("error", _tools is null ? "nothing here runs tools" : "too many tools are already running"));
            return;
        }

        var cancel = CancellationTokenSource.CreateLinkedTokenSource(_stop.Token);
        _running[id] = cancel;
        _ = Task.Run(async () =>
        {
            JsonObject result;
            try
            {
                var arguments = frame.Json["arguments"] switch
                {
                    JsonValue value when value.TryGetValue(out string? text) => text,
                    null => "",
                    var other => other.ToJsonString(),
                };
                result = answer("output", await _tools.CallAsync(frame.Text("name"), arguments, cancel.Token).ConfigureAwait(false));
            }
            catch (OperationCanceledException) when (cancel.IsCancellationRequested)
            {
                // Cancelled by the router, which is no longer waiting for an answer.
                return;
            }
            catch (Exception failure)
            {
                // Whatever a tool throws is the model's to hear about, not this loop's.
                result = answer("error", failure.Message);
            }
            finally
            {
                _running.TryRemove(id, out _);
                cancel.Dispose();
            }
            await TrySendAsync(result).ConfigureAwait(false);
        });
    }

    private async Task TrySendAsync(JsonObject frame)
    {
        try
        {
            await _socket!.SendAsync(frame).ConfigureAwait(false);
        }
        catch (SocketClosedException)
        {
            // The session ended while the tool ran; there is nobody to answer.
        }
    }

    /// <summary>What a watched session's socket asks for.</summary>
    internal sealed record Watch(bool Interim = false, bool Decisions = false);
}

/// <summary>How to carry a conversation on in a new session.</summary>
public sealed record ForkOptions
{
    /// <summary>The response to branch from, rather than from the end.</summary>
    public string? ResponseId { get; init; }

    /// <summary>A different agent config to carry on under.</summary>
    public string? Agent { get; init; }

    /// <summary>A call to carry the fork on in.</summary>
    public string? CallId { get; init; }

    /// <summary>The fork's title.</summary>
    public string? Title { get; init; }

    /// <summary>What the fork is about.</summary>
    public string? Description { get; init; }

    /// <summary>The project the fork is filed under.</summary>
    public string? Project { get; init; }

    /// <summary>Anything of the caller's own.</summary>
    public Dictionary<string, object?>? Custom { get; init; }

    /// <summary>Models to use instead of the session's.</summary>
    public ModelOverwrites? ModelOverwrites { get; init; }

    /// <summary>A different system prompt.</summary>
    public string? Instructions { get; init; }

    /// <summary>Records nothing.</summary>
    public bool? Incognito { get; init; }

    /// <summary>False starts the same configuration over, without the conversation so far.</summary>
    public bool? Messages { get; init; }
}
