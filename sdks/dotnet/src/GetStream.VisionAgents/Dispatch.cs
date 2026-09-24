using System.Collections.Concurrent;
using System.Diagnostics;
using System.Globalization;
using System.Text.Json.Nodes;

namespace GetStream.VisionAgents;

/// <summary>How a worker waits for work.</summary>
public sealed record DispatchOptions
{
    /// <summary>The router. Defaults to one read from the environment.</summary>
    public VisionAgentsClient? Client { get; init; }

    /// <summary>
    /// How many calls to hold at once. The router passes over a worker that is full rather
    /// than queueing behind it, so this is a promise about what this process can answer.
    /// </summary>
    public int Capacity { get; init; } = 4;

    /// <summary>How often to tell the router how this process is doing.</summary>
    public TimeSpan ReportEvery { get; init; } = TimeSpan.FromSeconds(15);
}

/// <summary>
/// Waits for inbound calls and messages, and runs a handler for each one.
/// </summary>
/// <remarks>
/// <para>
/// Neither arrives here first: a caller reached a Stream call over SIP, or somebody wrote
/// in a channel, and the router found out by webhook. The agent runs in this process, so
/// this connects out and waits, and the router pushes work down the connection. Nothing has
/// to be publicly reachable.
/// </para>
/// <para>
/// A message only arrives when no agent is running on its channel; one written to an agent
/// that is running is answered by that session. Several workers can wait at once, and the
/// work is shared between them.
/// </para>
/// </remarks>
public sealed class Dispatch
{
    private const string Path = "/v1/dispatch";

    private readonly TimeSpan _reportEvery;
    private readonly ConcurrentDictionary<long, Task> _running = new();
    // A channel is one conversation, so the agent that answered the last message on it is
    // the one that knows what has been said and should answer the next.
    private readonly Dictionary<string, Agent> _agents = [];
    private readonly SemaphoreSlim _agentsLock = new(1, 1);
    private readonly Stopwatch _clock = Stopwatch.StartNew();
    private Func<InboundCall, Task>? _onCall;
    private Func<InboundMessage, Task>? _onMessage;
    private Socket? _socket;
    private TaskCompletionSource _pong = new(TaskCreationOptions.RunContinuationsAsynchronously);
    private double _latencyMs;
    private long _handed;

    /// <summary>Waits for one customer's work on a router.</summary>
    /// <exception cref="ConfigurationException">The capacity is not a number of calls.</exception>
    public Dispatch(DispatchOptions? options = null)
    {
        options ??= new DispatchOptions();
        if (options.Capacity < 1)
        {
            throw new ConfigurationException("a worker that can hold no calls cannot answer any");
        }
        Capacity = options.Capacity;
        _reportEvery = options.ReportEvery;
        Client = options.Client ?? new VisionAgentsClient();
    }

    /// <summary>The router this waits on.</summary>
    public VisionAgentsClient Client { get; }

    /// <summary>How many calls this holds at once.</summary>
    public int Capacity { get; }

    /// <summary>What the router calls this connection, for matching a log line here against one there.</summary>
    public string WorkerId { get; private set; } = "";

    /// <summary>How much is being handled right now.</summary>
    public int Active => _running.Count;

    /// <summary>
    /// What to do with an arriving call. It runs as its own task, so one long call does not
    /// stop the next from being answered. The router is told the call was accepted when it
    /// returns, and rejected, with the reason, when it throws.
    /// </summary>
    public Dispatch WaitForCall(Func<InboundCall, Task> handler)
    {
        _onCall = handler;
        return this;
    }

    /// <summary>
    /// What to do with a message written to an agent that is not running. It runs as its
    /// own task, the way a call's does.
    /// </summary>
    /// <example>
    /// <code>
    /// dispatch.WaitForMessage(async message =>
    /// {
    ///     var agent = await dispatch.GetOrCreateAgentAsync(message, () => new Agent(new() { Config = "support" }));
    ///     await agent.Responses.CreateAsync(message.Text);
    /// });
    /// </code>
    /// Nothing is waited for: the answer is written into the channel as it is generated.
    /// </example>
    public Dispatch WaitForMessage(Func<InboundMessage, Task> handler)
    {
        _onMessage = handler;
        return this;
    }

    /// <summary>
    /// The agent answering on this message's channel, started in writing if none is.
    /// </summary>
    /// <remarks>
    /// The second message on a channel goes to the agent that answered the first, which is
    /// still open and knows what has been said; only a channel nothing is answering calls
    /// <paramref name="createAgent"/>. Agents are kept until this worker stops waiting.
    /// </remarks>
    public Task<Agent> GetOrCreateAgentAsync(InboundMessage message, Func<Agent> createAgent, CancellationToken cancellationToken = default) =>
        GetOrCreateAgentAsync(message, _ => Task.FromResult(createAgent()), cancellationToken);

    /// <inheritdoc cref="GetOrCreateAgentAsync(InboundMessage, Func{Agent}, CancellationToken)"/>
    public async Task<Agent> GetOrCreateAgentAsync(InboundMessage message, Func<CancellationToken, Task<Agent>> createAgent, CancellationToken cancellationToken = default)
    {
        await _agentsLock.WaitAsync(cancellationToken).ConfigureAwait(false);
        try
        {
            if (_agents.TryGetValue(message.ChannelId, out var answering) && answering.Session is { Live: true })
            {
                return answering;
            }
            var agent = await createAgent(cancellationToken).ConfigureAwait(false);
            await agent.ChatAsync(new SessionOptions { Persist = true, AgentId = message.AgentId }, cancellationToken).ConfigureAwait(false);
            _agents[message.ChannelId] = agent;
            return agent;
        }
        finally
        {
            _agentsLock.Release();
        }
    }

    /// <summary>
    /// Waits for calls and messages until cancelled or until the router closes the connection.
    /// </summary>
    /// <remarks>
    /// Returns rather than throws when cancelled. Work still being handled is waited for
    /// either way, because dropping a call would hang up on whoever is talking.
    /// </remarks>
    /// <exception cref="InvalidOperationException">No handler is registered, so work would arrive with nothing to do it.</exception>
    public async Task RunAsync(CancellationToken cancellationToken = default)
    {
        if (_onCall is null && _onMessage is null)
        {
            throw new InvalidOperationException("register a handler with WaitForCall or WaitForMessage before running");
        }

        Socket socket;
        try
        {
            socket = await Socket.ConnectAsync(Client.Backend,
                Client.Backend.SocketUrl(Path, new Dictionary<string, string> { ["capacity"] = Capacity.ToString(CultureInfo.InvariantCulture) }),
                cancellationToken).ConfigureAwait(false);
        }
        catch (OperationCanceledException) when (cancellationToken.IsCancellationRequested)
        {
            return;
        }

        _socket = socket;
        using var stop = CancellationTokenSource.CreateLinkedTokenSource(cancellationToken);
        var reporting = ReportAsync(stop.Token);
        try
        {
            while (await socket.ReceiveAsync(stop.Token).ConfigureAwait(false) is { } message)
            {
                if (message.Frame is { } frame)
                {
                    Read(frame);
                }
            }
        }
        catch (OperationCanceledException) when (cancellationToken.IsCancellationRequested)
        {
            // Cancelled, which is how a worker is asked to stop waiting.
        }
        finally
        {
            await stop.CancelAsync().ConfigureAwait(false);
            await reporting.ConfigureAwait(false);
            await Task.WhenAll(_running.Values).ConfigureAwait(false);
            await CloseAgentsAsync().ConfigureAwait(false);
            _socket = null;
            await socket.DisposeAsync().ConfigureAwait(false);
        }
    }

    internal static InboundCall CallOf(Frame frame) => new()
    {
        CallId = frame.Text("call_id"),
        CallType = frame.Text("call_type") is { Length: > 0 } type ? type : Edge.DefaultCallType,
        CalledNumber = frame.Text("called_number"),
        CallerNumber = frame.Text("caller_number"),
        Custom = CustomOf(frame),
        At = TimeOf(frame),
    };

    internal static InboundMessage MessageOf(Frame frame) => new()
    {
        ChannelId = frame.Text("channel_id"),
        ChannelType = frame.Text("channel_type"),
        AgentId = frame.Text("agent_id"),
        ConfigId = frame.Text("config_id"),
        Text = frame.Text("text"),
        MessageId = frame.Text("message_id"),
        UserId = frame.Text("user_id"),
        UserName = frame.Text("user_name"),
        Custom = CustomOf(frame),
        At = TimeOf(frame),
    };

    private void Read(Frame frame)
    {
        switch (frame.Type)
        {
            case "call" when _onCall is { } onCall:
                var call = CallOf(frame);
                Track(() => AnswerAsync(onCall, call));
                break;
            case "message" when _onMessage is { } onMessage:
                var message = MessageOf(frame);
                Track(() => ReplyAsync(onMessage, message));
                break;
            case "ready":
                WorkerId = frame.Text("worker_id");
                break;
            case "pong":
                _latencyMs = (_clock.Elapsed.TotalSeconds - frame.Number("at")) * 1000;
                _pong.TrySetResult();
                break;
        }
    }

    /// <summary>Runs work off the read loop, since reading is also what delivers the next call.</summary>
    private void Track(Func<Task> work)
    {
        var id = Interlocked.Increment(ref _handed);
        var done = new TaskCompletionSource(TaskCreationOptions.RunContinuationsAsynchronously);
        _running[id] = done.Task;
        _ = Task.Run(async () =>
        {
            try
            {
                await work().ConfigureAwait(false);
            }
            finally
            {
                _running.TryRemove(id, out _);
                done.SetResult();
            }
        });
    }

    /// <summary>
    /// Runs the handler for one call and tells the router how it went. Whatever it throws is
    /// somebody else's code failing, so it is reported rather than let escape.
    /// </summary>
    private async Task AnswerAsync(Func<InboundCall, Task> handler, InboundCall call)
    {
        try
        {
            await handler(call).ConfigureAwait(false);
        }
        catch (Exception failure)
        {
            await TellAsync(Frames.Of("rejected", ("call_id", call.CallId), ("reason", failure.Message))).ConfigureAwait(false);
            return;
        }
        await TellAsync(Frames.Of("accepted", ("call_id", call.CallId))).ConfigureAwait(false);
    }

    /// <summary>
    /// Runs the handler for one message. Nothing is reported back: accepting and rejecting are
    /// about a caller waiting on a line, and there is no line here.
    /// </summary>
    private static async Task ReplyAsync(Func<InboundMessage, Task> handler, InboundMessage message)
    {
        try
        {
            await handler(message).ConfigureAwait(false);
        }
        catch (Exception)
        {
            // A message nobody answered is the handler's to log; the worker carries on.
        }
    }

    /// <summary>Tells the router how this process is doing, on a timer.</summary>
    private async Task ReportAsync(CancellationToken cancellationToken)
    {
        try
        {
            while (true)
            {
                await Task.Delay(_reportEvery, cancellationToken).ConfigureAwait(false);
                // Measured from this side, because this is the side a call's audio crosses. A
                // pong that does not come back leaves the last figure standing.
                _pong = new TaskCompletionSource(TaskCreationOptions.RunContinuationsAsynchronously);
                await TellAsync(Frames.Of("ping", ("at", _clock.Elapsed.TotalSeconds))).ConfigureAwait(false);
                try
                {
                    await _pong.Task.WaitAsync(TimeSpan.FromSeconds(5), cancellationToken).ConfigureAwait(false);
                }
                catch (TimeoutException)
                {
                    // The last figure stands.
                }
                await TellAsync(Frames.Of("load", ("active_agents", Active), ("latency_ms", _latencyMs))).ConfigureAwait(false);
            }
        }
        catch (OperationCanceledException)
        {
            // Stopped with the worker.
        }
    }

    /// <summary>Sends one frame, if the socket is still there. None of these is something a call depends on.</summary>
    private async Task TellAsync(JsonObject frame)
    {
        if (_socket is not { Open: true } socket)
        {
            return;
        }
        try
        {
            await socket.SendAsync(frame).ConfigureAwait(false);
        }
        catch (SocketClosedException)
        {
            // Gone; there is nobody to tell.
        }
    }

    private async Task CloseAgentsAsync()
    {
        await _agentsLock.WaitAsync().ConfigureAwait(false);
        try
        {
            foreach (var agent in _agents.Values)
            {
                if (agent.Session is { } session)
                {
                    await session.CloseAsync().ConfigureAwait(false);
                }
            }
            _agents.Clear();
        }
        finally
        {
            _agentsLock.Release();
        }
    }

    /// <summary>A frame's custom data, narrowed to the strings a handler can read.</summary>
    private static Dictionary<string, string> CustomOf(Frame frame)
    {
        var custom = new Dictionary<string, string>();
        foreach (var (key, value) in frame.Nested("custom")?.Json ?? [])
        {
            if (value is JsonValue text && text.TryGetValue(out string? said))
            {
                custom[key] = said;
            }
        }
        return custom;
    }

    private static DateTimeOffset? TimeOf(Frame frame) =>
        DateTimeOffset.TryParse(frame.Text("at"), CultureInfo.InvariantCulture, DateTimeStyles.AssumeUniversal, out var at) ? at : null;
}
