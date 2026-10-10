using System.Collections.Concurrent;
using System.Diagnostics;
using System.Globalization;
using System.Net.WebSockets;
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
/// that is running is answered by that session, unless the agent leaves text to dispatch, in
/// which case it arrives with its <see cref="InboundMessage.SessionId"/> and
/// <see cref="AnswerAsync"/> answers it. Several workers can wait at once, and the work is
/// shared between them.
/// </para>
/// <para>
/// A worker can also <see cref="Host"/> an agent's tools: the router offers them to every
/// session opened under that agent, and sends each call down this connection.
/// </para>
/// </remarks>
public sealed class Dispatch
{
    private const string Path = "/v1/dispatch";
    // Bounds on the wait between attempts to reach a router that dropped this worker. A
    // router being redeployed is back within seconds; one down for longer is not worth
    // asking twice a second.
    private static readonly TimeSpan LastRetry = TimeSpan.FromSeconds(30);
    // How long a connection has to have lasted for its loss to be a fresh drop rather than
    // another failed attempt, so the wait starts again from the first.
    private static readonly TimeSpan SteadyAfter = TimeSpan.FromMinutes(1);

    private readonly TimeSpan _reportEvery;
    private readonly List<(string AgentId, Tools Tools, TimeSpan Timeout)> _hosted = [];
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
    // The calls and messages alone, which is what the router counts against this worker's
    // capacity. A hosted tool call is not one: the router tracks those by their answer.
    private int _handling;

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

    internal TimeSpan FirstRetry { get; init; } = TimeSpan.FromSeconds(1);

    /// <summary>
    /// What to do with an arriving call. It runs as its own task, so one long call does not
    /// stop the next from being answered. The router is told the call is done when it
    /// returns, with what it threw if it throws.
    /// </summary>
    public Dispatch WaitForCall(Func<InboundCall, Task> handler)
    {
        _onCall = handler;
        return this;
    }

    /// <summary>
    /// What to do with a message written to an agent that is not running, or to a running
    /// session whose agent leaves text to dispatch. It runs as its own task, the way a call's
    /// does.
    /// </summary>
    /// <example>
    /// <code>
    /// dispatch.WaitForMessage(async message =>
    /// {
    ///     if (message.SessionId != "")
    ///     {
    ///         await dispatch.AnswerAsync(message);
    ///         return;
    ///     }
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
    /// Runs an agent's <see cref="Agent.Tools"/> for every session opened under it, whoever opened it.
    /// </summary>
    /// <remarks>
    /// A session's own tools run in the process that opened it, which is no use to a
    /// conversation opened from a browser. Hosting is the other direction: the router offers
    /// them to each session whose agent id or config name is the agent's
    /// <see cref="Agent.Name"/>, and sends every call here. Call before <see cref="RunAsync"/>.
    /// </remarks>
    /// <param name="agent">The agent, usually <see cref="VisionAgentsClient.Agent"/>, whose tools are hosted.</param>
    /// <param name="toolTimeout">
    /// How long the router waits for one tool call to be answered before telling the model it
    /// failed, not how long the worker runs; null takes the router's default of two minutes.
    /// </param>
    public Dispatch Host(Agent agent, TimeSpan? toolTimeout = null)
    {
        _hosted.Add((agent.Name, agent.Tools, toolTimeout ?? TimeSpan.Zero));
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
    /// <exception cref="InvalidOperationException">A session is already holding the message.</exception>
    public Task<Agent> GetOrCreateAgentAsync(InboundMessage message, Func<Agent> createAgent, CancellationToken cancellationToken = default) =>
        GetOrCreateAgentAsync(message, _ => Task.FromResult(createAgent()), cancellationToken);

    /// <inheritdoc cref="GetOrCreateAgentAsync(InboundMessage, Func{Agent}, CancellationToken)"/>
    public async Task<Agent> GetOrCreateAgentAsync(InboundMessage message, Func<CancellationToken, Task<Agent>> createAgent, CancellationToken cancellationToken = default)
    {
        if (message.SessionId != "")
        {
            throw new InvalidOperationException("a session is already holding this conversation; answer it there with AnswerAsync");
        }
        await _agentsLock.WaitAsync(cancellationToken).ConfigureAwait(false);
        try
        {
            if (_agents.TryGetValue(message.ChannelId, out var answering) && answering.Session is { Live: true })
            {
                return answering;
            }
            var agent = await createAgent(cancellationToken).ConfigureAwait(false);
            await agent.ChatAsync(new SessionOptions { AgentId = message.AgentId }, cancellationToken).ConfigureAwait(false);
            _agents[message.ChannelId] = agent;
            return agent;
        }
        finally
        {
            _agentsLock.Release();
        }
    }

    /// <summary>
    /// Has the model answer a message written to a running session whose agent leaves text
    /// to dispatch, which is what the person who wrote it is waiting on.
    /// </summary>
    /// <remarks>
    /// The response is created with this worker's own credential, acting for whoever wrote
    /// the message, so it reaches a conversation that belongs to them and goes to the model
    /// rather than back to a worker. It carries the message's request id, so the answer lands on it.
    /// </remarks>
    /// <exception cref="InvalidOperationException">No session is holding the message.</exception>
    public async Task<Response> AnswerAsync(InboundMessage message, CancellationToken cancellationToken = default)
    {
        if (message.SessionId == "")
        {
            throw new InvalidOperationException("no session is holding this message; open one with GetOrCreateAgentAsync");
        }
        using var acting = Client.ActingFor(message.UserId);
        return await new Responses(acting, message.SessionId)
            .AnswerAsync(message.Text, null, message.RequestId, cancellationToken).ConfigureAwait(false);
    }

    /// <summary>
    /// Waits for work until cancelled, until the router closes the connection on purpose, or
    /// until it refuses the tools this worker hosts.
    /// </summary>
    /// <remarks>
    /// <para>
    /// A connection that drops any other way, a router being redeployed or a load balancer
    /// ending an idle socket, is opened again and the router told again what this worker
    /// hosts. Only the first connection failing is thrown, since that is a worker that was
    /// never going to work.
    /// </para>
    /// <para>
    /// Returns rather than throws when cancelled. Work still being handled is waited for
    /// either way, because dropping a call would hang up on whoever is talking.
    /// </para>
    /// </remarks>
    /// <exception cref="InvalidOperationException">
    /// No handler is registered and nothing is hosted, so work would arrive with nothing to do
    /// it; or the router refused the hosted tools.
    /// </exception>
    public async Task RunAsync(CancellationToken cancellationToken = default)
    {
        if (_onCall is null && _onMessage is null && _hosted.Count == 0)
        {
            throw new InvalidOperationException("register a handler with WaitForCall or WaitForMessage, or Host tools, before running");
        }

        Socket socket;
        try
        {
            socket = await Socket.ConnectAsync(Client.Backend, Waiting(), cancellationToken).ConfigureAwait(false);
        }
        catch (OperationCanceledException) when (cancellationToken.IsCancellationRequested)
        {
            return;
        }

        using var stop = CancellationTokenSource.CreateLinkedTokenSource(cancellationToken);
        var reporting = ReportAsync(stop.Token);
        try
        {
            var retry = FirstRetry;
            while (true)
            {
                _socket = socket;
                var opened = _clock.Elapsed;
                if (!await ServeAsync(socket, cancellationToken).ConfigureAwait(false))
                {
                    return;
                }
                _socket = null;
                await socket.DisposeAsync().ConfigureAwait(false);
                if (_clock.Elapsed - opened >= SteadyAfter)
                {
                    retry = FirstRetry;
                }

                while (true)
                {
                    await Task.Delay(retry, cancellationToken).ConfigureAwait(false);
                    retry = TimeSpan.FromTicks(Math.Min(retry.Ticks * 2, LastRetry.Ticks));
                    try
                    {
                        socket = await Socket.ConnectAsync(Client.Backend, Waiting(), cancellationToken).ConfigureAwait(false);
                        break;
                    }
                    catch (RouterException)
                    {
                        // Still away; the next attempt waits longer.
                    }
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
        SessionId = frame.Text("session_id"),
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
        SessionId = frame.Text("session_id"),
        RequestId = frame.Text("request_id"),
        Text = frame.Text("text"),
        MessageId = frame.Text("message_id"),
        UserId = frame.Text("user_id"),
        UserName = frame.Text("user_name"),
        Custom = CustomOf(frame),
        At = TimeOf(frame),
    };

    /// <summary>
    /// What this worker says about itself on the way in: how much it can hold, how much it is
    /// still holding from before a reconnect, and which kinds of work it answers. On the
    /// handshake rather than in a frame, because the router may hand over work before reading
    /// anything; <c>handles</c> is sent even when empty, for a worker that only hosts tools.
    /// </summary>
    private Uri Waiting() => Client.Backend.SocketUrl(Path, new Dictionary<string, string>
    {
        ["capacity"] = Capacity.ToString(CultureInfo.InvariantCulture),
        ["active"] = Volatile.Read(ref _handling).ToString(CultureInfo.InvariantCulture),
        ["handles"] = string.Join(',', new[] { _onCall is null ? null : "call", _onMessage is null ? null : "message" }.OfType<string>()),
    });

    /// <summary>
    /// Reads one connection until it ends, reporting whether the router dropped it rather
    /// than closing it on purpose. Going away is a router shutting down, which is when a
    /// worker should find the one replacing it.
    /// </summary>
    private async Task<bool> ServeAsync(Socket socket, CancellationToken cancellationToken)
    {
        while (await socket.ReceiveAsync(cancellationToken).ConfigureAwait(false) is { } message)
        {
            if (message.Frame is { } frame)
            {
                await ReadAsync(frame, cancellationToken).ConfigureAwait(false);
            }
        }
        return !cancellationToken.IsCancellationRequested && socket.CloseStatus != WebSocketCloseStatus.NormalClosure;
    }

    private async Task ReadAsync(Frame frame, CancellationToken cancellationToken)
    {
        switch (frame.Type)
        {
            case "call":
                await HandleAsync(frame.Text("work_id"), _onCall is { } onCall ? () => onCall(CallOf(frame)) : null,
                    "this worker answers no calls").ConfigureAwait(false);
                break;
            case "message":
                await HandleAsync(frame.Text("work_id"), _onMessage is { } onMessage ? () => onMessage(MessageOf(frame)) : null,
                    "this worker answers no messages").ConfigureAwait(false);
                break;
            case "ready":
                WorkerId = frame.Text("worker_id");
                foreach (var (agentId, tools, timeout) in _hosted)
                {
                    await TellAsync(Frames.Of("host_tools", ("agent_id", agentId), ("tools", tools.Declared()),
                        ("timeout_ms", (long)timeout.TotalMilliseconds))).ConfigureAwait(false);
                }
                break;
            case "tool_call":
                await RunHostedAsync(frame, cancellationToken).ConfigureAwait(false);
                break;
            case "hosting":
                Trace.TraceInformation("the router sends tools for agent {0} to this worker", frame.Text("agent_id"));
                break;
            case "hosting_refused":
                // A worker whose tools were refused is one nobody will call; saying so beats
                // sitting connected looking healthy.
                throw new InvalidOperationException(
                    $"the router refused to host tools for agent {frame.Text("agent_id")}: {frame.Text("reason")}");
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
    /// Starts one hosted call, off the read loop like any other work, and answers with what
    /// the tool returned or what it threw.
    /// </summary>
    private async Task RunHostedAsync(Frame frame, CancellationToken cancellationToken)
    {
        var id = frame.Text("id");
        var name = frame.Text("name");
        if (_hosted.Select(offer => offer.Tools).FirstOrDefault(tools => tools.Runs(name)) is not { } tools)
        {
            await TellAsync(Frames.Of("tool_result", ("id", id), ("error", $"this worker does not run {name}"))).ConfigureAwait(false);
            return;
        }
        Track(async () =>
        {
            JsonObject result;
            try
            {
                result = Frames.Of("tool_result", ("id", id),
                    ("output", await tools.CallAsync(name, frame.Text("arguments"), cancellationToken).ConfigureAwait(false)));
            }
            catch (Exception failure)
            {
                // Whatever a tool throws is the model's to hear about, not this loop's.
                result = Frames.Of("tool_result", ("id", id), ("error", failure.Message));
            }
            await TellAsync(result).ConfigureAwait(false);
        });
    }

    /// <summary>
    /// Runs the handler for one call or message and tells the router it is done. Whatever it
    /// throws is somebody else's code failing, so it is reported rather than let escape. Work
    /// with no handler is reported done too, because the router holds its room until then.
    /// </summary>
    private async Task HandleAsync(string workId, Func<Task>? handler, string unhandled)
    {
        if (handler is null)
        {
            await DoneAsync(workId, unhandled).ConfigureAwait(false);
            return;
        }
        Interlocked.Increment(ref _handling);
        Track(async () =>
        {
            string? failed = null;
            try
            {
                await handler().ConfigureAwait(false);
            }
            catch (Exception failure)
            {
                failed = failure.Message;
            }
            finally
            {
                Interlocked.Decrement(ref _handling);
            }
            await DoneAsync(workId, failed).ConfigureAwait(false);
        });
    }

    private Task DoneAsync(string workId, string? error) => TellAsync(error is null
        ? Frames.Of("done", ("work_id", workId))
        : Frames.Of("done", ("work_id", workId), ("error", error)));

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
