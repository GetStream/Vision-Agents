using System.Collections.Concurrent;
using System.Text.Json.Nodes;

namespace GetStream.VisionAgents;

/// <summary>Who a tool call answers, as the router reports them.</summary>
/// <param name="Id">Their user id.</param>
/// <param name="Name">Their name.</param>
/// <param name="Custom">The custom data on their user.</param>
public sealed record ToolUser(string Id, string Name, JsonObject Custom);

/// <summary>The conversation a tool call happens in.</summary>
/// <param name="Cid">The channel, as <c>type:id</c>.</param>
/// <param name="Custom">The custom data on the channel.</param>
public sealed record ToolConversation(string Cid, JsonObject Custom);

/// <summary>
/// A tool call asked to keep going through interruptions too late: the person had already
/// talked over the agent, so the router cancels the call instead.
/// </summary>
public class CallNotHeldException(string toolCallId)
    : Exception($"tool call {toolCallId} was not held: it was already cancelled");

/// <summary>
/// One call of one of the caller's functions: who it is for, and its line back to them.
/// </summary>
/// <remarks>
/// A function registered with a <see cref="ToolCall"/> parameter is handed one. A call made
/// with no socket behind it, as a test makes, has no line, so its updates go nowhere.
/// </remarks>
public sealed class ToolCall
{
    private readonly ToolLine? _line;

    internal ToolCall(ToolLine? line = null) => _line = line;

    /// <summary>The id the model gave the call, which a call sent again keeps.</summary>
    public string Id { get; init; } = "";

    /// <summary>The session the call is for.</summary>
    public string SessionId { get; init; } = "";

    /// <summary>The turn the model made the call in.</summary>
    public string TurnId { get; init; } = "";

    /// <summary>Who the call answers.</summary>
    public ToolUser User { get; init; } = new("", "", []);

    /// <summary>The conversation the call happens in.</summary>
    public ToolConversation Conversation { get; init; } = new("", []);

    /// <summary>Whether a person allowed this call before it ran.</summary>
    public bool Approved { get; init; }

    /// <summary>
    /// Tells the people in the conversation how the call is going, such as "Found three
    /// flights, comparing prices". The agent puts it in its own words.
    /// </summary>
    public Task UpdateAsync(string text, CancellationToken cancellationToken = default) =>
        SendUpdateAsync(text, false, cancellationToken);

    /// <summary><see cref="UpdateAsync"/>, said word for word.</summary>
    public Task SayAsync(string text, CancellationToken cancellationToken = default) =>
        SendUpdateAsync(text, true, cancellationToken);

    /// <summary>
    /// Keeps the call going if the person talks over the agent from here on, for a call that
    /// is past the point of taking back, such as one that has started a payment.
    /// </summary>
    /// <remarks>
    /// The router has the last word. A held call still ends at its timeout, or with the
    /// session.
    /// </remarks>
    /// <exception cref="CallNotHeldException">The router had already cancelled the call.</exception>
    public async Task DisallowInterruptionsAsync(CancellationToken cancellationToken = default)
    {
        if (_line is null || _line.Holds.Held(_line.Key))
        {
            return;
        }
        var answer = _line.Holds.Ask(_line.Key);
        await _line.Send(Frames.Of("tool_hold", ("tool_call_id", Id)), cancellationToken).ConfigureAwait(false);
        if (!await answer.WaitAsync(cancellationToken).ConfigureAwait(false))
        {
            throw new CallNotHeldException(Id);
        }
    }

    private Task SendUpdateAsync(string text, bool verbatim, CancellationToken cancellationToken) =>
        _line is null
            ? Task.CompletedTask
            : _line.Send(Frames.Of("tool_update", ("tool_call_id", Id), ("text", text), ("verbatim", verbatim)), cancellationToken);

    /// <summary>
    /// The call a <c>tool_call</c> frame asks for. <paramref name="sessionId"/> stands in for
    /// a frame that does not name its session.
    /// </summary>
    internal static ToolCall Of(Frame frame, string sessionId, ToolLine line)
    {
        var user = frame.Nested("user");
        var conversation = frame.Nested("conversation");
        return new ToolCall(line)
        {
            Id = frame.Text("id"),
            SessionId = frame.Text("session_id") is { Length: > 0 } named ? named : sessionId,
            TurnId = frame.Text("turn_id"),
            User = new ToolUser(user?.Text("id") ?? "", user?.Text("name") ?? "", Custom(user)),
            Conversation = new ToolConversation(conversation?.Text("cid") ?? "", Custom(conversation)),
            Approved = frame.Flag("approved"),
        };
    }

    /// <summary>
    /// Whether a <c>tool_call</c> frame is one for this process to run: <c>run</c> false says
    /// the router or another watcher has it.
    /// </summary>
    internal static bool ToRun(Frame frame) =>
        !(frame.Json["run"] is JsonValue run && run.TryGetValue(out bool runs) && !runs);

    private static JsonObject Custom(Frame? holder) =>
        holder?.Json["custom"] is JsonObject custom ? (JsonObject)custom.DeepClone() : [];
}

/// <summary>A call's way back over the socket that delivered it.</summary>
/// <param name="Send">Sends a frame on that socket.</param>
/// <param name="Holds">The socket's holds.</param>
/// <param name="Key">What the socket's <c>tool_cancel</c> names the call by.</param>
internal sealed record ToolLine(Func<JsonObject, CancellationToken, Task> Send, Holds Holds, string Key);

/// <summary>
/// The calls that asked to be held, waiting on the router's <c>tool_held</c> or
/// <c>tool_cancel</c>, and the ones it held.
/// </summary>
internal sealed class Holds
{
    private readonly ConcurrentDictionary<string, TaskCompletionSource<bool>> _asking = new();
    private readonly ConcurrentDictionary<string, bool> _held = new();

    public Task<bool> Ask(string key)
    {
        var answer = new TaskCompletionSource<bool>(TaskCreationOptions.RunContinuationsAsynchronously);
        _asking[key] = answer;
        return answer.Task;
    }

    /// <summary>Hands the router's answer to the call waiting on it, if one is.</summary>
    public void Answer(string key, bool held)
    {
        if (!_asking.TryRemove(key, out var asking))
        {
            return;
        }
        if (held)
        {
            _held[key] = true;
        }
        asking.TrySetResult(held);
    }

    /// <summary>Whether the router keeps this call going when the person talks over the agent.</summary>
    public bool Held(string key) => _held.ContainsKey(key);

    public void Forget(string key)
    {
        _asking.TryRemove(key, out _);
        _held.TryRemove(key, out _);
    }
}
