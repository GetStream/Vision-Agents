using System.Text.Json;
using System.Text.Json.Nodes;
using System.Text.Json.Schema;
using System.Text.RegularExpressions;
using GetStream.VisionAgents.Models;

namespace GetStream.VisionAgents;

/// <summary>
/// The caller's own functions, which every session an agent opens offers the model.
/// </summary>
/// <remarks>
/// The model runs in the backend; the functions run in this process, next to whatever they
/// need to reach. Held on the agent rather than per session, so a function registered once
/// is offered to the next conversation too.
/// </remarks>
public sealed class Tools
{
    private readonly Dictionary<string, Registered> _registered = new(StringComparer.Ordinal);
    private readonly Lock _lock = new();

    /// <summary>
    /// Registers a function whose arguments are read as <typeparamref name="TArguments"/>,
    /// with the parameter schema taken from that type.
    /// </summary>
    /// <param name="name">How the model asks for it.</param>
    /// <param name="description">What the model chooses it by.</param>
    /// <param name="function">What runs.</param>
    /// <param name="displayTitle">At most 80 characters, shown on the reply's <c>ai_tool_call</c> attachment.</param>
    /// <param name="interruptible">Whether the person talking over the agent cancels a call in flight.</param>
    /// <param name="timeout">How long one call may take, in place of the session's tool timeout.</param>
    /// <param name="approval">A question a person answers before each call runs, where each <c>{name}</c> is a parameter.</param>
    public Tools Register<TArguments, TResult>(
        string name,
        string description,
        Func<TArguments, CancellationToken, Task<TResult>> function,
        string? displayTitle = null,
        bool interruptible = false,
        TimeSpan? timeout = null,
        string? approval = null) =>
        Register<TArguments, TResult>(name, description, (arguments, _, cancellationToken) => function(arguments, cancellationToken),
            displayTitle, interruptible, timeout, approval);

    /// <summary>
    /// <see cref="Register{TArguments, TResult}(string, string, Func{TArguments, CancellationToken, Task{TResult}}, string?, bool, TimeSpan?, string?)"/>
    /// for a function that is also handed its <see cref="ToolCall"/>: who it answers, and its
    /// line back to them for updates and holds.
    /// </summary>
    public Tools Register<TArguments, TResult>(
        string name,
        string description,
        Func<TArguments, ToolCall, CancellationToken, Task<TResult>> function,
        string? displayTitle = null,
        bool interruptible = false,
        TimeSpan? timeout = null,
        string? approval = null)
    {
        var schema = Json.Options.GetJsonSchemaAsNode(typeof(TArguments));
        return Register(name, description, schema as JsonObject ?? [], async (arguments, toolCallContext, cancellationToken) =>
        {
            var read = arguments.Deserialize<TArguments>(Json.Options)
                ?? throw new ArgumentException($"{name} was called with no arguments");
            return await function(read, toolCallContext, cancellationToken).ConfigureAwait(false);
        }, displayTitle, interruptible, timeout, approval);
    }

    /// <summary>Registers a function with its parameter schema written out by hand.</summary>
    /// <param name="name">How the model asks for it.</param>
    /// <param name="description">What the model chooses it by.</param>
    /// <param name="parameters">The JSON schema of its arguments.</param>
    /// <param name="function">What runs.</param>
    /// <param name="displayTitle">At most 80 characters, shown on the reply's <c>ai_tool_call</c> attachment.</param>
    /// <param name="interruptible">Whether the person talking over the agent cancels a call in flight.</param>
    /// <param name="timeout">How long one call may take, in place of the session's tool timeout.</param>
    /// <param name="approval">A question a person answers before each call runs, where each <c>{name}</c> is a parameter.</param>
    public Tools Register(
        string name,
        string description,
        JsonObject parameters,
        Func<JsonElement, CancellationToken, Task<object?>> function,
        string? displayTitle = null,
        bool interruptible = false,
        TimeSpan? timeout = null,
        string? approval = null) =>
        Register(name, description, parameters, (arguments, _, cancellationToken) => function(arguments, cancellationToken),
            displayTitle, interruptible, timeout, approval);

    /// <summary>
    /// <see cref="Register(string, string, JsonObject, Func{JsonElement, CancellationToken, Task{object?}}, string?, bool, TimeSpan?, string?)"/>
    /// for a function that is also handed its <see cref="ToolCall"/>.
    /// </summary>
    public Tools Register(
        string name,
        string description,
        JsonObject parameters,
        Func<JsonElement, ToolCall, CancellationToken, Task<object?>> function,
        string? displayTitle = null,
        bool interruptible = false,
        TimeSpan? timeout = null,
        string? approval = null)
    {
        if (string.IsNullOrEmpty(name))
        {
            throw new ConfigurationException("a tool needs a name");
        }
        if (displayTitle is { Length: > 80 })
        {
            throw new ConfigurationException($"the display title of {name} is longer than 80 characters");
        }
        if (timeout is { } bound && bound <= TimeSpan.Zero)
        {
            throw new ConfigurationException($"{name} has a timeout that is not positive");
        }
        CheckApproval(name, approval, parameters);
        lock (_lock)
        {
            _registered[name] = new Registered(description, parameters, function, VisionAgentsClient.Blank(displayTitle),
                interruptible, timeout, approval?.Trim());
        }
        return this;
    }

    /// <summary>How many functions are registered.</summary>
    public int Count
    {
        get
        {
            lock (_lock)
            {
                return _registered.Count;
            }
        }
    }

    /// <summary>Whether a function by this name is registered.</summary>
    internal bool Runs(string name)
    {
        lock (_lock)
        {
            return _registered.ContainsKey(name);
        }
    }

    /// <summary>The functions as the session request declares them.</summary>
    internal List<SessionTool> Declared()
    {
        lock (_lock)
        {
            return _registered.Select(pair => new SessionTool
            {
                Name = pair.Key,
                Description = pair.Value.Description,
                Parameters = JsonSerializer.Deserialize<Dictionary<string, object>>(pair.Value.Parameters.ToJsonString()),
                DisplayTitle = pair.Value.DisplayTitle,
                // Only what differs from the router's defaults.
                Interruptible = pair.Value.Interruptible ? true : null,
                TimeoutMs = pair.Value.Timeout is { } timeout ? (long)timeout.TotalMilliseconds : null,
                Approval = pair.Value.Approval is { } title ? new SessionToolApproval { Title = title } : null,
            }).ToList();
        }
    }

    /// <summary>
    /// Runs one call and renders what it returned the way a tool result carries it.
    /// </summary>
    /// <remarks>
    /// Arguments are an opaque JSON string on the wire; an empty one reads as no arguments.
    /// </remarks>
    /// <exception cref="TimeoutException">The call took longer than the tool's timeout.</exception>
    internal async Task<string> CallAsync(string name, string arguments, CancellationToken cancellationToken, ToolCall? toolCallContext = null)
    {
        Registered? found;
        lock (_lock)
        {
            _registered.TryGetValue(name, out found);
        }
        if (found is null)
        {
            throw new InvalidOperationException($"{name} was asked for, and nothing by that name is registered");
        }

        using var parsed = JsonDocument.Parse(string.IsNullOrWhiteSpace(arguments) ? "{}" : arguments);
        object? output;
        if (found.Timeout is not { } timeout)
        {
            output = await found.Function(parsed.RootElement.Clone(), toolCallContext ?? new ToolCall(), cancellationToken).ConfigureAwait(false);
        }
        else
        {
            using var bounded = CancellationTokenSource.CreateLinkedTokenSource(cancellationToken);
            bounded.CancelAfter(timeout);
            try
            {
                // Waited on apart from the token too, for a function that never looks at it.
                output = await found.Function(parsed.RootElement.Clone(), toolCallContext ?? new ToolCall(), bounded.Token)
                    .WaitAsync(timeout, cancellationToken).ConfigureAwait(false);
            }
            catch (Exception failure) when (failure is TimeoutException or OperationCanceledException
                && !cancellationToken.IsCancellationRequested)
            {
                await bounded.CancelAsync().ConfigureAwait(false);
                throw new TimeoutException($"{name} took longer than {timeout.TotalSeconds:0.###}s");
            }
        }
        return output switch
        {
            null => "",
            string text => text,
            _ => JsonSerializer.Serialize(output, output.GetType(), Json.Options),
        };
    }

    /// <summary>Refuses an approval question the router could not fill in from the arguments.</summary>
    private static void CheckApproval(string name, string? approval, JsonObject parameters)
    {
        if (approval is null)
        {
            return;
        }
        if (string.IsNullOrWhiteSpace(approval))
        {
            throw new ConfigurationException($"{name}'s approval question cannot be blank");
        }
        var properties = parameters["properties"] as JsonObject;
        foreach (Match placeholder in Placeholder.Matches(approval))
        {
            var argument = placeholder.Groups[1].Value;
            if (properties?.ContainsKey(argument) != true)
            {
                throw new ConfigurationException($"{name}'s approval names {{{argument}}}, which is not one of its parameters");
            }
        }
    }

    private static readonly Regex Placeholder = new(@"\{([A-Za-z_][A-Za-z0-9_]*)\}", RegexOptions.Compiled);

    private sealed record Registered(
        string Description,
        JsonObject Parameters,
        Func<JsonElement, ToolCall, CancellationToken, Task<object?>> Function,
        string? DisplayTitle,
        bool Interruptible,
        TimeSpan? Timeout,
        string? Approval);
}
