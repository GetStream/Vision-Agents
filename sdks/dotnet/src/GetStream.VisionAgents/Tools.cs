using System.Text.Json;
using System.Text.Json.Nodes;
using System.Text.Json.Schema;
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
    public Tools Register<TArguments, TResult>(
        string name,
        string description,
        Func<TArguments, CancellationToken, Task<TResult>> function)
    {
        var schema = Json.Options.GetJsonSchemaAsNode(typeof(TArguments));
        return Register(name, description, schema as JsonObject ?? [], async (arguments, cancellationToken) =>
        {
            var read = arguments.Deserialize<TArguments>(Json.Options)
                ?? throw new ArgumentException($"{name} was called with no arguments");
            return await function(read, cancellationToken).ConfigureAwait(false);
        });
    }

    /// <summary>Registers a function with its parameter schema written out by hand.</summary>
    public Tools Register(
        string name,
        string description,
        JsonObject parameters,
        Func<JsonElement, CancellationToken, Task<object?>> function)
    {
        if (string.IsNullOrEmpty(name))
        {
            throw new ConfigurationException("a tool needs a name");
        }
        lock (_lock)
        {
            _registered[name] = new Registered(description, parameters, function);
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

    /// <summary>The functions as the session request declares them.</summary>
    internal List<SessionTool> Declared()
    {
        lock (_lock)
        {
            return _registered.Select(pair => new SessionTool
            {
                Name = pair.Key,
                Description = pair.Value.Description,
                Parameters = JsonSerializer.Deserialize<JsonElement>(pair.Value.Parameters.ToJsonString()),
            }).ToList();
        }
    }

    /// <summary>
    /// Runs one call and renders what it returned the way a tool result carries it.
    /// </summary>
    /// <remarks>
    /// Arguments are an opaque JSON string on the wire; an empty one reads as no arguments.
    /// </remarks>
    internal async Task<string> CallAsync(string name, string arguments, CancellationToken cancellationToken)
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
        var output = await found.Function(parsed.RootElement.Clone(), cancellationToken).ConfigureAwait(false);
        return output switch
        {
            null => "",
            string text => text,
            _ => JsonSerializer.Serialize(output, output.GetType(), Json.Options),
        };
    }

    private sealed record Registered(
        string Description,
        JsonObject Parameters,
        Func<JsonElement, CancellationToken, Task<object?>> Function);
}
