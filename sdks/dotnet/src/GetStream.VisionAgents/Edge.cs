using System.Security.Cryptography;
using GetStream.Models;

namespace GetStream.VisionAgents;

/// <summary>One Stream call, named the way the backend needs it named.</summary>
public sealed record Call(string Id, string Type = Edge.DefaultCallType);

/// <summary>The Stream credentials calls are created with. Empty fields read the environment.</summary>
public sealed record EdgeOptions
{
    /// <summary>The Stream app's key. Falls back to <c>STREAM_API_KEY</c>.</summary>
    public string? ApiKey { get; init; }

    /// <summary>Its secret, which signs the tokens. Falls back to <c>STREAM_API_SECRET</c>.</summary>
    public string? ApiSecret { get; init; }

    /// <summary>Stream's API, for pointing at another deployment.</summary>
    public string? BaseUrl { get; init; }

    /// <summary>
    /// Points the monitoring link at another deployment of the demo. Falls back to
    /// <c>EXAMPLE_BASE_URL</c>, then Stream's hosted one.
    /// </summary>
    public string? MonitorUrl { get; init; }

    /// <summary>The <see cref="HttpClient"/> Stream's SDK sends with, for <c>IHttpClientFactory</c>.</summary>
    public HttpClient? HttpClient { get; init; }
}

/// <summary>
/// Creates the Stream calls an agent joins, and mints the tokens for watching them.
/// </summary>
/// <remarks>
/// The router joins a call; it does not create one. Creating it is Stream's own API, which
/// Stream's server-side SDK (<c>getstream-net</c>) already speaks, so this is that SDK
/// rather than a second copy of it.
/// </remarks>
public sealed class Edge
{
    /// <summary>The Stream call type used when none is named.</summary>
    public const string DefaultCallType = "agent";

    /// <summary>Stream's hosted video demo, the page the Python examples open.</summary>
    public const string DefaultMonitorUrl = "https://getstream.io/video/demos";

    // A call outliving the browser's token is a call nobody is still on.
    private static readonly TimeSpan MonitorTokenValidity = TimeSpan.FromHours(1);

    private readonly StreamClient _stream;
    private readonly VideoClient _video;
    private readonly string _monitorUrl;

    /// <summary>Reads whatever the options leave empty out of the environment.</summary>
    /// <exception cref="ConfigurationException">There is no key or no secret.</exception>
    public Edge(EdgeOptions? options = null)
    {
        options ??= new EdgeOptions();
        ApiKey = options.ApiKey ?? Backend.Env("STREAM_API_KEY") ?? "";
        var secret = options.ApiSecret ?? Backend.Env("STREAM_API_SECRET") ?? "";
        if (ApiKey == "" || secret == "")
        {
            throw new ConfigurationException("STREAM_API_KEY and STREAM_API_SECRET are required to create a call");
        }
        var streamOptions = new StreamOptions { ApiKey = ApiKey, ApiSecret = secret };
        if (options.BaseUrl is { Length: > 0 } baseUrl)
        {
            streamOptions.BaseUrl = baseUrl;
        }
        if (options.HttpClient is { } http)
        {
            streamOptions.HttpClient = http;
        }
        _stream = new StreamClient(streamOptions);
        _video = new VideoClient(_stream);
        _monitorUrl = (options.MonitorUrl ?? Backend.Env("EXAMPLE_BASE_URL") ?? DefaultMonitorUrl).TrimEnd('/');
    }

    /// <summary>The Stream app's key.</summary>
    public string ApiKey { get; }

    /// <summary>
    /// Creates the call the backend will join, or returns the one already under that id.
    /// </summary>
    /// <remarks>A null id names a new call after a random one, which is what a one-off conversation wants.</remarks>
    public async Task<Call> CreateCallAsync(string? id, string? type, string createdBy, CancellationToken cancellationToken = default)
    {
        if (string.IsNullOrEmpty(createdBy))
        {
            throw new ConfigurationException("a call needs somebody to have created it");
        }
        var call = new Call(string.IsNullOrEmpty(id) ? Convert.ToHexStringLower(RandomNumberGenerator.GetBytes(8)) : id,
            string.IsNullOrEmpty(type) ? DefaultCallType : type);
        try
        {
            await _video.GetOrCreateCallAsync(call.Type, call.Id,
                new GetOrCreateCallRequest { Data = new CallRequest { CreatedByID = createdBy } },
                cancellationToken).ConfigureAwait(false);
        }
        catch (GetStreamException failure)
        {
            throw new RouterException(failure is GetStreamApiException api ? api.StatusCode : 0,
                $"POST /video/call/{call.Type}/{call.Id}", failure.Message, failure);
        }
        return call;
    }

    /// <summary>Mints a token for somebody to join a call as. Signed here, so no request is made.</summary>
    public string Token(string userId, TimeSpan? validity = null) =>
        string.IsNullOrEmpty(userId)
            ? throw new ConfigurationException("a token needs a user to name")
            : _stream.CreateUserToken(userId, validity ?? MonitorTokenValidity);

    /// <summary>A link a person can open to join a call from a browser and hear the agent.</summary>
    public string MonitorUrl(Call call, string userId, string? userName = null)
    {
        if (string.IsNullOrEmpty(call.Id))
        {
            throw new ConfigurationException("there is no call to watch");
        }
        var query = string.Join('&',
            $"api_key={Uri.EscapeDataString(ApiKey)}",
            $"token={Uri.EscapeDataString(Token(userId))}",
            "skip_lobby=true",
            $"user_name={Uri.EscapeDataString(string.IsNullOrEmpty(userName) ? userId : userName)}");
        return $"{_monitorUrl}/join/{Uri.EscapeDataString(call.Id)}?{query}";
    }
}
