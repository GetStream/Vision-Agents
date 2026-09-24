using System.Globalization;
using System.Net;
using System.Net.Http.Headers;
using System.Text;
using System.Text.Json;
using GetStream.VisionAgents.Models;

namespace GetStream.VisionAgents;

/// <summary>
/// The router held once: where it is, and who is calling it.
/// </summary>
/// <remarks>
/// <para>
/// Takes an <see cref="HttpClient"/> so it can be registered as a typed client with
/// <c>IHttpClientFactory</c>. One passed in is never disposed here; one created here is,
/// with the client.
/// </para>
/// <para>
/// One method per HTTP verb rather than one per operation, for anything the named classes
/// do not cover: every request and response shape is generated in
/// <see cref="GetStream.VisionAgents.Models"/>, so a wrapper per operation would say nothing
/// the types do not.
/// </para>
/// </remarks>
public sealed class VisionAgentsClient : IDisposable
{
    private readonly HttpClient _http;
    private readonly bool _ownsHttp;

    /// <summary>Points a client at a router, creating its own <see cref="HttpClient"/>.</summary>
    public VisionAgentsClient(VisionAgentsOptions? options = null)
        : this(new HttpClient(), options ?? new VisionAgentsOptions(), owns: true)
    {
    }

    /// <summary>Points a client at a router over an <see cref="HttpClient"/> the caller owns.</summary>
    public VisionAgentsClient(HttpClient httpClient, VisionAgentsOptions? options = null)
        : this(httpClient, options ?? new VisionAgentsOptions(), owns: false)
    {
    }

    private VisionAgentsClient(HttpClient httpClient, VisionAgentsOptions options, bool owns)
        : this(httpClient, new Backend(options), owns)
    {
    }

    private VisionAgentsClient(HttpClient httpClient, Backend backend, bool owns)
    {
        _http = httpClient ?? throw new ArgumentNullException(nameof(httpClient));
        _ownsHttp = owns;
        Backend = backend;
    }

    /// <summary>The router's base URL.</summary>
    public string Url => Backend.Url;

    /// <summary>
    /// Whether this speaks for a process the customer runs rather than for a device.
    /// </summary>
    /// <remarks>
    /// Only a server-side caller reaches the operations the spec does not mark client
    /// accessible, which is everything about how an agent is configured. Worth asking before
    /// a call rather than reading a 403 afterwards; the router is what enforces it.
    /// </remarks>
    public bool ServerSide => Backend.ServerSide;

    /// <summary>The sessions already held: listed, searched and read back.</summary>
    public Sessions Sessions => new(this);

    /// <summary>
    /// Speech-to-text, text-to-speech, the language model and search, on their own, under a
    /// stored router config and with cost labels on everything routed.
    /// </summary>
    public Router Router(string? config = null, IReadOnlyDictionary<string, string>? tags = null) => new(this, config, tags);

    internal Backend Backend { get; }

    internal HttpClient Http => _http;

    /// <summary>
    /// An agent addressed by the config it is stored under. No request is made.
    /// </summary>
    public Agent Agent(string config) => new(new AgentOptions { Config = config, Client = this });

    /// <summary>
    /// A client acting for one end user, holding the token that proves it.
    /// </summary>
    /// <remarks>
    /// A new client rather than a change to this one, because a process usually holds both:
    /// its own credential for what only a backend may do, and one per user for the
    /// conversations that belong to them. The <see cref="HttpClient"/> is shared.
    /// </remarks>
    public VisionAgentsClient AsUser(string userId, string token)
    {
        if (string.IsNullOrEmpty(userId))
        {
            throw new ConfigurationException("a user needs an id");
        }
        if (string.IsNullOrEmpty(token))
        {
            throw new ConfigurationException($"there is no token for {userId} to hold");
        }
        return new VisionAgentsClient(_http, Backend.As(userId, token), owns: false);
    }

    /// <summary>A client acting for a guest, which is what their conversations belong to.</summary>
    public VisionAgentsClient AsGuest(GuestUser guest) => AsUser(guest.Id, guest.Token);

    /// <summary>
    /// Mints a guest, so somebody can talk to an agent before they sign up.
    /// </summary>
    /// <remarks>
    /// Nothing is remembered here. A server handling two visitors that remembered a guest
    /// would hand them each other's conversations, so which visitor is which is the
    /// caller's to keep. Asking for an id that is already a guest returns that guest with a
    /// fresh token.
    /// </remarks>
    public Task<GuestUser> GuestUserAsync(GuestOptions? options = null, CancellationToken cancellationToken = default)
    {
        options ??= new GuestOptions();
        var request = new GuestUserRequest
        {
            Id = Blank(options.Id),
            Name = Blank(options.Name),
            Custom = options.Custom is { Count: > 0 } ? options.Custom : null,
        };
        return PostAsync<GuestUser>("/v1/agents/guests", request, cancellationToken);
    }

    /// <summary>
    /// Moves a guest's conversations onto the account they turned out to be.
    /// </summary>
    /// <remarks>
    /// Server side only: it is the backend that just authenticated the account that knows
    /// which guest it was. A device able to ask this could claim anybody's conversations by
    /// guessing a guest id, so this refuses first, with a reason, rather than letting a 403
    /// be the explanation.
    /// </remarks>
    public Task<ClaimGuestResult> ClaimGuestUserAsync(string guestId, string userId, CancellationToken cancellationToken = default)
    {
        if (!ServerSide)
        {
            throw new ConfigurationException(
                "claiming a guest is server side only: it is the backend that just authenticated the account that knows which guest it was");
        }
        if (string.IsNullOrEmpty(guestId) || string.IsNullOrEmpty(userId))
        {
            throw new ConfigurationException("claiming a guest needs the guest and the account");
        }
        return PostAsync<ClaimGuestResult>("/v1/agents/guests/claim",
            new ClaimGuestRequest { GuestId = guestId, UserId = userId }, cancellationToken);
    }

    /// <summary>GETs a path and reads the answer as <typeparamref name="T"/>.</summary>
    /// <param name="path">The path, with any path parameter escaped by the caller.</param>
    /// <param name="query">Query parameters. A null or empty value is left out.</param>
    /// <param name="cancellationToken">Stops waiting.</param>
    public Task<T> GetAsync<T>(string path, IReadOnlyDictionary<string, string?>? query = null, CancellationToken cancellationToken = default) =>
        SendAsync<T>(HttpMethod.Get, path, query, null, cancellationToken);

    /// <summary>POSTs a body and reads the answer as <typeparamref name="T"/>.</summary>
    public Task<T> PostAsync<T>(string path, object? body, CancellationToken cancellationToken = default) =>
        SendAsync<T>(HttpMethod.Post, path, null, body, cancellationToken);

    /// <summary>POSTs a body to an operation that answers with nothing.</summary>
    public Task PostAsync(string path, object? body, CancellationToken cancellationToken = default) =>
        SendAsync<Nothing>(HttpMethod.Post, path, null, body, cancellationToken);

    /// <summary>PUTs a body and reads the answer as <typeparamref name="T"/>.</summary>
    public Task<T> PutAsync<T>(string path, object? body, CancellationToken cancellationToken = default) =>
        SendAsync<T>(HttpMethod.Put, path, null, body, cancellationToken);

    /// <summary>PATCHes a body and reads the answer as <typeparamref name="T"/>.</summary>
    public Task<T> PatchAsync<T>(string path, object? body, CancellationToken cancellationToken = default) =>
        SendAsync<T>(HttpMethod.Patch, path, null, body, cancellationToken);

    /// <summary>DELETEs a path.</summary>
    public Task DeleteAsync(string path, CancellationToken cancellationToken = default) =>
        SendAsync<Nothing>(HttpMethod.Delete, path, null, null, cancellationToken);

    /// <summary>Disposes the <see cref="HttpClient"/> if this client created it.</summary>
    public void Dispose()
    {
        if (_ownsHttp)
        {
            _http.Dispose();
        }
    }

    /// <summary>
    /// Sends one request and reads what came back, raising what the router said instead.
    /// </summary>
    internal async Task<T> SendAsync<T>(
        HttpMethod method,
        string path,
        IReadOnlyDictionary<string, string?>? query,
        object? body,
        CancellationToken cancellationToken)
    {
        var operation = $"{method.Method} {path}";
        using var request = new HttpRequestMessage(method, Url + path + QueryString(query));
        foreach (var (name, value) in Backend.Headers())
        {
            request.Headers.TryAddWithoutValidation(name, value);
        }
        request.Headers.Accept.Add(new MediaTypeWithQualityHeaderValue("application/json"));
        if (body is not null)
        {
            request.Content = new StringContent(
                JsonSerializer.Serialize(body, body.GetType(), Json.Options), Encoding.UTF8, "application/json");
        }

        HttpResponseMessage response;
        try
        {
            response = await _http.SendAsync(request, cancellationToken).ConfigureAwait(false);
        }
        catch (HttpRequestException failure)
        {
            throw new RouterException(0, operation, failure.Message, failure);
        }

        using (response)
        {
            var text = await response.Content.ReadAsStringAsync(cancellationToken).ConfigureAwait(false);
            if (!response.IsSuccessStatusCode)
            {
                throw new RouterException((int)response.StatusCode, operation, Said(text, response.StatusCode))
                {
                    RetryAfter = response.Headers.RetryAfter?.Delta,
                };
            }
            if (typeof(T) == typeof(Nothing))
            {
                return default!;
            }
            if (text.Length == 0)
            {
                throw new RouterException((int)response.StatusCode, operation, "the router answered with nothing");
            }
            try
            {
                return JsonSerializer.Deserialize<T>(text, Json.Options)
                    ?? throw new RouterException((int)response.StatusCode, operation, "the router answered null");
            }
            catch (JsonException failure)
            {
                throw new RouterException((int)response.StatusCode, operation, $"the answer could not be read: {failure.Message}", failure);
            }
        }
    }

    internal static string Escape(string segment) => Uri.EscapeDataString(segment);

    internal static string? Blank(string? value) => string.IsNullOrEmpty(value) ? null : value;

    /// <summary>What the router said went wrong, from the error body every refusal shares.</summary>
    private static string Said(string text, HttpStatusCode status)
    {
        try
        {
            if (JsonSerializer.Deserialize<Error>(text, Json.Options) is { Error1: { Length: > 0 } said })
            {
                return said;
            }
        }
        catch (JsonException)
        {
            // Not the error shape, so what arrived is reported as it is below.
        }
        return text.Trim() is { Length: > 0 } raw ? raw : $"the router answered {(int)status}";
    }

    private static string QueryString(IReadOnlyDictionary<string, string?>? query)
    {
        if (query is null)
        {
            return "";
        }
        var set = query.Where(pair => !string.IsNullOrEmpty(pair.Value)).ToList();
        if (set.Count == 0)
        {
            return "";
        }
        return "?" + string.Join('&', set.Select(pair =>
            $"{Uri.EscapeDataString(pair.Key)}={Uri.EscapeDataString(pair.Value!)}"));
    }

    internal static string? Number(int? value) =>
        value is > 0 ? value.Value.ToString(CultureInfo.InvariantCulture) : null;

    internal sealed class Nothing;
}

/// <summary>Who a guest is, as far as anybody knows yet.</summary>
public sealed record GuestOptions
{
    /// <summary>A guest to reuse, for somebody coming back. Null mints a new one.</summary>
    public string? Id { get; init; }

    /// <summary>What to call them, for a transcript a person reads later.</summary>
    public string? Name { get; init; }

    /// <summary>Anything of the caller's own to keep against them.</summary>
    public Dictionary<string, object?>? Custom { get; init; }
}
