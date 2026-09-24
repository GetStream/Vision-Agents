using System.Security.Cryptography;
using System.Text;
using System.Text.Json;

namespace GetStream.VisionAgents;

/// <summary>
/// Where the router is and who is calling it.
/// </summary>
/// <remarks>
/// Every field left null falls back to the environment the Go, Python and JavaScript SDKs
/// read, so a process deployed next to a router needs none of them.
/// </remarks>
public sealed record VisionAgentsOptions
{
    /// <summary>The router's base URL. Falls back to <c>STREAM_ACCELERATION_URL</c>, then localhost.</summary>
    public string? Url { get; init; }

    /// <summary>
    /// Who the work is billed to, taken at face value, for a router with nothing in front of
    /// it. Falls back to <c>STREAM_ACCELERATION_CUSTOMER_ID</c>.
    /// </summary>
    public string? CustomerId { get; init; }

    /// <summary>The public half of a Stream credential. Falls back to <c>STREAM_API_KEY</c>.</summary>
    public string? ApiKey { get; init; }

    /// <summary>
    /// The secret belonging to that key, which mints a server-side token. Falls back to
    /// <c>STREAM_API_SECRET</c>.
    /// </summary>
    public string? ApiSecret { get; init; }

    /// <summary>
    /// A token minted for <see cref="UserId"/> to hold, used instead of the secret.
    /// </summary>
    public string? Token { get; init; }

    /// <summary>
    /// The end user this client acts for. With the secret it is sent as
    /// <c>X-Stream-User-Id</c>, so the sessions it opens belong to that user.
    /// </summary>
    public string? UserId { get; init; }

    /// <summary>
    /// Whether the router is reached through Stream's authenticating proxy, which every
    /// hosted deployment sits behind. Falls back to <c>STREAM_ACCELERATION_AUTHENTICATE</c>.
    /// </summary>
    /// <remarks>
    /// Opt-in rather than inferred from holding a credential, because a Stream key and
    /// secret are in the environment for plenty of reasons that have nothing to do with how
    /// this router is reached.
    /// </remarks>
    public bool? Authenticate { get; init; }
}

/// <summary>
/// The resolved credential, and the headers it turns into.
/// </summary>
internal sealed class Backend
{
    public const string DefaultUrl = "http://localhost:8080";
    public const string UrlEnv = "STREAM_ACCELERATION_URL";
    public const string CustomerEnv = "STREAM_ACCELERATION_CUSTOMER_ID";
    public const string ApiKeyEnv = "STREAM_API_KEY";
    public const string ApiSecretEnv = "STREAM_API_SECRET";
    public const string AuthenticateEnv = "STREAM_ACCELERATION_AUTHENTICATE";

    // Short, because a token is minted per request and a stolen one should stop working
    // sooner than the credential behind it.
    private static readonly TimeSpan TokenValidity = TimeSpan.FromHours(1);

    public Backend(VisionAgentsOptions options)
    {
        Url = (options.Url ?? Env(UrlEnv) ?? DefaultUrl).TrimEnd('/');
        CustomerId = options.CustomerId ?? Env(CustomerEnv) ?? "";
        // Naming a customer is choosing the way a router with nothing in front of it is
        // reached, and a key that happens to be in the environment does not overrule that.
        ApiKey = options.ApiKey ?? (options.CustomerId is null ? Env(ApiKeyEnv) ?? "" : "");
        // A token handed in is the caller's answer to who they are, so an ambient secret
        // does not turn it into a backend.
        ApiSecret = options.ApiSecret ?? (options.Token is null && ApiKey != "" ? Env(ApiSecretEnv) ?? "" : "");
        Token = options.Token ?? "";
        UserId = options.UserId ?? "";
        Authenticate = options.Authenticate ?? Flag(Env(AuthenticateEnv));

        if (ApiKey == "" && CustomerId == "")
        {
            throw new ConfigurationException(
                $"who is calling is not set; pass CustomerId or {CustomerEnv} for a router that trusts one, "
                + $"or ApiKey with ApiSecret or Token");
        }
        if (ApiKey != "" && ApiSecret == "" && Token == "")
        {
            throw new ConfigurationException("ApiKey needs the secret it belongs to, or a token minted with it");
        }
    }

    public string Url { get; }
    public string CustomerId { get; }
    public string ApiKey { get; }
    public string ApiSecret { get; }
    public string Token { get; }
    public string UserId { get; }
    public bool Authenticate { get; }

    /// <summary>Whether this speaks for a process the customer runs rather than a device.</summary>
    public bool ServerSide => ApiSecret != "" || (ApiKey == "" && CustomerId != "");

    public Backend As(string userId, string token) => new(new VisionAgentsOptions
    {
        Url = Url,
        CustomerId = ApiKey == "" ? CustomerId : null,
        ApiKey = ApiKey == "" ? null : ApiKey,
        ApiSecret = "",
        Token = token,
        UserId = userId,
        Authenticate = Authenticate,
    });

    /// <summary>
    /// What every request and every socket handshake carries. Minted per call, so a client
    /// left idle longer than a token lasts does not wake up holding an expired one.
    /// </summary>
    public IReadOnlyList<KeyValuePair<string, string>> Headers()
    {
        if (ApiKey == "")
        {
            return [new("X-Customer-Id", CustomerId)];
        }

        if (Authenticate)
        {
            // jwt whoever the token is for: the proxy works out the caller from the token it
            // verified, and refuses a request that claims to be server.
            var proxied = Token != "" ? Token
                : UserId != "" ? SignToken(new() { ["user_id"] = UserId }, ApiSecret)
                : SignToken(new() { ["server"] = true }, ApiSecret);
            return
            [
                new("api_key", ApiKey),
                new("stream-auth-type", "jwt"),
                new("Authorization", $"Bearer {proxied}"),
            ];
        }

        if (ApiSecret != "")
        {
            List<KeyValuePair<string, string>> server =
            [
                new("X-Api-Key", ApiKey),
                new("Authorization", $"Bearer {SignToken(new() { ["server"] = true }, ApiSecret)}"),
                new("Stream-Auth-Type", "server"),
            ];
            if (UserId != "")
            {
                server.Add(new("X-Stream-User-Id", UserId));
            }
            return server;
        }

        return
        [
            new("X-Api-Key", ApiKey),
            new("Authorization", $"Bearer {Token}"),
            new("Stream-Auth-Type", "jwt"),
        ];
    }

    /// <summary>The WebSocket URL for a path on the router.</summary>
    /// <remarks>
    /// The credential travels in headers, which the router reads before the query string,
    /// so a token never ends up in a URL somebody's proxy logs.
    /// </remarks>
    public Uri SocketUrl(string path, IReadOnlyDictionary<string, string>? query = null)
    {
        var url = new StringBuilder(Url.StartsWith("https://", StringComparison.Ordinal)
            ? "wss://" + Url["https://".Length..]
            : Url.StartsWith("http://", StringComparison.Ordinal) ? "ws://" + Url["http://".Length..] : Url);
        url.Append(path);
        if (query is { Count: > 0 })
        {
            url.Append(path.Contains('?') ? '&' : '?');
            url.Append(string.Join('&', query.Select(pair =>
                $"{Uri.EscapeDataString(pair.Key)}={Uri.EscapeDataString(pair.Value)}")));
        }
        return new Uri(url.ToString());
    }

    /// <summary>Signs a Stream token, HS256.</summary>
    public static string SignToken(Dictionary<string, object> claims, string secret, TimeSpan? validity = null)
    {
        if (secret == "")
        {
            throw new ConfigurationException("a token cannot be signed without a secret");
        }

        var issued = DateTimeOffset.UtcNow.ToUnixTimeSeconds();
        var payload = new Dictionary<string, object>
        {
            ["iat"] = issued,
            ["exp"] = issued + (long)(validity ?? TokenValidity).TotalSeconds,
        };
        foreach (var (name, value) in claims)
        {
            payload[name] = value;
        }

        var signing = $"{Base64Url(JsonSerializer.SerializeToUtf8Bytes(new { alg = "HS256", typ = "JWT" }))}."
            + Base64Url(JsonSerializer.SerializeToUtf8Bytes(payload));
        var mac = HMACSHA256.HashData(Encoding.UTF8.GetBytes(secret), Encoding.UTF8.GetBytes(signing));
        return $"{signing}.{Base64Url(mac)}";
    }

    internal static string? Env(string name) =>
        Environment.GetEnvironmentVariable(name) is { Length: > 0 } value ? value : null;

    private static bool Flag(string? value) =>
        value == "1" || string.Equals(value, "true", StringComparison.OrdinalIgnoreCase);

    private static string Base64Url(byte[] bytes) =>
        Convert.ToBase64String(bytes).TrimEnd('=').Replace('+', '-').Replace('/', '_');
}
