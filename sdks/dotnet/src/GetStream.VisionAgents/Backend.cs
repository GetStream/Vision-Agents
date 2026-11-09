using System.Security.Cryptography;
using System.Text;
using System.Text.Json;
using System.Text.RegularExpressions;

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
    /// <summary>
    /// The router's base URL. Falls back to <see cref="Region"/>, then
    /// <c>STREAM_ACCELERATION_URL</c>, then <c>STREAM_ACCELERATION_REGION</c>, then Stream's
    /// hosted router, so only a self-hosted or local router needs it.
    /// </summary>
    public string? Url { get; init; }

    /// <summary>
    /// Holds this client to Stream's hosted router in one region, such as <c>us-east4</c>. It
    /// cannot be set together with <see cref="Url"/>.
    /// </summary>
    public string? Region { get; init; }

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
    /// hosted deployment sits behind. On for the hosted router, in any region; anywhere else it
    /// falls back to <c>STREAM_ACCELERATION_AUTHENTICATE</c>.
    /// </summary>
    /// <remarks>
    /// Away from the hosted router it is opt-in rather than inferred from holding a credential, because a Stream key and
    /// secret are in the environment for plenty of reasons that have nothing to do with how
    /// this router is reached.
    /// </remarks>
    public bool? Authenticate { get; init; }

    /// <summary>
    /// The customer's team this client's work is counted against, its spend and its spend
    /// limits, such as <c>finance</c>: 1 to 59 characters of A-Z, a-z, 0-9, <c>_ . -</c>.
    /// </summary>
    /// <remarks>
    /// It and <see cref="UsageCategories"/> are signed into every token this client mints, and
    /// sent as <c>X-Stream-Usage-Categories</c> to a router with no proxy in front of it. A
    /// <see cref="Token"/> handed in carries whatever it was signed with, so neither can be set
    /// beside one.
    /// </remarks>
    public string? Team { get; init; }

    /// <summary>
    /// The customer's own groupings this client's work is counted against, such as
    /// <c>project:onboarding</c>: at most 10 together with the team, each 1 to 64 characters of
    /// A-Z, a-z, 0-9, <c>_ . : -</c>, and none starting <c>team:</c>. Kept sorted, each once.
    /// </summary>
    public IReadOnlyList<string>? UsageCategories { get; init; }
}

/// <summary>The rules the router and the proxy hold a team and usage categories to.</summary>
internal static class UsageCategories
{
    /// <summary>
    /// Where a caller the proxy is not in front of names its categories, its team among them as
    /// <c>team:&lt;name&gt;</c>, joined with commas. Behind the proxy they are claims on the token
    /// instead, and the proxy deletes the header.
    /// </summary>
    public const string Header = "X-Stream-Usage-Categories";

    public const string TeamPrefix = "team:";
    public const int Max = 10;

    // A team is a category of its own, team:<name>, which is why its name is shorter and has no ':'.
    private static readonly Regex TeamName = new(@"^[A-Za-z0-9_.-]{1,59}\z", RegexOptions.CultureInvariant);
    private static readonly Regex CategoryName = new(@"^[A-Za-z0-9_.:-]{1,64}\z", RegexOptions.CultureInvariant);

    public static string CheckTeam(string team) => TeamName.IsMatch(team ?? "")
        ? team!
        : throw new ConfigurationException($"team \"{team}\" is not 1 to 59 characters of A-Z, a-z, 0-9, _ . and -");

    public static string CheckCategory(string category)
    {
        if (!CategoryName.IsMatch(category ?? ""))
        {
            throw new ConfigurationException($"usage category \"{category}\" is not 1 to 64 characters of A-Z, a-z, 0-9, _ . : and -");
        }
        // In any case, as the proxy reads it: Team:finance would be a second team.
        if (category!.StartsWith(TeamPrefix, StringComparison.OrdinalIgnoreCase))
        {
            throw new ConfigurationException($"usage category \"{category}\" names a team: use Team for a team");
        }
        return category;
    }

    /// <summary>The categories each once and sorted, as the proxy keeps them, with the team checked.</summary>
    public static string[] Checked(string? team, IEnumerable<string>? categories)
    {
        var kept = (categories ?? []).Select(CheckCategory).Distinct(StringComparer.Ordinal).Order(StringComparer.Ordinal).ToArray();
        if (team is not null)
        {
            CheckTeam(team);
        }
        var count = kept.Length + (team is null ? 0 : 1);
        if (count > Max)
        {
            throw new ConfigurationException($"at most {Max} usage categories, the team included; these are {count}");
        }
        return kept;
    }
}

/// <summary>
/// The resolved credential, and the headers it turns into.
/// </summary>
internal sealed class Backend
{
    /// <summary>Stream's hosted router, which is where a client goes when nothing says otherwise.</summary>
    public const string DefaultUrl = "https://accelerate-us-east4.gcp.stream-io-api.com";
    public const string UrlEnv = "STREAM_ACCELERATION_URL";
    /// <summary>Holds this SDK to the hosted router in one region, such as <c>us-east4</c>.</summary>
    public const string RegionEnv = "STREAM_ACCELERATION_REGION";
    public const string CustomerEnv = "STREAM_ACCELERATION_CUSTOMER_ID";
    public const string ApiKeyEnv = "STREAM_API_KEY";
    public const string ApiSecretEnv = "STREAM_API_SECRET";
    public const string AuthenticateEnv = "STREAM_ACCELERATION_AUTHENTICATE";

    // Short, because a token is minted per request and a stolen one should stop working
    // sooner than the credential behind it.
    private static readonly TimeSpan TokenValidity = TimeSpan.FromHours(1);

    // The shape of a region name, such as us-east4 or europe-west4. Only the shape is checked:
    // which regions are served is the hosted router's to say, so offering another needs no
    // release of this SDK. \z rather than $, which would let a trailing newline through.
    private static readonly Regex RegionName = new(@"^[a-z]+-[a-z]+[0-9]+\z", RegexOptions.CultureInvariant);

    public Backend(VisionAgentsOptions options) : this(options, Env)
    {
    }

    /// <summary>
    /// Reads the environment through <paramref name="env"/>, so a test can say what is in it
    /// without changing the process's own.
    /// </summary>
    internal Backend(VisionAgentsOptions options, Func<string, string?> env)
    {
        if (options.Url is not null && options.Region is not null)
        {
            throw new ConfigurationException("set a Url or a Region, not both");
        }
        var url = options.Url ?? (options.Region is { } region ? RegionUrl(region) : null);
        if (url is null && env(UrlEnv) is { } fromEnv)
        {
            url = fromEnv;
        }
        if (url is null && env(RegionEnv) is { } regionFromEnv)
        {
            url = RegionUrl(regionFromEnv);
        }
        Url = (url ?? DefaultUrl).TrimEnd('/');
        CustomerId = options.CustomerId ?? env(CustomerEnv) ?? "";
        // Naming a customer is choosing the way a router with nothing in front of it is
        // reached, and a key that happens to be in the environment does not overrule that.
        ApiKey = options.ApiKey ?? (options.CustomerId is null ? env(ApiKeyEnv) ?? "" : "");
        // A token handed in is the caller's answer to who they are, so an ambient secret
        // does not turn it into a backend.
        ApiSecret = options.ApiSecret ?? (options.Token is null && ApiKey != "" ? env(ApiSecretEnv) ?? "" : "");
        Token = options.Token ?? "";
        UserId = options.UserId ?? "";
        // The hosted router, in any region, is always behind the proxy.
        Authenticate = options.Authenticate ?? (Flag(env(AuthenticateEnv)) || Hosted(Url));
        Team = VisionAgentsClient.Blank(options.Team);
        Categories = UsageCategories.Checked(Team, options.UsageCategories);
        if (Token != "" && (Team is not null || Categories.Count > 0))
        {
            throw new ConfigurationException(
                "a token handed in carries the team and usage categories it was signed with; set neither beside it");
        }

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

    private Backend(Backend from)
    {
        Url = from.Url;
        CustomerId = from.CustomerId;
        ApiKey = from.ApiKey;
        ApiSecret = from.ApiSecret;
        Token = from.Token;
        UserId = from.UserId;
        Authenticate = from.Authenticate;
        Team = from.Team;
        Categories = from.Categories;
    }

    public string Url { get; }
    public string CustomerId { get; }
    public string ApiKey { get; }
    public string ApiSecret { get; }
    public string Token { get; }
    public string UserId { get; private init; }
    public bool Authenticate { get; }
    public string? Team { get; }

    /// <summary>The usage categories, sorted and each once, the team not among them.</summary>
    public IReadOnlyList<string> Categories { get; }

    /// <summary>
    /// The end user this server-side credential speaks for. Unlike <see cref="UserId"/> it
    /// keeps the server's token, so what it writes is answered by the model rather than
    /// handed to a dispatch worker.
    /// </summary>
    public string ActingFor { get; private init; } = "";

    /// <summary>Whether this speaks for a process the customer runs rather than a device.</summary>
    public bool ServerSide => ApiSecret != "" || (ApiKey == "" && CustomerId != "");

    public Backend Acting(string userId) => new(this) { UserId = "", ActingFor = userId };

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
        if (ActingFor != "")
        {
            return [.. new Backend(this).Headers(), new("X-Stream-User-Id", ActingFor)];
        }
        var headers = Credentials();
        if (!Authenticate && (Team is not null || Categories.Count > 0))
        {
            IEnumerable<string> named = Team is null ? Categories : [.. Categories, UsageCategories.TeamPrefix + Team];
            headers.Add(new(UsageCategories.Header, string.Join(',', named.Order(StringComparer.Ordinal))));
        }
        return headers;
    }

    private List<KeyValuePair<string, string>> Credentials()
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
                : UserId != "" ? SignToken(Claims(new() { ["user_id"] = UserId }), ApiSecret)
                : SignToken(Claims(new() { ["server"] = true }), ApiSecret);
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
                new("Authorization", $"Bearer {SignToken(Claims(new() { ["server"] = true }), ApiSecret)}"),
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

    /// <summary>The claims with the team and the categories, each only when there is one.</summary>
    private Dictionary<string, object> Claims(Dictionary<string, object> claims)
    {
        if (Team is not null)
        {
            claims["team"] = Team;
        }
        if (Categories.Count > 0)
        {
            claims["usage_categories"] = Categories;
        }
        return claims;
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

    /// <summary>
    /// The hosted router held to one region: requests to it are served there and nowhere else,
    /// even when that region is full or down.
    /// </summary>
    public static string RegionUrl(string name)
    {
        if (!RegionName.IsMatch(name))
        {
            throw new ConfigurationException($"\"{name}\" is not a region, such as us-east4");
        }
        return $"https://accelerate-{name}.gcp.stream-io-api.com";
    }

    // Whether url is the hosted router in some region, which is always behind the proxy.
    private static bool Hosted(string url)
    {
        const string prefix = "https://accelerate-", suffix = ".gcp.stream-io-api.com";
        return url.StartsWith(prefix, StringComparison.Ordinal)
            && url.EndsWith(suffix, StringComparison.Ordinal)
            && url.Length > prefix.Length + suffix.Length
            && RegionName.IsMatch(url[prefix.Length..^suffix.Length]);
    }

    internal static string? Env(string name) =>
        Environment.GetEnvironmentVariable(name) is { Length: > 0 } value ? value : null;

    private static bool Flag(string? value) =>
        value == "1" || string.Equals(value, "true", StringComparison.OrdinalIgnoreCase);

    private static string Base64Url(byte[] bytes) =>
        Convert.ToBase64String(bytes).TrimEnd('=').Replace('+', '-').Replace('/', '_');
}
