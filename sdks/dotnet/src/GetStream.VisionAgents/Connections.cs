using System.Globalization;
using GetStream.VisionAgents.Models;

namespace GetStream.VisionAgents;

/// <summary>What <see cref="Connections.CreateAsync"/> asks beside the connector.</summary>
public sealed record NewConnectionOptions
{
    /// <summary>Tells the owner's connections to the same connector apart. Null is none.</summary>
    public string? Label { get; init; }

    /// <summary>The app's user the connection is for. Null is the app's own.</summary>
    public string? UserId { get; init; }
}

/// <summary>What <see cref="Connections.AuthorizeAsync"/> asks beside the connection.</summary>
public sealed record AuthorizeOptions
{
    /// <summary>
    /// The app's page the browser goes back to when the consent ends, with <c>connection_id</c>
    /// and <c>status</c> added to its query. Null goes back to the dashboard.
    /// </summary>
    public string? ReturnUrl { get; init; }

    /// <summary>The user the connection belongs to. Null is an app connection.</summary>
    public string? UserId { get; init; }
}

/// <summary>Which connections <see cref="Connections.ListAsync"/> reads.</summary>
public sealed record ConnectionFilter
{
    /// <summary>Keeps one connector's. Null is every connector's.</summary>
    public string? ConnectorId { get; init; }

    /// <summary>Lists that user's connections. Null lists the app's own.</summary>
    public string? UserId { get; init; }
}

/// <summary>
/// The accounts the app and its users hold at connectors, such as a user's Linear: what an
/// agent's tools reach that account through. Only a backend may ask.
/// </summary>
/// <remarks>
/// Every method takes the user, named once per call: a <c>userId</c> parameter, or the
/// <c>UserId</c> of the options. A user given is sent as the
/// owner and as <c>X-Stream-User-Id</c> alike, so the two never differ. A user's connection is
/// reached only by a call naming that user; the app's is reached by every call, one naming a
/// user included. No user is the app's own connection. A user given replaces the one the client
/// was made for, with the server credential; a client holding a user's token (a device) refuses
/// a user with a <see cref="ConfigurationException"/> before sending.
/// </remarks>
public sealed class Connections
{
    // The page size ListAsync reads in, the most the router answers (listConnections' limit).
    private const int Page = 200;

    private readonly VisionAgentsClient _client;

    internal Connections(VisionAgentsClient client) => _client = client;

    /// <summary>Stores a pending connection to a connector, owned by the user or else by the app.</summary>
    /// <param name="connectorId">A built-in, such as <c>linear</c>, or one of the app's own.</param>
    /// <param name="options">The label, and the app's user the connection is for. Null is the app's own, with no label.</param>
    /// <param name="cancellationToken">Stops waiting.</param>
    public Task<Connection> CreateAsync(string connectorId, NewConnectionOptions? options = null, CancellationToken cancellationToken = default)
    {
        var userId = VisionAgentsClient.Blank(options?.UserId);
        var request = new ConnectionRequest
        {
            ConnectorId = connectorId,
            Label = VisionAgentsClient.Blank(options?.Label),
            Owner = userId is null ? new ConnectionOwner { Type = "app" } : new ConnectionOwner { Type = "user", UserId = userId },
        };
        return SendAsync(userId, client => client.PostAsync<Connection>("/v1/agents/connections", request, cancellationToken));
    }

    /// <summary>
    /// Starts the provider's consent for a connection: a first one for a pending connection, a
    /// reconnect for one connected before. Open the <c>LaunchUrl</c> it returns in a popup from a
    /// page of the return URL's origin and post it the <c>HandoffToken</c>; the attempt ends at
    /// <c>ExpiresAt</c>.
    /// </summary>
    /// <param name="id">The connection.</param>
    /// <param name="options">The return URL, and the user the connection belongs to. Null is an app connection going back to the dashboard.</param>
    /// <param name="cancellationToken">Stops waiting.</param>
    /// <exception cref="ConfigurationException">The id is empty.</exception>
    public Task<Authorization> AuthorizeAsync(string id, AuthorizeOptions? options = null, CancellationToken cancellationToken = default) =>
        SendAsync(options?.UserId, client => client.PostAsync<Authorization>($"{Path(id)}/authorizations",
            new AuthorizationRequest { ReturnUrl = VisionAgentsClient.Blank(options?.ReturnUrl) }, cancellationToken));

    /// <summary>
    /// One connection of the user's, or of the app's when no user is given. Another user's is
    /// not found, the same as one that does not exist; the app's is reached with any user given.
    /// </summary>
    /// <exception cref="ConfigurationException">The id is empty.</exception>
    public Task<Connection> GetAsync(string id, string? userId = null, CancellationToken cancellationToken = default) =>
        SendAsync(userId, client => client.GetAsync<Connection>(Path(id), cancellationToken: cancellationToken));

    /// <summary>
    /// Every connection of the user's, or of the app's when no user is given, newest first, read
    /// page by page. With a user it holds only that user's connections, not the app's.
    /// </summary>
    /// <param name="filter">The connector to keep, and the user to list. Null lists the app's own, of every connector.</param>
    /// <param name="cancellationToken">Stops waiting.</param>
    public Task<List<Connection>> ListAsync(ConnectionFilter? filter = null, CancellationToken cancellationToken = default)
    {
        var userId = VisionAgentsClient.Blank(filter?.UserId);
        return SendAsync(userId, async client =>
        {
            var query = new Dictionary<string, string?>
            {
                ["owner_type"] = userId is null ? "app" : "user",
                ["connector_id"] = filter?.ConnectorId,
                ["limit"] = Page.ToString(CultureInfo.InvariantCulture),
            };
            var connections = new List<Connection>();
            while (true)
            {
                var page = await client.GetAsync<ConnectionPage>("/v1/agents/connections", query, cancellationToken).ConfigureAwait(false);
                connections.AddRange(page.Items ?? []);
                if (!page.HasMore || string.IsNullOrEmpty(page.NextCursor))
                {
                    return connections;
                }
                query["cursor"] = page.NextCursor;
            }
        });
    }

    /// <summary>
    /// Disconnects a connection of the user's, or of the app's when no user is given, and drops
    /// its credentials. One an agent config binds as its fixed connection is refused with a 409.
    /// </summary>
    /// <exception cref="ConfigurationException">The id is empty.</exception>
    public Task DeleteAsync(string id, string? userId = null, CancellationToken cancellationToken = default) =>
        SendAsync<VisionAgentsClient.Nothing>(userId, async client =>
        {
            await client.DeleteAsync(Path(id), cancellationToken).ConfigureAwait(false);
            return null!;
        });

    /// <summary>
    /// One connection's path. An empty id is refused before anything is sent: it would ask for
    /// <c>/v1/agents/connections/</c>, which names no connection.
    /// </summary>
    private static string Path(string id) => string.IsNullOrEmpty(id)
        ? throw new ConfigurationException("a connection needs its id")
        : $"/v1/agents/connections/{VisionAgentsClient.Escape(id)}";

    /// <summary>
    /// Sends with this client acting for the user, or as it is when no user is given, whoever it
    /// acts for otherwise.
    /// </summary>
    /// <remarks>
    /// A user given replaces the one the client was made for (<see cref="VisionAgentsOptions.UserId"/>)
    /// and the request keeps the server credential, never a token minted for that one user: the router
    /// answers a user token as the device it is, and a device may not name another user.
    /// </remarks>
    private async Task<T> SendAsync<T>(string? userId, Func<VisionAgentsClient, Task<T>> send)
    {
        userId = VisionAgentsClient.Blank(userId);
        if (userId is null)
        {
            return await send(_client).ConfigureAwait(false);
        }
        // Before anything is sent, so the reason is given rather than the router's 403.
        if (!_client.ServerSide)
        {
            throw new ConfigurationException(
                "a connections call names a user only with a server credential: this client holds a user's token, "
                + "which cannot act for another user; leave the user out or use the app's own client");
        }
        using var acting = _client.ActingFor(userId);
        return await send(acting).ConfigureAwait(false);
    }
}
