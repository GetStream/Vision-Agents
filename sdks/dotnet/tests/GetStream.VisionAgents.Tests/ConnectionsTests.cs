using GetStream.VisionAgents.Models;

namespace GetStream.VisionAgents.Tests;

public sealed class ConnectionsTests
{
    private const string Header = "X-Stream-User-Id";

    [Fact]
    public async Task AUserGivenIsTheOwnerAndTheHeaderOnEveryCallReadsIncluded()
    {
        await using var router = await Answering();
        using var client = Fixtures.Client(router);
        var connections = client.Connections;
        var cancel = TestContext.Current.CancellationToken;

        var created = await connections.CreateAsync("linear", new NewConnectionOptions { Label = "work", UserId = "alice" }, cancel);
        var consent = await connections.AuthorizeAsync(created.Id, new AuthorizeOptions { ReturnUrl = "https://app.example.com/settings", UserId = "alice" }, cancel);
        var got = await connections.GetAsync(created.Id, "alice", cancel);
        var listed = await connections.ListAsync(new ConnectionFilter { ConnectorId = "linear", UserId = "alice" }, cancel);
        await connections.DeleteAsync(created.Id, "alice", cancel);

        Assert.Equal(("conn-1", "pending"), (got.Id, got.Status));
        Assert.Equal(("https://router.example.com/launch", "handoff-1"), (consent.LaunchUrl, consent.HandoffToken));
        Assert.Equal(["conn-1"], listed.Select(connection => connection.Id));
        Assert.Equal(5, router.Requests.Count);
        Assert.All(router.Requests, seen => Assert.Equal(("alice", "examples"), (seen.Headers[Header], seen.Headers["X-Customer-Id"])));
        Assert.Equal("""{"connector_id":"linear","label":"work","owner":{"type":"user","user_id":"alice"}}""",
            router.Only("POST", "/v1/agents/connections").Body!.ToJsonString());
        Assert.Equal("""{"return_url":"https://app.example.com/settings"}""",
            router.Only("POST", "/v1/agents/connections/conn-1/authorizations").Body!.ToJsonString());
        Assert.Equal(("user", "linear", "200"), Query(router.Only("GET", "/v1/agents/connections")));
        router.Only("GET", "/v1/agents/connections/conn-1");
        router.Only("DELETE", "/v1/agents/connections/conn-1");
    }

    [Theory]
    [InlineData(null)]
    [InlineData("")]
    public async Task NoUserIsTheAppsOwnConnectionAndSendsNoHeader(string? userId)
    {
        await using var router = await Answering();
        using var client = Fixtures.Client(router);
        var connections = client.Connections;
        var cancel = TestContext.Current.CancellationToken;

        await connections.CreateAsync("linear", new NewConnectionOptions { UserId = userId }, cancel);
        await connections.AuthorizeAsync("conn-1", new AuthorizeOptions { UserId = userId }, cancel);
        await connections.GetAsync("conn-1", userId, cancel);
        await connections.ListAsync(new ConnectionFilter { UserId = userId }, cancel);
        await connections.DeleteAsync("conn-1", userId, cancel);

        Assert.Equal(5, router.Requests.Count);
        Assert.All(router.Requests, seen => Assert.False(seen.Headers.ContainsKey(Header)));
        Assert.Equal("""{"connector_id":"linear","owner":{"type":"app"}}""", router.Only("POST", "/v1/agents/connections").Body!.ToJsonString());
        Assert.Equal("{}", router.Only("POST", "/v1/agents/connections/conn-1/authorizations").Body!.ToJsonString());
        Assert.Equal(("app", "", "200"), Query(router.Only("GET", "/v1/agents/connections")));
    }

    [Fact]
    public async Task ABackendActingForAUserKeepsItsServerCredential()
    {
        await using var router = await Answering();
        var options = new VisionAgentsOptions { Url = router.Url, ApiKey = "key", ApiSecret = Fixtures.Secret };
        using var keyed = new VisionAgentsClient(options);
        using var proxied = new VisionAgentsClient(options with { Authenticate = true });
        var cancel = TestContext.Current.CancellationToken;

        await keyed.Connections.GetAsync("conn-1", "alice", cancel);
        await proxied.Connections.GetAsync("conn-1", "alice", cancel);

        var (direct, behindProxy) = (router.Requests[0], router.Requests[1]);
        Assert.Equal(("alice", "server"), (direct.Headers[Header], direct.Headers["Stream-Auth-Type"]));
        Assert.Equal(("alice", "jwt"), (behindProxy.Headers[Header], behindProxy.Headers["stream-auth-type"]));
        foreach (var seen in new[] { direct, behindProxy })
        {
            var claims = Fixtures.Claims(seen.Headers["Authorization"]);
            Assert.True(claims["server"]!.GetValue<bool>());
            Assert.Null(claims["user_id"]);
        }
    }

    [Fact]
    public async Task AListReadsEveryPageWithTheSameFilter()
    {
        await using var router = await TestRouter.StartAsync();
        // A third request is a list that never stops, answered with a refusal so it fails rather than hangs.
        router.On("GET", "/v1/agents/connections", seen => router.To("GET", "/v1/agents/connections").Count > 2
            ? new Reply(400, Fixtures.Failure("invalid_request", "too_many_pages", "read past the last page"))
            : seen.Query.GetValueOrDefault("cursor") switch
        {
            null => new Reply(200, new { items = new[] { Connection("conn-3"), Connection("conn-2") }, has_more = true, next_cursor = "after-2" }),
            "after-2" => new Reply(200, new { items = new[] { Connection("conn-1") }, has_more = false }),
            var other => new Reply(400, Fixtures.Failure("invalid_request", "invalid_cursor", $"no page after {other}")),
        });
        using var client = Fixtures.Client(router);

        var listed = await client.Connections.ListAsync(new ConnectionFilter { ConnectorId = "linear", UserId = "alice" }, TestContext.Current.CancellationToken);

        Assert.Equal(["conn-3", "conn-2", "conn-1"], listed.Select(connection => connection.Id));
        var pages = router.To("GET", "/v1/agents/connections");
        Assert.Equal(2, pages.Count);
        Assert.Equal("after-2", pages[1].Query["cursor"]);
        Assert.All(pages, page => Assert.Equal(("user", "linear", "200", "alice"), (page.Query["owner_type"], page.Query["connector_id"], page.Query["limit"], page.Headers[Header])));
    }

    [Theory]
    [InlineData("""{"items": [], "has_more": true}""")]
    [InlineData("""{"items": null, "has_more": true, "next_cursor": ""}""")]
    [InlineData("""{"has_more": false, "next_cursor": "after-1"}""")]
    public async Task AListStopsAtAPageThatNamesNoNextOne(string page)
    {
        await using var router = await TestRouter.StartAsync();
        // A second request is a page read past the last one, refused so the test fails rather than hangs.
        router.On("GET", "/v1/agents/connections", _ => router.To("GET", "/v1/agents/connections").Count == 1
            ? new Reply(200, page)
            : new Reply(400, Fixtures.Failure("invalid_request", "too_many_pages", "read past the last page")));
        using var client = Fixtures.Client(router);

        var listed = await client.Connections.ListAsync(cancellationToken: TestContext.Current.CancellationToken);

        Assert.Empty(listed);
        router.Only("GET", "/v1/agents/connections");
    }

    [Fact]
    public async Task AnEmptyConnectionIdIsRefusedBeforeAnythingIsSent()
    {
        await using var router = await Answering();
        using var client = Fixtures.Client(router);
        var connections = client.Connections;
        var cancel = TestContext.Current.CancellationToken;

        await Assert.ThrowsAsync<ConfigurationException>(() => connections.AuthorizeAsync("", new AuthorizeOptions { UserId = "alice" }, cancel));
        await Assert.ThrowsAsync<ConfigurationException>(() => connections.GetAsync("", "alice", cancel));
        await Assert.ThrowsAsync<ConfigurationException>(() => connections.DeleteAsync("", cancellationToken: cancel));

        Assert.Empty(router.Requests);
    }

    [Fact]
    public async Task AUserGivenReplacesTheOneTheClientWasMadeForAndNoneLeavesItAsItIs()
    {
        await using var router = await Answering();
        using var ada = new VisionAgentsClient(new VisionAgentsOptions { Url = router.Url, ApiKey = "key", ApiSecret = Fixtures.Secret, UserId = "ada" });
        var cancel = TestContext.Current.CancellationToken;

        await ada.Connections.GetAsync("conn-1", "alice", cancel);
        await ada.Connections.GetAsync("conn-1", "", cancel);
        await ada.Connections.GetAsync("conn-1", cancellationToken: cancel);

        Assert.Equal(["alice", "ada", "ada"], router.Requests.Select(seen => seen.Headers[Header]));
    }

    [Fact]
    public async Task EmptyOptionalStringsAreLeftOutAndTheOwnerIsTheUserGiven()
    {
        await using var router = await Answering();
        using var client = Fixtures.Client(router);
        var cancel = TestContext.Current.CancellationToken;

        await client.Connections.CreateAsync("linear", new NewConnectionOptions { Label = "", UserId = "bob" }, cancel);
        await client.Connections.AuthorizeAsync("conn-1", new AuthorizeOptions { ReturnUrl = "", UserId = "bob" }, cancel);

        Assert.Equal("""{"connector_id":"linear","owner":{"type":"user","user_id":"bob"}}""", router.Only("POST", "/v1/agents/connections").Body!.ToJsonString());
        Assert.Equal("{}", router.Only("POST", "/v1/agents/connections/conn-1/authorizations").Body!.ToJsonString());
    }

    [Fact]
    public async Task AFilterWithAConnectorAndNoUserListsTheAppsConnectionsOfThatConnector()
    {
        await using var router = await Answering();
        using var client = Fixtures.Client(router);

        await client.Connections.ListAsync(new ConnectionFilter { ConnectorId = "linear" }, TestContext.Current.CancellationToken);

        var list = router.Only("GET", "/v1/agents/connections");
        Assert.Equal(("app", "linear", "200"), Query(list));
        Assert.False(list.Headers.ContainsKey(Header));
    }

    [Fact]
    public async Task AConnectionIdIsOnePathSegment()
    {
        await using var router = await TestRouter.StartAsync();
        router.On("GET", "/v1/agents/connections/a%2Fb", 200, Connection("a/b"));
        using var client = Fixtures.Client(router);

        var got = await client.Connections.GetAsync("a/b", cancellationToken: TestContext.Current.CancellationToken);

        Assert.Equal("a/b", got.Id);
    }

    [Fact]
    public async Task AnotherUsersConnectionIsTheRoutersNotFound()
    {
        await using var router = await TestRouter.StartAsync();
        router.On("GET", "/v1/agents/connections/conn-1", seen => seen.Headers.GetValueOrDefault(Header) == "alice"
            ? new Reply(200, Connection("conn-1"))
            : new Reply(404, Fixtures.Failure("not_found", "connection_not_found", "no connection conn-1")));
        router.On("DELETE", "/v1/agents/connections/conn-1", 409, Fixtures.Failure("conflict", "connection_in_use", "an agent config binds it"));
        using var client = Fixtures.Client(router);
        var cancel = TestContext.Current.CancellationToken;

        var refused = await Assert.ThrowsAsync<RouterException>(() => client.Connections.GetAsync("conn-1", "bob", cancel));
        var bound = await Assert.ThrowsAsync<RouterException>(() => client.Connections.DeleteAsync("conn-1", "alice", cancel));

        Assert.Equal((404, "connection_not_found"), (refused.Status, refused.Code));
        Assert.Equal((409, "connection_in_use"), (bound.Status, bound.Code));
    }

    private static async Task<TestRouter> Answering()
    {
        var router = await TestRouter.StartAsync();
        router.On("POST", "/v1/agents/connections", 201, Connection("conn-1"));
        router.On("POST", "/v1/agents/connections/conn-1/authorizations", 201, new
        {
            id = "att-1",
            kind = "authorization",
            launch_url = "https://router.example.com/launch",
            handoff_token = "handoff-1",
            expires_at = "2026-11-19T12:10:00Z",
        });
        router.On("GET", "/v1/agents/connections/conn-1", 200, Connection("conn-1"));
        router.On("GET", "/v1/agents/connections", 200, new { items = new[] { Connection("conn-1") }, has_more = false });
        router.On("DELETE", "/v1/agents/connections/conn-1", 204);
        return router;
    }

    private static (string, string, string) Query(Seen seen) =>
        (seen.Query.GetValueOrDefault("owner_type") ?? "", seen.Query.GetValueOrDefault("connector_id") ?? "", seen.Query.GetValueOrDefault("limit") ?? "");

    private static object Connection(string id) => new
    {
        id,
        connector_id = "linear",
        status = "pending",
        auth_scheme = "oauth2_code",
        owner = new { type = "user", user_id = "alice" },
        created_at = "2026-11-19T12:00:00Z",
        updated_at = "2026-11-19T12:00:00Z",
    };
}
