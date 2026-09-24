using System.Net;
using System.Net.Sockets;
using System.Text.Json;
using GetStream.VisionAgents.Models;

namespace GetStream.VisionAgents.Tests;

public sealed class ClientTests
{
    [Fact]
    public async Task ARouterThatTrustsACustomerIsToldWhichOne()
    {
        await using var router = await TestRouter.StartAsync();
        router.On("GET", "/v1/agents/configs", 200, Array.Empty<object>());
        using var client = Fixtures.Client(router);

        await client.GetAsync<List<AgentConfig>>("/v1/agents/configs", cancellationToken: TestContext.Current.CancellationToken);

        var seen = router.Only("GET", "/v1/agents/configs");
        Assert.Equal("examples", seen.Headers["X-Customer-Id"]);
        Assert.False(seen.Headers.ContainsKey("Authorization"));
        Assert.True(client.ServerSide);
    }

    [Fact]
    public async Task AKeyAndSecretSignAServerTokenPerRequest()
    {
        await using var router = await TestRouter.StartAsync();
        router.On("GET", "/v1/agents/configs", 200, Array.Empty<object>());
        using var client = new VisionAgentsClient(new VisionAgentsOptions { Url = router.Url, ApiKey = "key", ApiSecret = Fixtures.Secret });

        await client.GetAsync<List<AgentConfig>>("/v1/agents/configs", cancellationToken: TestContext.Current.CancellationToken);

        var seen = router.Only("GET", "/v1/agents/configs");
        Assert.Equal("key", seen.Headers["X-Api-Key"]);
        Assert.Equal("server", seen.Headers["Stream-Auth-Type"]);
        var claims = Fixtures.Claims(seen.Headers["Authorization"]);
        Assert.True(claims["server"]!.GetValue<bool>());
        Assert.True(claims["exp"]!.GetValue<long>() > DateTimeOffset.UtcNow.ToUnixTimeSeconds());
        Assert.True(client.ServerSide);
    }

    [Fact]
    public async Task AUserHoldsTheirOwnTokenAndIsNotServerSide()
    {
        await using var router = await TestRouter.StartAsync();
        router.On("GET", "/v1/agents/sessions", 200, Array.Empty<object>());
        using var server = new VisionAgentsClient(new VisionAgentsOptions { Url = router.Url, ApiKey = "key", ApiSecret = Fixtures.Secret });

        var user = server.AsUser("ada", "user-token");
        await user.Sessions.ListAsync(cancellationToken: TestContext.Current.CancellationToken);

        var seen = router.Only("GET", "/v1/agents/sessions");
        Assert.Equal("Bearer user-token", seen.Headers["Authorization"]);
        Assert.Equal("jwt", seen.Headers["Stream-Auth-Type"]);
        Assert.False(user.ServerSide);
    }

    [Fact]
    public async Task ARouterBehindTheProxyIsSentAJwtWhoeverTheCallerIs()
    {
        await using var router = await TestRouter.StartAsync();
        router.On("GET", "/v1/agents/configs", 200, Array.Empty<object>());
        using var client = new VisionAgentsClient(new VisionAgentsOptions
        {
            Url = router.Url, ApiKey = "key", ApiSecret = Fixtures.Secret, UserId = "ada", Authenticate = true,
        });

        await client.GetAsync<List<AgentConfig>>("/v1/agents/configs", cancellationToken: TestContext.Current.CancellationToken);

        var seen = router.Only("GET", "/v1/agents/configs");
        Assert.Equal("key", seen.Headers["api_key"]);
        Assert.Equal("jwt", seen.Headers["stream-auth-type"]);
        Assert.Equal("ada", Fixtures.Claims(seen.Headers["Authorization"]).Text("user_id"));
    }

    [Fact]
    public void AKeyWithNothingToProveItIsRefused()
    {
        Assert.Throws<ConfigurationException>(() => new VisionAgentsClient(new VisionAgentsOptions { Url = "http://x", ApiKey = "key" }));
    }

    [Fact]
    public async Task ARefusalSaysWhatTheRouterSaid()
    {
        await using var router = await TestRouter.StartAsync();
        router.On("POST", "/v1/agents/sessions", 400, new { error = "naming both agent and config_id is refused" });
        using var client = Fixtures.Client(router);

        var refused = await Assert.ThrowsAsync<RouterException>(() =>
            client.PostAsync<Models.Session>("/v1/agents/sessions", new CreateSessionRequest(), TestContext.Current.CancellationToken));

        Assert.Equal(400, refused.Status);
        Assert.Equal("POST /v1/agents/sessions", refused.Operation);
        Assert.Equal("naming both agent and config_id is refused", refused.Said);
    }

    [Fact]
    public async Task ARequestThatNeverArrivedIsStatusZero()
    {
        var listener = new TcpListener(IPAddress.Loopback, 0);
        listener.Start();
        var port = ((IPEndPoint)listener.LocalEndpoint).Port;
        listener.Stop();
        using var client = new VisionAgentsClient(new VisionAgentsOptions { Url = $"http://127.0.0.1:{port}", CustomerId = "examples" });

        var refused = await Assert.ThrowsAsync<RouterException>(() =>
            client.GetAsync<List<AgentConfig>>("/v1/agents/configs", cancellationToken: TestContext.Current.CancellationToken));

        Assert.Equal(0, refused.Status);
    }

    [Fact]
    public async Task AGuestIsMintedAndClaimedByTheBackend()
    {
        await using var router = await TestRouter.StartAsync();
        router.On("POST", "/v1/agents/guests", 200, new { id = "guest-1", token = "guest-token", name = "Visitor" });
        router.On("POST", "/v1/agents/guests/claim", 200, new { guest_id = "guest-1", user_id = "ada", sessions = 2 });
        router.On("GET", "/v1/agents/sessions", 200, Array.Empty<object>());
        using var client = new VisionAgentsClient(new VisionAgentsOptions { Url = router.Url, ApiKey = "key", ApiSecret = Fixtures.Secret });
        var cancel = TestContext.Current.CancellationToken;

        var guest = await client.GuestUserAsync(new GuestOptions { Name = "Visitor" }, cancel);
        await client.AsGuest(guest).Sessions.ListAsync(cancellationToken: cancel);
        await client.ClaimGuestUserAsync(guest.Id, "ada", cancel);

        Assert.Equal("Visitor", router.Only("POST", "/v1/agents/guests").Body.Text("name"));
        Assert.Equal("Bearer guest-token", router.Only("GET", "/v1/agents/sessions").Headers["Authorization"]);
        var claim = router.Only("POST", "/v1/agents/guests/claim").Body;
        Assert.Equal(("guest-1", "ada"), (claim.Text("guest_id"), claim.Text("user_id")));
    }

    [Fact]
    public async Task ADeviceCannotClaimAGuest()
    {
        await using var router = await TestRouter.StartAsync();
        using var client = new VisionAgentsClient(new VisionAgentsOptions { Url = router.Url, ApiKey = "key", ApiSecret = Fixtures.Secret });

        await Assert.ThrowsAsync<ConfigurationException>(() =>
            client.AsUser("guest-1", "guest-token").ClaimGuestUserAsync("guest-1", "ada", TestContext.Current.CancellationToken));
        Assert.Empty(router.Requests);
    }

    [Fact]
    public async Task SessionsAreListedAndSearchedWithTheQueryTheRouterReads()
    {
        await using var router = await TestRouter.StartAsync();
        router.On("GET", "/v1/agents/sessions", 200, new[] { Fixtures.Session("s1") });
        router.On("GET", "/v1/agents/sessions/search", 200, new[] { Fixtures.Session("s2") });
        router.On("GET", "/v1/agents/sessions/s1", 200, Fixtures.Session("s1"));
        using var client = Fixtures.Client(router);
        var cancel = TestContext.Current.CancellationToken;

        var listed = await client.Sessions.ListAsync(new SessionQuery
        {
            Agent = "support", State = "running", Custom = new() { ["suite"] = "unit" }, Limit = 10,
            CreatedAfter = new DateTimeOffset(2026, 9, 1, 0, 0, 0, TimeSpan.Zero),
        }, cancel);
        var found = await client.Sessions.SearchAsync("billing", new SessionQuery { Project = "docs" }, cancel);
        var read = await client.Sessions.GetAsync("s1", cancel);

        Assert.Equal("s1", Assert.Single(listed).Id);
        Assert.Equal("s2", Assert.Single(found).Id);
        Assert.False(read.Live);
        var query = router.Only("GET", "/v1/agents/sessions").Query;
        Assert.Equal("support", query["agent"]);
        Assert.Equal("running", query["state"]);
        Assert.Equal("""{"suite":"unit"}""", query["custom"]);
        Assert.Equal("10", query["limit"]);
        Assert.StartsWith("2026-09-01T00:00:00", query["created_after"]);
        Assert.False(query.ContainsKey("offset"));
        var search = router.Only("GET", "/v1/agents/sessions/search").Query;
        Assert.Equal(("billing", "docs"), (search["q"], search["project"]));
    }

    [Fact]
    public async Task AConversationIsUnwoundAPageAtATime()
    {
        await using var router = await TestRouter.StartAsync();
        var items = Enumerable.Range(0, 5).Select(index => new { response_id = $"r{index}", kind = "said", text = $"{index}" }).ToArray();
        router.On("GET", "/v1/agents/sessions/s1/responses/items", seen =>
        {
            var offset = seen.Query.TryGetValue("offset", out var at) ? int.Parse(at) : 0;
            return new Reply(200, items.Skip(offset).Take(int.Parse(seen.Query["limit"])));
        });
        using var client = Fixtures.Client(router);

        var read = new List<AgentResponseItem>();
        await foreach (var item in client.Sessions.Responses("s1").Items.UnwindAsync(page: 2, TestContext.Current.CancellationToken))
        {
            read.Add(item);
        }

        Assert.Equal(["0", "1", "2", "3", "4"], read.Select(item => item.Text));
        Assert.Equal(3, router.To("GET", "/v1/agents/sessions/s1/responses/items").Count);
    }

    [Fact]
    public async Task ATurnIsNamedAndItsOwnItemsAreRead()
    {
        await using var router = await TestRouter.StartAsync();
        router.On("POST", "/v1/agents/sessions/s1/responses", 202, new { id = "r1", session_id = "s1", status = "running" });
        router.On("GET", "/v1/agents/sessions/s1/responses/items", 200, new[] { new { response_id = "r1", kind = "said", text = "hi" } });
        using var client = Fixtures.Client(router);
        var cancel = TestContext.Current.CancellationToken;

        var response = await client.Sessions.Responses("s1").CreateAsync("hi", cancellationToken: cancel);
        var items = await response.Items.AllAsync(cancel);

        Assert.Equal(("r1", "running"), (response.Id, response.Status));
        Assert.Equal("hi", router.Only("POST", "/v1/agents/sessions/s1/responses").Body.Text("text"));
        Assert.Equal("r1", router.Only("GET", "/v1/agents/sessions/s1/responses/items").Query["response_id"]);
        Assert.Equal("said", Assert.Single(items).Kind);
    }

    [Fact]
    public async Task ARewindNamesTheResponseToGoBackTo()
    {
        await using var router = await TestRouter.StartAsync();
        router.On("GET", "/v1/agents/sessions/s1/responses", 200, new[] { new { id = "r1", session_id = "s1", status = "completed" } });
        router.On("POST", "/v1/agents/sessions/s1/rewind", 204);
        router.On("POST", "/v1/agents/sessions/kept/rewind", 400, new { error = "a conversation kept in Stream Chat cannot be rewound; fork it at the response instead" });
        using var client = Fixtures.Client(router);
        var cancel = TestContext.Current.CancellationToken;

        var responses = client.Sessions.Responses("s1");
        var kept = Assert.Single(await responses.ListAsync(cancellationToken: cancel));
        await responses.RewindAsync(kept.Id, cancel);

        Assert.Equal("r1", router.Only("POST", "/v1/agents/sessions/s1/rewind").Body.Text("response_id"));
        var refused = await Assert.ThrowsAsync<RouterException>(() => client.Sessions.Responses("kept").RewindAsync("r1", cancel));
        Assert.Equal(400, refused.Status);
        await Assert.ThrowsAsync<ConfigurationException>(() => responses.RewindAsync("", cancel));
    }

    [Fact]
    public async Task ARecordIsForkedAtAResponseAndClosedByDeletingIt()
    {
        await using var router = await TestRouter.StartAsync();
        router.On("GET", "/v1/agents/sessions/s1", 200, Fixtures.Session("s1"));
        router.On("POST", "/v1/agents/sessions/s1/fork", 201, Fixtures.Session("s2"));
        router.On("DELETE", "/v1/agents/sessions/s1", 204);
        using var client = Fixtures.Client(router);
        var cancel = TestContext.Current.CancellationToken;

        await using var parent = await client.Sessions.GetAsync("s1", cancel);
        var forked = await parent.ForkAsync(new ForkOptions { ResponseId = "r1", Title = "again" }, cancel);
        await parent.CloseAsync(cancel);

        Assert.Equal("s2", forked.Id);
        var fork = router.Only("POST", "/v1/agents/sessions/s1/fork").Body;
        Assert.Equal(("r1", "again"), (fork.Text("response_id"), fork.Text("title")));
        router.Only("DELETE", "/v1/agents/sessions/s1");
    }

    [Fact]
    public async Task TheBodyIsWrittenWithTheRouterSpelling()
    {
        await using var router = await TestRouter.StartAsync();
        router.On("POST", "/v1/agents/sessions", 201, Fixtures.Session());
        using var client = Fixtures.Client(router);

        await client.PostAsync<Models.Session>("/v1/agents/sessions",
            new CreateSessionRequest { ConversationId = "agent:1", PersistConversation = true }, TestContext.Current.CancellationToken);

        var body = router.Only("POST", "/v1/agents/sessions").Body!.AsObject();
        Assert.Equal(["conversation_id", "persist_conversation"], body.Select(pair => pair.Key).Order());
        Assert.Equal(JsonValueKind.True, body["persist_conversation"]!.GetValueKind());
    }
}
