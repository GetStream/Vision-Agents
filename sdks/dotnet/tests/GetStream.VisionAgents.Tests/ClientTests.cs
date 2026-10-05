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
        router.On("POST", "/v1/agents/sessions/query", 200, Page());
        using var server = new VisionAgentsClient(new VisionAgentsOptions { Url = router.Url, ApiKey = "key", ApiSecret = Fixtures.Secret });

        var user = server.AsUser("ada", "user-token");
        await user.Sessions.QueryAsync(cancellationToken: TestContext.Current.CancellationToken);

        var seen = router.Only("POST", "/v1/agents/sessions/query");
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
        router.On("POST", "/v1/agents/sessions/query", 200, Page());
        using var client = new VisionAgentsClient(new VisionAgentsOptions { Url = router.Url, ApiKey = "key", ApiSecret = Fixtures.Secret });
        var cancel = TestContext.Current.CancellationToken;

        var guest = await client.GuestUserAsync(new GuestOptions { Name = "Visitor" }, cancel);
        await client.AsGuest(guest).Sessions.QueryAsync(cancellationToken: cancel);
        await client.ClaimGuestUserAsync(guest.Id, "ada", cancel);

        Assert.Equal("Visitor", router.Only("POST", "/v1/agents/guests").Body.Text("name"));
        Assert.Equal("Bearer guest-token", router.Only("POST", "/v1/agents/sessions/query").Headers["Authorization"]);
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
    public async Task SessionsAreQueriedAndSearchedWithTheFilterTheRouterReads()
    {
        await using var router = await TestRouter.StartAsync();
        var queried = 0;
        router.On("POST", "/v1/agents/sessions/query", _ => new Reply(200, ++queried == 1
            ? Page(true, "c2", Fixtures.Session("s1"))
            : Page(false, null, Fixtures.Session("s2"))));
        router.On("GET", "/v1/agents/sessions/s1", 200, Fixtures.Session("s1"));
        using var client = Fixtures.Client(router);
        var cancel = TestContext.Current.CancellationToken;

        var listed = await client.Agent("support").Sessions.QueryAsync(new SessionQuery
        {
            UserId = "u1", State = "live", AgentId = "jean", Modality = "text", Limit = 10, Cursor = "c1",
        }, cancel);
        var found = await client.Sessions.SearchAsync("billing", new SessionQuery { UserId = "u1" }, cancel);
        var read = await client.Sessions.GetAsync("s1", cancel);

        Assert.Equal(("s1", true, "c2"), (Assert.Single(listed.Items).Id, listed.HasMore, listed.NextCursor));
        Assert.Equal(("s2", false), (Assert.Single(found.Items).Id, found.HasMore));
        Assert.False(read.Live);
        var bodies = router.To("POST", "/v1/agents/sessions/query").Select(seen => seen.Body!.AsObject()).ToList();
        Assert.Equal(["cursor", "filter", "limit"], bodies[0].Select(pair => pair.Key).Order());
        Assert.Equal(("c1", 10), (bodies[0].Text("cursor"), bodies[0]["limit"]!.GetValue<int>()));
        Assert.Equal("""{"agent":"support","agent_id":"jean","modality":"text","state":"live","user_id":"u1"}""",
            bodies[0]["filter"]!.ToJsonString());
        Assert.Equal("""{"text":{"$q":"billing"},"user_id":"u1"}""", bodies[1]["filter"]!.ToJsonString());
    }

    [Fact]
    public async Task ASessionIsUpdatedStoppedAndDeletedByItsId()
    {
        await using var router = await TestRouter.StartAsync();
        router.On("PATCH", "/v1/agents/sessions/s1", 200, Fixtures.Session("s1"));
        router.On("DELETE", "/v1/agents/sessions/s1", 204);
        router.On("DELETE", "/v1/agents/sessions/s1/memories", 204);
        router.On("DELETE", "/v1/agents/users/ada/memories", 204);
        using var client = Fixtures.Client(router);
        var cancel = TestContext.Current.CancellationToken;

        var updated = await client.Agent("support").Sessions.UpdateAsync("s1", new UpdateSessionRequest { Title = "Pricing" }, cancel);
        await client.Sessions.DeleteMemoriesAsync("s1", cancel);
        await client.Sessions.DeleteAsync("s1", cancel);
        await client.Memories.TruncateAsync("ada", cancel);

        Assert.Equal("s1", updated.Id);
        Assert.Equal("""{"title":"Pricing"}""", router.Only("PATCH", "/v1/agents/sessions/s1").Body!.ToJsonString());
        router.Only("DELETE", "/v1/agents/sessions/s1/memories");
        router.Only("DELETE", "/v1/agents/sessions/s1");
        router.Only("DELETE", "/v1/agents/users/ada/memories");
        await Assert.ThrowsAsync<ConfigurationException>(() => client.Memories.TruncateAsync("", cancel));
    }

    [Fact]
    public async Task SimulationsAreWrittenRunAndReadBack()
    {
        await using var router = await TestRouter.StartAsync();
        router.On("POST", "/v1/agents/simulations", 201, Simulation());
        router.On("GET", "/v1/agents/simulations", 200, new[] { Simulation() });
        router.On("GET", "/v1/agents/simulations/sim-1", 200, Simulation());
        router.On("PUT", "/v1/agents/simulations/sim-1", 200, Simulation());
        router.On("DELETE", "/v1/agents/simulations/sim-1", 204);
        router.On("POST", "/v1/agents/simulations/sim-1/run", 202, Run("running"));
        router.On("GET", "/v1/agents/simulation-runs/run-1", 200, Run("passed"));
        router.On("GET", "/v1/agents/simulation-runs", 200, new[] { Run("passed") });
        router.On("POST", "/v1/agents/simulation-runs/run-1/cancel", 200, Run("cancelled"));
        using var client = Fixtures.Client(router);
        var cancel = TestContext.Current.CancellationToken;
        var asked = new SimulationRequest { Name = "lunch", ConfigId = "cfg-1", Scenario = "Order lunch.", Assertion = "One wrap." };

        var simulation = await client.Simulations.CreateAsync(asked, cancel);
        Assert.Single(await client.Simulations.ListAsync(cancel));
        await client.Simulations.GetAsync(simulation.Id, cancel);
        await client.Simulations.UpdateAsync(simulation.Id, asked, cancel);
        var run = await client.Simulations.RunAsync(simulation.Id, cancel);
        var finished = await client.Simulations.Runs.GetAsync(run.Id, cancel);
        Assert.Single(await client.Simulations.Runs.ListAsync(new SimulationRunQuery { SimulationId = "sim-1", Limit = 5 }, cancel));
        var cancelled = await client.Simulations.Runs.CancelAsync(run.Id, cancel);
        await client.Simulations.DeleteAsync(simulation.Id, cancel);

        Assert.Equal(("running", "passed", "cancelled"), (run.State, finished.State, cancelled.State));
        Assert.Equal(("lunch", "cfg-1"), (router.Only("POST", "/v1/agents/simulations").Body.Text("name"), router.Only("PUT", "/v1/agents/simulations/sim-1").Body.Text("config_id")));
        var runs = router.Only("GET", "/v1/agents/simulation-runs").Query;
        Assert.Equal(("sim-1", "5"), (runs["simulation_id"], runs["limit"]));
        router.Only("DELETE", "/v1/agents/simulations/sim-1");
    }

    [Fact]
    public async Task AConversationIsUnwoundAPageAtATime()
    {
        await using var router = await TestRouter.StartAsync();
        var items = Enumerable.Range(0, 5).Select(index => new { response_id = $"r{index}", kind = "said", text = $"{index}" }).ToArray();
        router.On("GET", "/v1/agents/sessions/s1/responses/items", seen =>
        {
            var at = seen.Query.TryGetValue("cursor", out var cursor) ? int.Parse(cursor) : 0;
            var next = at + int.Parse(seen.Query["limit"]);
            return new Reply(200, new { items = items.Skip(at).Take(next - at), has_more = next < items.Length, next_cursor = next < items.Length ? $"{next}" : null });
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
        router.On("GET", "/v1/agents/sessions/s1/responses/items", 200, new { items = new[] { new { response_id = "r1", kind = "said", text = "hi" } }, has_more = false });
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
        router.On("GET", "/v1/agents/sessions/s1/responses", 200, new { items = new[] { new { id = "r1", session_id = "s1", status = "completed" } }, has_more = false });
        router.On("POST", "/v1/agents/sessions/s1/rewind", 204);
        router.On("POST", "/v1/agents/sessions/kept/rewind", 400, new { error = "a conversation kept in Stream Chat cannot be rewound; fork it at the response instead" });
        using var client = Fixtures.Client(router);
        var cancel = TestContext.Current.CancellationToken;

        var responses = client.Sessions.Responses("s1");
        var kept = Assert.Single((await responses.ListAsync(cancellationToken: cancel)).Items);
        await responses.RewindAsync(kept.Id, cancel);

        Assert.Equal("r1", router.Only("POST", "/v1/agents/sessions/s1/rewind").Body.Text("response_id"));
        var refused = await Assert.ThrowsAsync<RouterException>(() => client.Sessions.Responses("kept").RewindAsync("r1", cancel));
        Assert.Equal(400, refused.Status);
        await Assert.ThrowsAsync<ConfigurationException>(() => responses.RewindAsync("", cancel));
    }

    [Fact]
    public async Task ARecordIsForkedAtAResponseAndClosedByStoppingIt()
    {
        await using var router = await TestRouter.StartAsync();
        router.On("GET", "/v1/agents/sessions/s1", 200, Fixtures.Session("s1"));
        router.On("POST", "/v1/agents/sessions/s1/fork", 201, Fixtures.Session("s2"));
        router.On("POST", "/v1/agents/sessions/s1/stop", 204);
        using var client = Fixtures.Client(router);
        var cancel = TestContext.Current.CancellationToken;

        await using var parent = await client.Sessions.GetAsync("s1", cancel);
        var forked = await parent.ForkAsync(new ForkOptions { ResponseId = "r1", Title = "again", ProjectId = "docs" }, cancel);
        await parent.CloseAsync(cancel);

        Assert.Equal("s2", forked.Id);
        var fork = router.Only("POST", "/v1/agents/sessions/s1/fork").Body;
        Assert.Equal(("r1", "again", "docs"), (fork.Text("response_id"), fork.Text("title"), fork.Text("project_id")));
        router.Only("POST", "/v1/agents/sessions/s1/stop");
        Assert.Empty(router.To("DELETE", "/v1/agents/sessions/s1"));
    }

    [Fact]
    public async Task ASessionIsChangedWithOnlyWhatWasSet()
    {
        await using var router = await TestRouter.StartAsync();
        router.On("GET", "/v1/agents/sessions/s1", 200, Fixtures.Session("s1"));
        router.On("PATCH", "/v1/agents/sessions/s1", 200, Fixtures.Session("s1"));
        router.On("DELETE", "/v1/agents/sessions/s1", 204);
        router.On("DELETE", "/v1/agents/sessions/s1/memories", 204);
        router.On("POST", "/v1/agents/sessions/s1/stop", 204);
        using var client = Fixtures.Client(router);
        var cancel = TestContext.Current.CancellationToken;

        await using var session = await client.Sessions.GetAsync("s1", cancel);
        var updated = await session.UpdateAsync(new UpdateSessionRequest { Llm = "llm-thinking", Thinking = "high" }, cancel);
        await session.DeleteMemoriesAsync(cancel);
        await session.DeleteAsync(cancel);

        Assert.Equal("s1", updated.Id);
        var body = router.Only("PATCH", "/v1/agents/sessions/s1").Body!.AsObject();
        Assert.Equal(["llm", "thinking"], body.Select(pair => pair.Key).Order());
        Assert.Equal(("llm-thinking", "high"), (body.Text("llm"), body.Text("thinking")));
        router.Only("DELETE", "/v1/agents/sessions/s1/memories");
        router.Only("DELETE", "/v1/agents/sessions/s1");
    }

    [Fact]
    public async Task TheBodyIsWrittenWithTheRouterSpelling()
    {
        await using var router = await TestRouter.StartAsync();
        router.On("POST", "/v1/agents/sessions", 201, Fixtures.Session());
        using var client = Fixtures.Client(router);

        await client.PostAsync<Models.Session>("/v1/agents/sessions",
            new CreateSessionRequest { ConversationId = "agent:1", Incognito = true }, TestContext.Current.CancellationToken);

        var body = router.Only("POST", "/v1/agents/sessions").Body!.AsObject();
        Assert.Equal(["conversation_id", "incognito"], body.Select(pair => pair.Key).Order());
        Assert.Equal(JsonValueKind.True, body["incognito"]!.GetValueKind());
    }

    [Fact]
    public void APolicyKeepsAnAbsentModelListApartFromAnEmptyOne()
    {
        var absent = JsonSerializer.Deserialize<Policy>("{}", Json.Options)!;
        var none = JsonSerializer.Deserialize<Policy>("""{"allowed_models":[],"tags":{"team":"a"}}""", Json.Options)!;

        Assert.Null(absent.AllowedModels);
        Assert.Empty(none.AllowedModels!);
        Assert.Equal("{}", JsonSerializer.Serialize(absent, Json.Options));
        Assert.Equal("""{"allowed_models":[],"tags":{"team":"a"}}""", JsonSerializer.Serialize(none, Json.Options));
    }

    private static object Page(bool hasMore = false, string? next = null, params object[] items) =>
        new { items, has_more = hasMore, next_cursor = next };

    private static object Simulation() => new
    {
        id = "sim-1", config_id = "cfg-1", name = "lunch", scenario = "Order lunch.", assertion = "One wrap.", mode = "text",
        variations = 1, max_turns = 8, created_at = "2026-09-24T10:00:00Z", updated_at = "2026-09-24T10:00:00Z",
    };

    private static object Run(string state) => new
    {
        id = "run-1", simulation_id = "sim-1", state, created_at = "2026-09-24T10:00:00Z",
    };
}
