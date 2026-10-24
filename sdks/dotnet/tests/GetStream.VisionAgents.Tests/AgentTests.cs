using System.Text.Json.Nodes;
using GetStream.VisionAgents.Models;

namespace GetStream.VisionAgents.Tests;

public sealed class AgentTests : IDisposable
{
    private const string Events = "/v1/agents/sessions/s1/events";

    private readonly string _workspace = Path.Combine(Path.GetTempPath(), "va-agent-" + Guid.NewGuid().ToString("N"));

    public void Dispose()
    {
        if (Directory.Exists(_workspace))
        {
            Directory.Delete(_workspace, recursive: true);
        }
    }

    [Fact]
    public void TheAgentsConfigurationIsRenderedIntoTheSession()
    {
        using var client = new VisionAgentsClient(new VisionAgentsOptions { Url = "http://x", CustomerId = "examples" });
        var agent = new Agent(new AgentOptions
        {
            Config = "support",
            Name = "Jean Luc!",
            Instructions = "You are Jean.",
            Pipeline = new Pipeline { Llm = "llm-fast", Language = "fr", ToolTimeout = TimeSpan.FromSeconds(2) },
            Harness = new Harness { Subagents = new Dictionary<string, string> { ["default"] = "llm-thinking" }, Sandbox = Sandbox.Daytona() },
            CostTracking = new Dictionary<string, string> { ["team"] = "support" },
            MemoryFilter = new Dictionary<string, string> { [Agent.UserKey] = "ada", ["topic"] = "billing" },
            Client = client,
        });

        var request = agent.Request(null, false, new SessionOptions { Id = "billing_ada-1", Title = "Billing", ProjectId = "docs" }, null, false);

        Assert.Equal("jean-luc", agent.UserId);
        Assert.Equal(("support", null), (request.Agent, request.ConfigId));
        Assert.Null(request.StartVoice);
        Assert.Equal(("jean-luc", "Jean Luc!", "jean-luc"), (request.UserId, request.UserName, request.AgentId));
        Assert.Equal("llm-fast", request.Llm);
        Assert.Equal(["fr"], request.Languages!);
        Assert.Equal(2000, request.ToolTimeoutMs);
        Assert.Equal("support", request.Tags!["team"]);
        Assert.Equal("ada", request.Memory!.UserId);
        Assert.Equal(new Dictionary<string, string> { ["topic"] = "billing" }, request.Memory.Filter);
        Assert.Equal((null, "Billing"), (request.Incognito, request.Title));
        Assert.Equal(("billing_ada-1", "docs"), (request.Id, request.ProjectId));
    }

    [Fact]
    public async Task TheHarnessIsWrittenOntoTheConfigAndTheSessionRunsUnderIt()
    {
        await using var router = await TestRouter.StartAsync();
        router.On("POST", "/v1/agents/sync", 200, new { unchanged = false, config = Fixtures.Config("jean", "cfg-1") });
        router.On("POST", "/v1/agents/sessions", 201, Fixtures.Session());
        router.OnSocket(Events, peer => peer.ReceiveAsync("close"));
        using var client = Fixtures.Client(router);
        await using var agent = new Agent(new AgentOptions
        {
            Name = "jean",
            Harness = new Harness
            {
                Name = "default",
                Subagents = new Dictionary<string, string> { ["default"] = "llm-thinking" },
                Sandbox = Sandbox.Daytona(),
                Skills = [new Skill { Name = "think", Description = "Work it out", Instructions = "Reason.", Deadline = TimeSpan.FromSeconds(30) }],
            },
            Client = client,
        });
        var cancel = TestContext.Current.CancellationToken;

        await agent.SyncAsync(cancel);
        await agent.ChatAsync(cancellationToken: cancel);

        var synced = router.Only("POST", "/v1/agents/sync").Body;
        Assert.Equal(("default", "llm-thinking", "daytona"), (synced.Text("harness"), synced.Text("subagent"), synced.Text("sandbox")));
        Assert.Equal(30_000, synced!["skills"]![0]!["deadline_ms"]!.GetValue<long>());
        var opened = router.Only("POST", "/v1/agents/sessions").Body!.AsObject();
        Assert.Equal("cfg-1", opened.Text("config_id"));
        Assert.DoesNotContain(opened, pair => pair.Key is "subagent" or "sandbox" or "skills" or "tasks");
    }

    [Fact]
    public void AHarnessThatDoesNotExistIsRefused()
    {
        Assert.Throws<ConfigurationException>(() => new Agent(new AgentOptions
        {
            Name = "jean",
            Harness = new Harness { Name = "fancy" },
            Client = new VisionAgentsClient(new VisionAgentsOptions { Url = "http://x", CustomerId = "examples" }),
        }));
    }

    [Fact]
    public async Task TheConfigIsPatchedWithOnlyWhatWasSet()
    {
        await using var router = await TestRouter.StartAsync();
        router.On("GET", "/v1/agents/configs", 200, new[] { Fixtures.Config("support", "cfg-9") });
        router.On("PATCH", "/v1/agents/configs/cfg-9", 200, Fixtures.Config("support", "cfg-9"));
        using var client = Fixtures.Client(router);
        var cancel = TestContext.Current.CancellationToken;

        var config = await client.Agent("support").UpdateConfigAsync(new AgentConfigPatch { Guardrail = "No refunds.", VisibleTools = ["athena_*"] }, cancel);

        Assert.Equal("cfg-9", config.Id);
        Assert.Equal("support", router.Only("GET", "/v1/agents/configs").Query["name"]);
        var patch = router.Only("PATCH", "/v1/agents/configs/cfg-9").Body!.AsObject();
        Assert.Equal(["guardrail", "visible_tools"], patch.Select(pair => pair.Key).Order());
        await Assert.ThrowsAsync<ConfigurationException>(() => client.Agent("nowhere").UpdateConfigAsync(new AgentConfigPatch(), cancel));
    }

    [Fact]
    public async Task AReviewGuideIsPatchedAndReadBack()
    {
        const string guide = "Score how fast the refund was settled.";
        await using var router = await TestRouter.StartAsync();
        router.On("GET", "/v1/agents/configs", 200, new[] { Fixtures.Config("support", "cfg-9") });
        router.On("PATCH", "/v1/agents/configs/cfg-9", 200, new
        {
            id = "cfg-9",
            name = "support",
            mode = "voice",
            created_at = "2026-09-24T10:00:00Z",
            updated_at = "2026-09-24T10:00:00Z",
            review = new { instructions = guide },
        });
        using var client = Fixtures.Client(router);

        var config = await client.Agent("support").UpdateConfigAsync(new AgentConfigPatch { Review = new AgentReview { Instructions = guide } }, TestContext.Current.CancellationToken);

        Assert.Equal("""{"review":{"instructions":"Score how fast the refund was settled."}}""", router.Only("PATCH", "/v1/agents/configs/cfg-9").Body!.ToJsonString());
        Assert.Equal(guide, config.Review?.Instructions);
    }

    [Fact]
    public void AnIncognitoConversationAsksForNoTranscript()
    {
        using var client = new VisionAgentsClient(new VisionAgentsOptions { Url = "http://x", CustomerId = "examples" });
        var agent = new Agent(new AgentOptions { Name = "jean", Client = client });

        var request = agent.Request(null, false, new SessionOptions { Incognito = true }, null, false);

        Assert.True(request.Incognito);
    }

    [Fact]
    public void AConfigIsNamedOrGivenByIdButNotBoth()
    {
        using var client = new VisionAgentsClient(new VisionAgentsOptions { Url = "http://x", CustomerId = "examples" });

        var request = new Agent(new AgentOptions { ConfigId = "cfg-1", Client = client }).Request(null, false, null, null, false);

        Assert.Equal(("cfg-1", null), (request.ConfigId, request.Agent));
        Assert.Throws<ConfigurationException>(() => new Agent(new AgentOptions { Config = "docs", ConfigId = "cfg-1", Client = client }));
    }

    [Fact]
    public void AHarnessThatCannotDecideIsRefused()
    {
        Assert.Throws<ConfigurationException>(() => new Agent(new AgentOptions
        {
            Name = "jean",
            Harness = new Harness { Subagents = new Dictionary<string, string> { ["a"] = "x", ["b"] = "y" } },
            Client = new VisionAgentsClient(new VisionAgentsOptions { Url = "http://x", CustomerId = "examples" }),
        }));
    }

    [Fact]
    public async Task TheAgentHasNoResponsesBeforeItJoinsAnything()
    {
        await using var agent = new Agent(new AgentOptions
        {
            Name = "jean",
            Client = new VisionAgentsClient(new VisionAgentsOptions { Url = "http://x", CustomerId = "examples" }),
        });

        Assert.Throws<InvalidOperationException>(() => agent.Responses);
    }

    [Fact]
    public async Task JoiningOpensTheSessionWithVoiceOnItsOwnCall()
    {
        await using var router = await TestRouter.StartAsync();
        router.On("POST", "/v1/agents/sessions", 201, Fixtures.Session("s1", "s1"));
        router.OnSocket(Events, async peer =>
        {
            await peer.SendAsync(new { type = "joined" });
            await peer.ReceiveAsync("close");
        });
        using var client = Fixtures.Client(router);
        await using var agent = new Agent(new AgentOptions { Name = "Jean", Client = client, Edge = Fixtures.Edge(router) });
        var cancel = TestContext.Current.CancellationToken;

        var session = await agent.JoinAsync(cancellationToken: cancel);
        var first = await FirstAsync(session);
        await session.CloseAsync(cancel);

        Assert.Equal("joined", first.Kind);
        Assert.DoesNotContain(router.Requests, seen => seen.Path.StartsWith("/api/v2/video/call/"));
        var opened = router.Only("POST", "/v1/agents/sessions").Body!.AsObject();
        Assert.True(opened["start_voice"]!.GetValue<bool>());
        Assert.DoesNotContain(opened, pair => pair.Key is "call_id" or "call_type" or "text" or "instructions");
        Assert.Equal("false", router.Only("GET", Events).Query["decisions"]);
        Assert.Same(session, agent.Session);
        Assert.True(session.Voice.Started);
        Assert.Equal(new Call("s1", "agent"), session.Call);
        Assert.Contains("https://demo.test/join/s1?api_key=key&token=", agent.MonitorUrl(session));
    }

    [Fact]
    public async Task VoiceIsStartedAndStoppedOnAChat()
    {
        await using var router = await TestRouter.StartAsync();
        router.On("POST", "/v1/agents/sessions", 201, Fixtures.Session());
        router.On("POST", "/v1/agents/sessions/s1/voice", 200, Fixtures.Session("s1", "s1"));
        router.On("DELETE", "/v1/agents/sessions/s1/voice", 200, Fixtures.Session());
        router.OnSocket(Events, peer => peer.ReceiveAsync("close"));
        using var client = Fixtures.Client(router);
        await using var agent = new Agent(new AgentOptions { Name = "jean", Client = client, Edge = Fixtures.Edge(router) });
        var cancel = TestContext.Current.CancellationToken;

        var session = await agent.ChatAsync(cancellationToken: cancel);
        Assert.False(session.Voice.Started);
        Assert.Throws<ConfigurationException>(() => agent.MonitorUrl(session));

        var started = await session.Voice.StartAsync(cancel);
        Assert.Equal(("s1", true), (started.CallId, session.Voice.Started));
        Assert.Contains("/join/s1?", agent.MonitorUrl(session));

        await session.Voice.StopAsync(cancel);
        Assert.False(session.Voice.Started);
        Assert.Single(router.To("POST", "/v1/agents/sessions/s1/voice"));
        Assert.Single(router.To("DELETE", "/v1/agents/sessions/s1/voice"));
        Assert.Null(router.Only("POST", "/v1/agents/sessions").Body!["start_voice"]);
    }

    [Fact]
    public async Task AConversationIsResumedByItsSessionId()
    {
        await using var router = await TestRouter.StartAsync();
        router.On("GET", "/v1/agents/sessions/s1", 200, Fixtures.Session());
        router.OnSocket(Events, async peer =>
        {
            await peer.SendAsync(new { type = "responding", turn_id = "u1" });
            await peer.ReceiveAsync("close");
        });
        using var client = Fixtures.Client(router);
        await using var agent = new Agent(new AgentOptions { Name = "jean", Client = client });
        var cancel = TestContext.Current.CancellationToken;

        var session = await agent.ResumeAsync("s1", cancel);

        Assert.Equal("responding", (await FirstAsync(session)).Kind);
        Assert.Same(session, agent.Session);
        Assert.Empty(router.To("POST", "/v1/agents/sessions"));
    }

    [Fact]
    public async Task ToolsTheModelCallsRunHereAndAnswerByCallId()
    {
        await using var router = await TestRouter.StartAsync();
        router.On("POST", "/v1/agents/sessions", 201, Fixtures.Session());
        var results = new List<JsonObject>();
        router.OnSocket(Events, async peer =>
        {
            await peer.SendAsync(new { type = "tool_call", id = "t1", name = "weather", arguments = """{"name":"Paris"}""", request_id = "c1", turn_id = "u1" });
            results.Add(await peer.ReceiveAsync("tool_result"));
            await peer.SendAsync(new { type = "tool_call", id = "t2", name = "broken", arguments = "{}", request_id = "c2", turn_id = "u1" });
            results.Add(await peer.ReceiveAsync("tool_result"));
            // Every watcher is sent every call and the first answer wins, so a call for a tool
            // held elsewhere is left for whoever holds it: the next answer is t4's.
            await peer.SendAsync(new { type = "tool_call", id = "t3", name = "missing", arguments = "", request_id = "c3", turn_id = "u1" });
            await peer.SendAsync(new { type = "tool_call", id = "t4", name = "weather", arguments = """{"name":"Oslo"}""", request_id = "c4", turn_id = "u1" });
            results.Add(await peer.ReceiveAsync("tool_result"));
            await peer.SendAsync(new { type = "responded", text = "Sunny.", turn_id = "u1", pending_work = false });
            await peer.ReceiveAsync("close");
        });
        using var client = Fixtures.Client(router);
        await using var agent = new Agent(new AgentOptions { Name = "jean", Client = client });
        agent.Tools
            .Register<City, Forecast>("weather", "The weather in a city", (city, _) => Task.FromResult(new Forecast($"sunny in {city.Name}")))
            .Register<City, Forecast>("broken", "Always fails", (_, _) => throw new InvalidOperationException("the forecast is down"));
        var cancel = TestContext.Current.CancellationToken;

        var session = await agent.ChatAsync(cancellationToken: cancel);
        var responded = await FirstAsync(session);
        await session.CloseAsync(cancel);

        Assert.Equal(("responded", "Sunny."), (responded.Kind, responded.Text));
        Assert.Equal(("t1", "c1", "u1"), (results[0].Text("tool_call_id"), results[0].Text("request_id"), results[0].Text("turn_id")));
        Assert.Equal("""{"summary":"sunny in Paris"}""", results[0].Text("output"));
        Assert.Equal("the forecast is down", results[1].Text("error"));
        Assert.Equal("t4", results[2].Text("tool_call_id"));
        var declared = router.Only("POST", "/v1/agents/sessions").Body!["tools"]!.AsArray();
        Assert.Equal(["weather", "broken"], declared.Select(tool => tool.Text("name")));
        Assert.Equal("string", declared[0]!["parameters"]!["properties"]!["name"]!.Text("type"));
    }

    [Fact]
    public async Task ASessionIsSteeredOverItsSocket()
    {
        await using var router = await TestRouter.StartAsync();
        router.On("POST", "/v1/agents/sessions", 201, Fixtures.Session());
        var sent = new List<JsonObject>();
        var received = new TaskCompletionSource();
        router.OnSocket(Events, async peer =>
        {
            for (var index = 0; index < 3; index++)
            {
                sent.Add(await peer.ReceiveAsync());
            }
            received.SetResult();
        });
        using var client = Fixtures.Client(router);
        await using var agent = new Agent(new AgentOptions { Name = "jean", Client = client });
        var cancel = TestContext.Current.CancellationToken;

        var session = await agent.ChatAsync(cancellationToken: cancel);
        await session.SayAsync("Hello.", cancel);
        await session.InterruptAsync(cancel);
        await session.CloseAsync(cancel);
        await session.WaitAsync(cancel);
        await received.Task.WaitAsync(cancel);

        Assert.Equal(["say", "interrupt", "close"], sent.Select(frame => frame.Text("type")));
        Assert.Equal("Hello.", sent[0].Text("text"));
        Assert.False(session.Live);
        Assert.Empty(router.To("POST", "/v1/agents/sessions/s1/stop"));
    }

    [Fact]
    public async Task AResponseIsCreatedInTheConversationTheAgentOpened()
    {
        await using var router = await TestRouter.StartAsync();
        router.On("POST", "/v1/agents/sessions", 201, Fixtures.Session());
        router.On("POST", "/v1/agents/sessions/s1/responses", 202, new { id = "r1", session_id = "s1", status = "running" });
        router.OnSocket(Events, peer => peer.ReceiveAsync("close"));
        using var client = Fixtures.Client(router);
        await using var agent = new Agent(new AgentOptions { Name = "jean", Client = client });
        var cancel = TestContext.Current.CancellationToken;

        await agent.ChatAsync(cancellationToken: cancel);
        var response = await agent.Responses.CreateAsync("What is the plan?", cancellationToken: cancel);

        Assert.Equal("r1", response.Id);
        Assert.Equal("What is the plan?", router.Only("POST", "/v1/agents/sessions/s1/responses").Body.Text("text"));
    }

    [Fact]
    public async Task EachTextQuestionCarriesAFreshRequestIdAndOneWithAnImageNone()
    {
        await using var router = await TestRouter.StartAsync();
        router.On("POST", "/v1/agents/sessions", 201, Fixtures.Session());
        router.On("POST", "/v1/agents/sessions/s1/responses", 202, new { id = "r1", session_id = "s1", status = "running" });
        router.OnSocket(Events, peer => peer.ReceiveAsync("close"));
        using var client = Fixtures.Client(router);
        await using var agent = new Agent(new AgentOptions { Name = "jean", Client = client });
        var cancel = TestContext.Current.CancellationToken;

        await agent.ChatAsync(cancellationToken: cancel);
        await agent.Responses.CreateAsync("first", cancellationToken: cancel);
        await agent.Responses.CreateAsync("second", cancellationToken: cancel);
        await agent.Responses.CreateAsync("look", [new ImageSource { Url = "https://example.com/a.png" }], cancel);

        var sent = router.To("POST", "/v1/agents/sessions/s1/responses").Select(seen => seen.Body!).ToList();
        Assert.All(sent.Take(2), body => Assert.Matches("^[0-9a-f]{32}$", body.Text("request_id")));
        Assert.NotEqual(sent[0].Text("request_id"), sent[1].Text("request_id"));
        Assert.Null(sent[2]["request_id"]);
    }

    [Fact]
    public async Task AWatchedSessionForksIntoAnotherWatchedOne()
    {
        await using var router = await TestRouter.StartAsync();
        router.On("POST", "/v1/agents/sessions", 201, Fixtures.Session("s1"));
        router.On("POST", "/v1/agents/sessions/s1/fork", 201, Fixtures.Session("s2"));
        router.OnSocket(Events, peer => peer.ReceiveAsync("close"));
        router.OnSocket("/v1/agents/sessions/s2/events", async peer =>
        {
            await peer.SendAsync(new { type = "responding", turn_id = "u1" });
            await peer.ReceiveAsync("close");
        });
        using var client = Fixtures.Client(router);
        await using var agent = new Agent(new AgentOptions { Name = "jean", Client = client });
        var cancel = TestContext.Current.CancellationToken;

        var parent = await agent.ChatAsync(cancellationToken: cancel);
        await using var forked = await parent.ForkAsync(new ForkOptions { ResponseId = "r1" }, cancel);

        Assert.Equal("responding", (await FirstAsync(forked)).Kind);
        Assert.Equal("r1", router.Only("POST", "/v1/agents/sessions/s1/fork").Body.Text("response_id"));
    }

    [Fact]
    public async Task ASocketThatCannotOpenStopsTheSession()
    {
        await using var router = await TestRouter.StartAsync();
        router.On("POST", "/v1/agents/sessions", 201, Fixtures.Session());
        router.On("POST", "/v1/agents/sessions/s1/stop", 204);
        using var client = Fixtures.Client(router);
        await using var agent = new Agent(new AgentOptions { Name = "jean", Client = client });

        var refused = await Assert.ThrowsAsync<RouterException>(() => agent.ChatAsync(cancellationToken: TestContext.Current.CancellationToken));

        Assert.Equal(404, refused.Status);
        router.Only("POST", "/v1/agents/sessions/s1/stop");
        Assert.Empty(router.To("DELETE", "/v1/agents/sessions/s1"));
    }

    [Fact]
    public async Task ARefusedSocketCarriesTheStatusAndTheRequestToQuote()
    {
        await using var router = await TestRouter.StartAsync();
        router.On("POST", "/v1/agents/sessions", 201, Fixtures.Session());
        router.On("POST", "/v1/agents/sessions/s1/stop", 204);
        router.On("GET", Events, _ => new Reply(403,
            Fixtures.Failure("permission", "forbidden", "this session is somebody else's"),
            new Dictionary<string, string> { ["X-Request-Id"] = "req-socket" }));
        using var client = Fixtures.Client(router);
        await using var agent = new Agent(new AgentOptions { Name = "jean", Client = client });

        var refused = await Assert.ThrowsAsync<RouterException>(() => agent.ChatAsync(cancellationToken: TestContext.Current.CancellationToken));

        Assert.Equal((403, $"GET {Events}", "req-socket"), (refused.Status, refused.Operation, refused.RequestId));
        // ClientWebSocket keeps a refused upgrade's status and headers, never its body.
        Assert.Equal(("the router answered 403", null, null), (refused.Said, refused.Type, refused.Code));
    }

    [Fact]
    public async Task ADispatchedCallIsAnsweredOnTheNumberThatWasRung()
    {
        await using var router = await TestRouter.StartAsync();
        router.On("POST", "/v1/agents/sessions", 201, Fixtures.Session("inbound-1", "inbound-1"));
        router.OnSocket("/v1/agents/sessions/inbound-1/events", async peer =>
        {
            await peer.SendAsync(new { type = "participant_joined", participant = new { id = "p0", user_id = "jean", name = "Jean" } });
            await peer.SendAsync(new { type = "participant_joined", participant = new { id = "p1", user_id = "+15550001111", name = "Caller" } });
            await peer.ReceiveAsync("close");
        });
        using var client = Fixtures.Client(router);
        await using var agent = new Agent(new AgentOptions { Name = "jean", Client = client });
        var cancel = TestContext.Current.CancellationToken;

        var session = await agent.JoinAsync(new InboundCall { CallId = "inbound-1", SessionId = "inbound-1", CalledNumber = "+15552223333" }, cancellationToken: cancel);
        var caller = await session.WaitForParticipantAsync(cancel);

        Assert.Equal("+15550001111", caller.UserId);
        var opened = router.Only("POST", "/v1/agents/sessions").Body;
        Assert.Equal(("inbound-1", true, "+15552223333"), (opened.Text("id"), opened!["start_voice"]!.GetValue<bool>(), opened["phone"].Text("number")));
        await Assert.ThrowsAsync<ConfigurationException>(() => agent.JoinAsync(new InboundCall { CallId = "old" }, cancellationToken: cancel));
    }

    [Fact]
    public async Task AnOutboundCallIsPlacedForTheSessionTheAgentOpens()
    {
        await using var router = await TestRouter.StartAsync();
        router.On("POST", "/v1/phone/calls", 201, new { vendor_call_id = "vendor-1", session_id = "placed-1", status = "ringing" });
        router.On("POST", "/v1/agents/sessions", 201, Fixtures.Session("placed-1", "placed-1"));
        router.OnSocket("/v1/agents/sessions/placed-1/events", peer => peer.ReceiveAsync("close"));
        using var client = Fixtures.Client(router);
        await using var agent = new Agent(new AgentOptions
        {
            Name = "jean", Client = client, CostTracking = new Dictionary<string, string> { ["team"] = "sales" },
        });

        await agent.OutboundCallAsync("+15550001111", "+15552223333", cancellationToken: TestContext.Current.CancellationToken);

        var placed = router.Only("POST", "/v1/phone/calls").Body!.AsObject();
        Assert.Equal(("+15550001111", "+15552223333", "sales"), (placed.Text("from"), placed.Text("to"), placed["tags"].Text("team")));
        Assert.DoesNotContain(placed, pair => pair.Key is "call_id" or "call_type");
        var opened = router.Only("POST", "/v1/agents/sessions").Body;
        Assert.Equal(("placed-1", true, true), (opened.Text("id"), opened!["start_voice"]!.GetValue<bool>(), opened["navigating"]!.GetValue<bool>()));
        Assert.Equal(("+15550001111", "vendor-1"), (opened["phone"].Text("number"), opened["phone"].Text("vendor_call_id")));
    }

    [Fact]
    public async Task WaitingForACallAttachesTheNumberAndJoinsTheSessionTheRouterHandsOver()
    {
        await using var router = await TestRouter.StartAsync();
        router.On("POST", "/v1/phone/numbers/+15552223333/attach", 200, new { e164 = "+15552223333" });
        router.On("POST", "/v1/agents/sessions", 201, Fixtures.Session("rung-1", "rung-1"));
        router.OnSocket("/v1/dispatch", async peer =>
        {
            await peer.SendAsync(new
            {
                type = "call", work_id = "w1", call_id = "rung-1", call_type = "agent", session_id = "rung-1",
                called_number = "+15552223333", caller_number = "+15550001111",
            });
            await peer.Closed;
        });
        router.OnSocket("/v1/agents/sessions/rung-1/events", async peer =>
        {
            await peer.SendAsync(new { type = "heard", text = "Hello?", participant = new { id = "p1", user_id = "caller", name = "" } });
            await peer.ReceiveAsync("close");
        });
        using var client = Fixtures.Client(router);
        await using var agent = new Agent(new AgentOptions { Name = "jean", Client = client });
        var cancel = TestContext.Current.CancellationToken;

        var session = await agent.WaitForCallAsync("+15552223333", cancellationToken: cancel);

        Assert.Equal("Hello?", (await FirstAsync(session)).Text);
        Assert.Empty(router.Only("POST", "/v1/phone/numbers/+15552223333/attach").Body!.AsObject());
        Assert.Equal(("1", "call"), (router.Only("GET", "/v1/dispatch").Query["capacity"], router.Only("GET", "/v1/dispatch").Query["handles"]));
        var opened = router.Only("POST", "/v1/agents/sessions").Body;
        Assert.Equal(("rung-1", true, "+15552223333"), (opened.Text("id"), opened!["start_voice"]!.GetValue<bool>(), opened["phone"].Text("number")));
    }

    [Fact]
    public async Task ADirectoryIsSyncedBeforeTheFirstSessionAndOnlyReadBackOnceStamped()
    {
        var root = Path.Combine(_workspace, "agents", "jean");
        Directory.CreateDirectory(Path.Combine(root, "knowledge"));
        File.WriteAllText(Path.Combine(root, "agent.yaml"),
            "name: jean\nllm: openai/gpt-5.6\ngreeting:\n  text: Hello.\n  mode: variation\nplugins: [sentry]\nharness: default\ntags:\n  team: docs\ndispatch:\n  text: enabled\n");
        File.WriteAllText(Path.Combine(root, "instructions.md"), "You are Jean.\n");
        File.WriteAllText(Path.Combine(root, "knowledge", "pricing.md"), "A penny.\n");
        File.WriteAllText(Path.Combine(root, "knowledge", "urls.yaml"), "- url: https://example.com/plans\n  title: Plans\n  refresh_hours: 24\n");
        Directory.CreateDirectory(Path.Combine(root, "simulations"));
        File.WriteAllText(Path.Combine(root, "simulations", "lunch.yaml"),
            "- name: lunch\n  scenario: Order a club, then swap it.\n  assertion: One wrap.\n  variations: 3\n");

        await using var router = await TestRouter.StartAsync();
        router.On("POST", "/v1/agents/sync", 200, new { unchanged = false, config = Fixtures.Config("jean", "cfg-1") });
        router.On("GET", "/v1/agents/configs", 200, new[] { Fixtures.Config("jean", "cfg-1") });
        router.On("POST", "/v1/agents/sessions", 201, Fixtures.Session());
        router.OnSocket(Events, peer => peer.ReceiveAsync("close"));
        using var client = Fixtures.Client(router);
        var cancel = TestContext.Current.CancellationToken;
        var costs = new Dictionary<string, string> { ["tenant"] = "acme" };

        await using (var first = new Agent(new AgentOptions { Config = root, Client = client, CostTracking = costs }))
        {
            await first.ChatAsync(cancellationToken: cancel);
        }
        await using (var second = new Agent(new AgentOptions { Config = root, Client = client, CostTracking = costs }))
        {
            Assert.Equal("cfg-1", (await second.SyncAsync(cancel)).Id);
        }

        var synced = router.Only("POST", "/v1/agents/sync").Body!;
        Assert.Equal(("jean", "openai/gpt-5.6", "You are Jean."), (synced.Text("name"), synced.Text("llm"), synced.Text("instructions")));
        Assert.Equal(("docs", "acme"), (synced["tags"].Text("team"), synced["tags"].Text("tenant")));
        Assert.Equal("enabled", synced["dispatch"].Text("text"));
        Assert.Null(synced["dispatch"]!["incoming_call"]);
        Assert.Equal("pricing.md", synced["knowledge"]![0].Text("source"));
        Assert.Equal(("https://example.com/plans", "Plans"), (synced["knowledge_urls"]![0].Text("url"), synced["knowledge_urls"]![0].Text("title")));
        Assert.Equal(24, synced["knowledge_urls"]![0]!["refresh_hours"]!.GetValue<long>());
        Assert.Equal(("Hello.", "variation", "default"), (synced["greeting"].Text("text"), synced["greeting"].Text("mode"), synced.Text("harness")));
        Assert.Equal("sentry", synced["plugins"]![0]!.GetValue<string>());
        Assert.Null(synced["speed"]);
        var simulation = Assert.Single(synced["simulations"]!.AsArray())!.AsObject();
        Assert.Equal(["assertion", "name", "scenario", "variations"], simulation.Select(pair => pair.Key).Order());
        Assert.Equal(3, simulation["variations"]!.GetValue<long>());
        Assert.Equal(synced.Text("hash"), Folder.Load(root).ReadStamp());
        Assert.Equal("cfg-1", router.Only("POST", "/v1/agents/sessions").Body.Text("config_id"));
        Assert.Equal("jean", router.Only("GET", "/v1/agents/configs").Query["name"]);
    }

    [Fact]
    public async Task ASkillFolderSendsItsFilesAndASingleFileSkillSendsNone()
    {
        var root = Path.Combine(_workspace, "agents", "jean");
        Directory.CreateDirectory(Path.Combine(root, "skills", "pdf", "scripts"));
        File.WriteAllText(Path.Combine(root, "agent.yaml"), "name: jean\n");
        File.WriteAllText(Path.Combine(root, "skills", "think.md"), "---\ndescription: Work it out\n---\nReason it through.\n");
        File.WriteAllText(Path.Combine(root, "skills", "pdf", "SKILL.md"), "---\ndescription: Fill PDF forms\n---\nFill the form.\n");
        File.WriteAllText(Path.Combine(root, "skills", "pdf", "scripts", "fill.py"), "print('filled')\n");

        await using var router = await TestRouter.StartAsync();
        router.On("POST", "/v1/agents/sync", 200, new { unchanged = false, config = Fixtures.Config("jean", "cfg-1") });
        using var client = Fixtures.Client(router);
        await using var agent = new Agent(new AgentOptions { Config = root, Client = client });

        await agent.SyncAsync(TestContext.Current.CancellationToken);

        var skills = router.Only("POST", "/v1/agents/sync").Body!["skills"]!.AsArray();
        Assert.Equal(["pdf", "think"], skills.Select(skill => skill.Text("name")));
        Assert.Equal("print('filled')\n", skills[0]!["files"].Text("scripts/fill.py"));
        Assert.Single(skills[0]!["files"]!.AsObject());
        Assert.Null(skills[1]!["files"]);
    }

    [Fact]
    public async Task ADirectoryThatSaysNothingOfConnectorsSendsNoneAndLooksNothingUp()
    {
        await using var router = await TestRouter.StartAsync();

        var synced = await SyncDeclaringAsync(router, "llm: openai/gpt-5.6\n");

        Assert.False(synced.AsObject().ContainsKey("connectors"));
        Assert.Empty(router.To("GET", "/v1/agents/connections"));
    }

    [Fact]
    public async Task AnEmptyConnectorsListIsSentAsEmptyToClearTheStoredBindings()
    {
        await using var router = await TestRouter.StartAsync();

        var synced = await SyncDeclaringAsync(router, "connectors: []\n");

        Assert.Empty(synced["connectors"]!.AsArray());
    }

    [Fact]
    public async Task AConnectorNamedAloneIsASessionBindingOfEveryTool()
    {
        await using var router = await TestRouter.StartAsync();

        var synced = await SyncDeclaringAsync(router, "connectors:\n  - linear\n");

        var binding = Assert.Single(synced["connectors"]!.AsArray())!.AsObject();
        Assert.Equal("linear", binding.Text("name"));
        Assert.Equal("linear", binding.Text("connector_id"));
        Assert.Equal(["*"], binding["tool_patterns"]!.AsArray().Select(pattern => pattern!.GetValue<string>()));
        Assert.Empty(binding["tools"]!.AsArray());
        Assert.Equal("session", binding["connection"].Text("type"));
        Assert.Null(binding["connection"]!["connection_id"]);
        Assert.Null(binding["required"]);
        Assert.Empty(router.To("GET", "/v1/agents/connections"));
    }

    [Fact]
    public async Task AConnectorObjectNamesItsConnectorOrDefaultsItToItsName()
    {
        await using var router = await TestRouter.StartAsync();

        var synced = await SyncDeclaringAsync(router,
            "connectors:\n  - name: github\n  - name: issues\n    connector: linear\n");

        var bindings = synced["connectors"]!.AsArray();
        Assert.Equal([("github", "github"), ("issues", "linear")], bindings.Select(binding => (binding.Text("name"), binding.Text("connector_id"))));
    }

    [Fact]
    public async Task AConnectorsToolsAreSentAsWrittenAndLeftOutOrNullMeansEveryTool()
    {
        await using var router = await TestRouter.StartAsync();

        var synced = await SyncDeclaringAsync(router,
            "connectors:\n  - name: github\n    tools: [list_*, get_issue]\n    required: true\n" +
            "  - name: linear\n  - name: slack\n    tools: null\n");

        var bindings = synced["connectors"]!.AsArray();
        Assert.Equal(["list_*", "get_issue"], bindings[0]!["tool_patterns"]!.AsArray().Select(pattern => pattern!.GetValue<string>()));
        Assert.True(bindings[0]!["required"]!.GetValue<bool>());
        Assert.Equal(["*"], bindings[1]!["tool_patterns"]!.AsArray().Select(pattern => pattern!.GetValue<string>()));
        Assert.Equal(["*"], bindings[2]!["tool_patterns"]!.AsArray().Select(pattern => pattern!.GetValue<string>()));
    }

    [Fact]
    public async Task AConnectionIsFixedByItsIdAndAnIdOutranksAnotherConnectionsLabel()
    {
        await using var router = await TestRouter.StartAsync();
        Connections(router, ("", Listed(null, ("c-2", "c-1"), ("c-1", "work"))));

        var synced = await SyncDeclaringAsync(router, "connectors:\n  - name: github\n    connection: c-1\n");

        var connection = Assert.Single(synced["connectors"]!.AsArray())!["connection"];
        Assert.Equal(("fixed", "c-1"), (connection.Text("type"), connection.Text("connection_id")));
        var asked = router.Only("GET", "/v1/agents/connections").Query;
        Assert.Equal(("app", "github"), (asked["owner_type"], asked["connector_id"]));
    }

    [Fact]
    public async Task AConnectionIsFixedByTheOneLabelThatMatches()
    {
        await using var router = await TestRouter.StartAsync();
        Connections(router, ("", Listed(null, ("c-1", "work"), ("c-2", "home"))));

        var synced = await SyncDeclaringAsync(router, "connectors:\n  - name: github\n    connection: home\n");

        var connection = Assert.Single(synced["connectors"]!.AsArray())!["connection"];
        Assert.Equal(("fixed", "c-2"), (connection.Text("type"), connection.Text("connection_id")));
    }

    [Fact]
    public async Task EveryPageOfConnectionsIsReadSoALaterIdBeatsAnEarlierLabel()
    {
        await using var router = await TestRouter.StartAsync();
        Connections(router,
            ("", Listed("next", ("c-1", "c-2"))),
            ("next", Listed(null, ("c-2", "work"), ("c-3", "work"))));

        var byId = await SyncDeclaringAsync(router, "connectors:\n  - name: github\n    connection: c-2\n");
        Assert.Equal("c-2", byId["connectors"]![0]!["connection"].Text("connection_id"));

        Assert.Equal(["", "next"], router.To("GET", "/v1/agents/connections").Select(seen => seen.Query.GetValueOrDefault("cursor", "")));
    }

    [Fact]
    public async Task ALabelOnTwoPagesIsRefusedAsAmbiguous()
    {
        await using var router = await TestRouter.StartAsync();
        Connections(router,
            ("", Listed("next", ("c-1", "work"))),
            ("next", Listed(null, ("c-2", "work"))));

        var refused = await Assert.ThrowsAsync<ConfigurationException>(() =>
            SyncDeclaringAsync(router, "connectors:\n  - name: github\n    connection: work\n"));

        Assert.Equal("""connector "github": "work" matches 2 app connections""", refused.Message);
        Assert.Empty(router.To("POST", "/v1/agents/sync"));
    }

    [Fact]
    public async Task AConnectionNothingMatchesIsRefusedAndNothingIsSynced()
    {
        await using var router = await TestRouter.StartAsync();
        Connections(router, ("", Listed(null, ("c-1", "work"))));

        var refused = await Assert.ThrowsAsync<ConfigurationException>(() =>
            SyncDeclaringAsync(router, "connectors:\n  - name: github\n    connection: ghost\n"));

        Assert.Equal("connector \"github\": no app connection with id or label \"ghost\"", refused.Message);
        Assert.Empty(router.To("POST", "/v1/agents/sync"));
    }

    [Fact]
    public async Task ALabeledConnectionWithNoIdIsNotChosen()
    {
        // Go: skips connections whose id is nil (config.go:377); .NET must do the same
        await using var router = await TestRouter.StartAsync();
        // Listed("", "work") puts a connection with empty id and label "work"
        Connections(router, ("", Listed(null, ("", "work"))));

        var refused = await Assert.ThrowsAsync<ConfigurationException>(() =>
            SyncDeclaringAsync(router, "connectors:\n  - name: github\n    connection: work\n"));

        Assert.Equal("connector \"github\": no app connection with id or label \"work\"", refused.Message);
        Assert.Empty(router.To("POST", "/v1/agents/sync"));
    }

    [Fact]
    public async Task ARepeatedCursorStopsAfterTwoPagesNotMore()
    {
        // Guard: asked.Add(cursor) returns false when "next" is already in the set → exits after 2 GETs
        await using var router = await TestRouter.StartAsync();
        var gets = 0;
        router.On("GET", "/v1/agents/connections", seen => {
            gets++;
            // Cap at 4 to make the test fail fast when the guard is removed (while true)
            if (gets > 3) return new Reply(500, new { message = "too many pages" });
            return new Reply(200, Listed("next", ("c-1", "work")));
        });

        var synced = await SyncDeclaringAsync(router, "connectors:\n  - name: github\n    connection: c-1\n");

        Assert.Equal("c-1", synced["connectors"]![0]!["connection"].Text("connection_id"));
        Assert.Equal(2, gets);
    }

    [Fact]
    public async Task TheListQueryNarrowsByConnectorIdNotName()
    {
        // When connector: linear is declared, the GET uses connector_id=linear, not the name
        await using var router = await TestRouter.StartAsync();
        Connections(router, ("", Listed(null, ("c-1", "work"))));

        await SyncDeclaringAsync(router, "connectors:\n  - name: issues\n    connector: linear\n    connection: work\n");

        Assert.Equal("linear", router.Only("GET", "/v1/agents/connections").Query["connector_id"]);
    }

    [Fact]
    public async Task PluginsStillSyncAndTheDeprecationIsLoggedOncePerSync()
    {
        const string Warning = "agent.yaml plugins are deprecated: rename plugins to connectors";
        var heard = new Heard();
        System.Diagnostics.Trace.Listeners.Add(heard);
        try
        {
            await using var router = await TestRouter.StartAsync();
            router.On("GET", "/v1/agents/configs", 200, new[] { Fixtures.Config("jean-plugins", "cfg-1") });

            var root = System.IO.Path.Combine(_workspace, "agents", "jean-plugins");
            System.IO.Directory.CreateDirectory(root);
            System.IO.File.WriteAllText(System.IO.Path.Combine(root, "agent.yaml"), "name: jean-plugins\nplugins: [sentry]\n");
            router.On("POST", "/v1/agents/sync", 200, new { unchanged = false, config = Fixtures.Config("jean-plugins", "cfg-1") });
            using var client = Fixtures.Client(router);
            await using var agent = new Agent(new AgentOptions { Config = root, Client = client });

            await agent.SyncAsync(TestContext.Current.CancellationToken);
            Assert.Equal(1, heard.Warnings.Count(line => line == Warning));

            // Second sync hits the stamp path; warning must still fire once more
            await agent.SyncAsync(TestContext.Current.CancellationToken);
            Assert.Equal(2, heard.Warnings.Count(line => line == Warning));

            // Only one POST: the second sync returned early via the stamp
            Assert.Single(router.To("POST", "/v1/agents/sync"));
        }
        finally
        {
            System.Diagnostics.Trace.Listeners.Remove(heard);
        }
    }

    [Fact]
    public async Task AnAgentWithoutPluginsLogsNoDeprecation()
    {
        var heard = new Heard();
        System.Diagnostics.Trace.Listeners.Add(heard);
        try
        {
            await using var router = await TestRouter.StartAsync();

            await SyncDeclaringAsync(router, "connectors:\n  - linear\n");

            Assert.DoesNotContain(heard.Warnings, line => line.Contains("plugins are deprecated"));
        }
        finally
        {
            System.Diagnostics.Trace.Listeners.Remove(heard);
        }
    }

    [Fact]
    public async Task APageAddedToTheKnowledgeBaseIsWaitedFor()
    {
        await using var router = await TestRouter.StartAsync();
        var reads = 0;
        router.On("POST", "/v1/agents/knowledge/urls", 201, Page("pending", 0));
        router.On("GET", "/v1/agents/knowledge/urls/k1", _ => new Reply(200, ++reads < 2 ? Page("pending", 0) : Page("ready", 12)));
        using var client = Fixtures.Client(router);
        await using var agent = new Agent(new AgentOptions { Config = "support", Client = client });

        var page = await agent.Knowledge.AddUrlAsync("https://example.com/plans", "Plans", refreshHours: 24, cancellationToken: TestContext.Current.CancellationToken);

        Assert.Equal(("ready", 12), (page.State, page.Passages));
        var added = router.Only("POST", "/v1/agents/knowledge/urls").Body;
        Assert.Equal(("support", "https://example.com/plans", "Plans"), (added.Text("namespace"), added.Text("url"), added.Text("title")));
        Assert.Equal(24, added!["refresh_hours"]!.GetValue<int>());
        Assert.Null(added["description"]);
        await Assert.ThrowsAsync<ConfigurationException>(() =>
            agent.Knowledge.AddUrlAsync("https://example.com/plans", refreshHours: 0, cancellationToken: TestContext.Current.CancellationToken));
    }

    [Fact]
    public async Task SyncSendsHowTheSandboxIsBuiltAndWhatItMayReach()
    {
        await using var router = await TestRouter.StartAsync();

        var synced = await SyncDeclaringAsync(router,
            "sandbox: daytona\nsandbox_options:\n  setup: [pip install bpy==5.2.2]\n  timeout: 5m\n  memory_gb: 4\n" +
            "  network: true\n  allowed_domains: [api.example.com]\n  network_allow_list: [10.0.0.0/8]\n");

        Assert.Equal(
            """{"allowed_domains":["api.example.com"],"memory_gb":4,"network":true,"network_allow_list":["10.0.0.0/8"],"setup":["pip install bpy==5.2.2"],"timeout_ms":300000}""",
            synced["sandbox_options"]!.ToJsonString());
    }

    private async Task<JsonNode> SyncDeclaringAsync(TestRouter router, string declaration)
    {
        var root = Path.Combine(_workspace, "agents", "jean");
        Directory.CreateDirectory(root);
        File.WriteAllText(Path.Combine(root, "agent.yaml"), "name: jean\n" + declaration);
        router.On("POST", "/v1/agents/sync", 200, new { unchanged = false, config = Fixtures.Config("jean", "cfg-1") });
        using var client = Fixtures.Client(router);
        await using var agent = new Agent(new AgentOptions { Config = root, Client = client });
        await agent.SyncAsync(TestContext.Current.CancellationToken);
        return router.Only("POST", "/v1/agents/sync").Body!;
    }

    /// <summary>Answers the connection list: each page by the cursor it is asked for, "" for the first.</summary>
    private static void Connections(TestRouter router, params (string Cursor, object Page)[] pages) =>
        router.On("GET", "/v1/agents/connections", seen => new Reply(200, pages.Single(page => page.Cursor == seen.Query.GetValueOrDefault("cursor", "")).Page));

    private static object Listed(string? next, params (string Id, string Label)[] connections) => new
    {
        has_more = next is not null,
        next_cursor = next,
        items = connections.Select(connection => new { id = connection.Id, label = connection.Label, connector_id = "github" }),
    };

    private static object Page(string state, int passages) => new
    {
        id = "k1", @namespace = "support", url = "https://example.com/plans", state, passages,
        created_at = "2026-09-24T10:00:00Z", updated_at = "2026-09-24T10:00:00Z",
    };

    private static async Task<SessionEvent> FirstAsync(Session session)
    {
        using var patience = new CancellationTokenSource(TimeSpan.FromSeconds(5));
        await foreach (var happened in session.EventsAsync(patience.Token))
        {
            return happened;
        }
        throw new InvalidOperationException("the session ended with nothing said");
    }

    private sealed class Heard : System.Diagnostics.TraceListener
    {
        private readonly System.Collections.Concurrent.ConcurrentQueue<string> _warnings = new();

        public IReadOnlyCollection<string> Warnings => _warnings;

        public override void TraceEvent(System.Diagnostics.TraceEventCache? cache, string source, System.Diagnostics.TraceEventType type, int id, string? message)
        {
            if (type == System.Diagnostics.TraceEventType.Warning && message is not null)
            {
                _warnings.Enqueue(message);
            }
        }

        public override void TraceEvent(System.Diagnostics.TraceEventCache? cache, string source, System.Diagnostics.TraceEventType type, int id, string? format, params object?[]? args) =>
            TraceEvent(cache, source, type, id, format is null ? null : args is { Length: > 0 } ? string.Format(format, args) : format);

        public override void Write(string? message)
        {
        }

        public override void WriteLine(string? message)
        {
        }
    }

    private sealed record City(string Name);

    private sealed record Forecast(string Summary);
}
