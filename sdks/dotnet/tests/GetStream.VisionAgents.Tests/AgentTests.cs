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
            Harness = new Harness
            {
                Subagents = new Dictionary<string, string> { ["default"] = "llm-thinking" },
                Sandbox = Sandbox.Daytona(),
                Skills = [new Skill { Name = "think", Description = "Work it out", Instructions = "Reason.", Deadline = TimeSpan.FromSeconds(30) }],
                Tasks = 2,
            },
            CostTracking = new Dictionary<string, string> { ["team"] = "support" },
            MemoryFilter = new Dictionary<string, string> { [Agent.UserKey] = "ada", ["topic"] = "billing" },
            Client = client,
        });

        var request = agent.Request(null, new SessionOptions { Persist = true, Title = "Billing" }, null, false);

        Assert.Equal("jean-luc", agent.UserId);
        Assert.Equal(("support", null), (request.Agent, request.ConfigId));
        Assert.True(request.Text);
        Assert.Equal(("jean-luc", "Jean Luc!", "jean-luc"), (request.UserId, request.UserName, request.AgentId));
        Assert.Equal(("llm-fast", "llm-thinking", "daytona", 2), (request.Llm, request.Subagent, request.Sandbox, request.Tasks));
        Assert.Equal(["fr"], request.Languages!);
        Assert.Equal(2000, request.ToolTimeoutMs);
        Assert.Equal(30_000, Assert.Single(request.Skills!).DeadlineMs);
        Assert.Equal("support", request.Tags!["team"]);
        Assert.Equal("ada", request.Memory!.UserId);
        Assert.Equal(new Dictionary<string, string> { ["topic"] = "billing" }, request.Memory.Filter);
        Assert.Equal((true, "Billing"), (request.PersistConversation, request.Title));
    }

    [Fact]
    public void AnAbsentSkillListAndAnEmptyOneDiffer()
    {
        using var client = new VisionAgentsClient(new VisionAgentsOptions { Url = "http://x", CustomerId = "examples" });

        var defaults = new Agent(new AgentOptions { Name = "jean", Harness = new Harness(), Client = client });
        var none = new Agent(new AgentOptions { Name = "jean", Harness = new Harness { UseSkills = false }, Client = client });

        Assert.Null(defaults.Request(null, null, null, false).Skills);
        Assert.Empty(none.Request(null, null, null, false).Skills!);
    }

    [Fact]
    public void AnIncognitoConversationAsksForNoTranscript()
    {
        using var client = new VisionAgentsClient(new VisionAgentsOptions { Url = "http://x", CustomerId = "examples" });
        var agent = new Agent(new AgentOptions { Name = "jean", Client = client });

        var request = agent.Request(null, new SessionOptions { Persist = true, Incognito = true }, null, false);

        Assert.Equal((true, null), (request.Incognito, request.PersistConversation));
    }

    [Fact]
    public void AConfigIsNamedOrGivenByIdButNotBoth()
    {
        using var client = new VisionAgentsClient(new VisionAgentsOptions { Url = "http://x", CustomerId = "examples" });

        var request = new Agent(new AgentOptions { ConfigId = "cfg-1", Client = client }).Request(null, null, null, false);

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
    public async Task JoiningCreatesTheStreamCallAndOpensTheSessionOnIt()
    {
        await using var router = await TestRouter.StartAsync();
        router.On("POST", "/api/v2/video/call/agent/call-1", 201, Fixtures.StreamCall("call-1"));
        router.On("POST", "/v1/agents/sessions", 201, Fixtures.Session("s1", "call-1"));
        router.OnSocket(Events, async peer =>
        {
            await peer.SendAsync(new { type = "joined" });
            await peer.ReceiveAsync("close");
        });
        using var client = Fixtures.Client(router);
        await using var agent = new Agent(new AgentOptions { Name = "Jean", Client = client, Edge = Fixtures.Edge(router) });
        var cancel = TestContext.Current.CancellationToken;

        var session = await agent.JoinAsync("call-1", cancellationToken: cancel);
        var first = await FirstAsync(session);
        await session.CloseAsync(cancel);

        Assert.Equal("joined", first.Kind);
        var created = router.Only("POST", "/api/v2/video/call/agent/call-1").Body;
        Assert.Equal("jean", created?["data"].Text("created_by_id"));
        var opened = router.Only("POST", "/v1/agents/sessions").Body;
        Assert.Equal(("call-1", "agent"), (opened.Text("call_id"), opened.Text("call_type")));
        Assert.Equal("false", router.Only("GET", Events).Query["decisions"]);
        Assert.Same(session, agent.Session);
        Assert.Contains("https://demo.test/join/call-1?api_key=key&token=", agent.MonitorUrl(session));
    }

    [Fact]
    public async Task ToolsTheModelCallsRunHereAndAnswerByCallId()
    {
        await using var router = await TestRouter.StartAsync();
        router.On("POST", "/v1/agents/sessions", 201, Fixtures.Session());
        var results = new List<JsonObject>();
        router.OnSocket(Events, async peer =>
        {
            await peer.SendAsync(new { type = "tool_call", id = "t1", name = "weather", arguments = """{"name":"Paris"}""", command_id = "c1", turn_id = "u1" });
            results.Add(await peer.ReceiveAsync("tool_result"));
            await peer.SendAsync(new { type = "tool_call", id = "t2", name = "broken", arguments = "{}", command_id = "c2", turn_id = "u1" });
            results.Add(await peer.ReceiveAsync("tool_result"));
            await peer.SendAsync(new { type = "tool_call", id = "t3", name = "missing", arguments = "", command_id = "c3", turn_id = "u1" });
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
        Assert.Equal(("t1", "c1", "u1"), (results[0].Text("tool_call_id"), results[0].Text("command_id"), results[0].Text("turn_id")));
        Assert.Equal("""{"summary":"sunny in Paris"}""", results[0].Text("output"));
        Assert.Equal("the forecast is down", results[1].Text("error"));
        Assert.Contains("missing", results[2].Text("error"));
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
        router.OnSocket(Events, async peer =>
        {
            for (var index = 0; index < 5; index++)
            {
                sent.Add(await peer.ReceiveAsync());
            }
        });
        using var client = Fixtures.Client(router);
        await using var agent = new Agent(new AgentOptions { Name = "jean", Client = client });
        var cancel = TestContext.Current.CancellationToken;

        var session = await agent.ChatAsync(cancellationToken: cancel);
        await session.SayAsync("Hello.", cancel);
        await session.RespondAsync("What is new?", cancellationToken: cancel);
        await session.InterruptAsync(cancel);
        await session.SetInstructionsAsync("Be brief.", cancel);
        await session.CloseAsync(cancel);
        await session.WaitAsync(cancel);

        Assert.Equal(["say", "respond", "interrupt", "instructions", "close"], sent.Select(frame => frame.Text("type")));
        Assert.Equal("Hello.", sent[0].Text("text"));
        Assert.Equal("Be brief.", sent[3].Text("instructions"));
        Assert.False(session.Live);
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
    public async Task ASocketThatCannotOpenDeletesTheSession()
    {
        await using var router = await TestRouter.StartAsync();
        router.On("POST", "/v1/agents/sessions", 201, Fixtures.Session());
        router.On("DELETE", "/v1/agents/sessions/s1", 204);
        using var client = Fixtures.Client(router);
        await using var agent = new Agent(new AgentOptions { Name = "jean", Client = client });

        var refused = await Assert.ThrowsAsync<RouterException>(() => agent.ChatAsync(cancellationToken: TestContext.Current.CancellationToken));

        Assert.Equal(404, refused.Status);
        router.Only("DELETE", "/v1/agents/sessions/s1");
    }

    [Fact]
    public async Task ADispatchedCallIsAnsweredOnTheNumberThatWasRung()
    {
        await using var router = await TestRouter.StartAsync();
        router.On("POST", "/v1/agents/sessions", 201, Fixtures.Session("s1", "inbound-1"));
        router.OnSocket(Events, async peer =>
        {
            await peer.SendAsync(new { type = "participant_joined", participant = new { id = "p0", user_id = "jean", name = "Jean" } });
            await peer.SendAsync(new { type = "participant_joined", participant = new { id = "p1", user_id = "+15550001111", name = "Caller" } });
            await peer.ReceiveAsync("close");
        });
        using var client = Fixtures.Client(router);
        await using var agent = new Agent(new AgentOptions { Name = "jean", Client = client });
        var cancel = TestContext.Current.CancellationToken;

        var session = await agent.JoinAsync(new InboundCall { CallId = "inbound-1", CalledNumber = "+15552223333" }, cancellationToken: cancel);
        var caller = await session.WaitForParticipantAsync(cancel);

        Assert.Equal("+15550001111", caller.UserId);
        var opened = router.Only("POST", "/v1/agents/sessions").Body;
        Assert.Equal(("inbound-1", "+15552223333"), (opened.Text("call_id"), opened?["phone"].Text("number")));
        Assert.Empty(router.To("POST", "/api/v2/video/call/agent/inbound-1"));
    }

    [Fact]
    public async Task AnOutboundCallIsPlacedIntoTheCallTheAgentJoins()
    {
        await using var router = await TestRouter.StartAsync();
        router.On("POST", "/api/v2/video/call/agent/*", seen => new Reply(201, Fixtures.StreamCall(seen.Path.Split('/')[^1])));
        router.On("POST", "/v1/phone/calls", 201, new { vendor_call_id = "vendor-1", call_id = "any", call_type = "agent", from = "+1", to = "+2" });
        router.On("POST", "/v1/agents/sessions", 201, Fixtures.Session("s1", "placed"));
        router.OnSocket(Events, peer => peer.ReceiveAsync("close"));
        using var client = Fixtures.Client(router);
        await using var agent = new Agent(new AgentOptions
        {
            Name = "jean", Client = client, Edge = Fixtures.Edge(router), CostTracking = new Dictionary<string, string> { ["team"] = "sales" },
        });

        await agent.OutboundCallAsync("+15550001111", "+15552223333", cancellationToken: TestContext.Current.CancellationToken);

        var placed = router.Only("POST", "/v1/phone/calls").Body;
        var callId = Assert.Single(router.Requests, seen => seen.Path.StartsWith("/api/v2/video/call/agent/")).Path.Split('/')[^1];
        Assert.Equal(("+15550001111", "+15552223333", callId, "sales"),
            (placed.Text("from"), placed.Text("to"), placed.Text("call_id"), placed?["tags"].Text("team")));
        var opened = router.Only("POST", "/v1/agents/sessions").Body;
        Assert.Equal(callId, opened.Text("call_id"));
        Assert.True(opened!["navigating"]!.GetValue<bool>());
        Assert.Equal(("+15550001111", "vendor-1"), (opened["phone"].Text("number"), opened["phone"].Text("vendor_call_id")));
    }

    [Fact]
    public async Task WaitingForACallAttachesTheNumberAndReturnsOnceSomebodySpeaks()
    {
        await using var router = await TestRouter.StartAsync();
        router.On("POST", "/api/v2/video/call/agent/*", seen => new Reply(201, Fixtures.StreamCall(seen.Path.Split('/')[^1])));
        router.On("POST", "/v1/phone/numbers/+15552223333/attach", 200, new { e164 = "+15552223333" });
        router.On("POST", "/v1/agents/sessions", 201, Fixtures.Session("s1", "waiting"));
        router.OnSocket(Events, async peer =>
        {
            await peer.SendAsync(new { type = "heard", text = "Hello?", participant = new { id = "p1", user_id = "caller", name = "" } });
            await peer.ReceiveAsync("close");
        });
        using var client = Fixtures.Client(router);
        await using var agent = new Agent(new AgentOptions { Name = "jean", Client = client, Edge = Fixtures.Edge(router) });
        var cancel = TestContext.Current.CancellationToken;

        var session = await agent.WaitForCallAsync("+15552223333", cancellationToken: cancel);

        Assert.Equal("Hello?", (await FirstAsync(session)).Text);
        var attached = router.Only("POST", "/v1/phone/numbers/+15552223333/attach").Body;
        Assert.Equal(router.Only("POST", "/v1/agents/sessions").Body.Text("call_id"), attached.Text("call_id"));
        Assert.Equal("+15552223333", router.Only("POST", "/v1/agents/sessions").Body?["phone"].Text("number"));
    }

    [Fact]
    public async Task ADirectoryIsSyncedBeforeTheFirstSessionAndOnlyReadBackOnceStamped()
    {
        var root = Path.Combine(_workspace, "agents", "jean");
        Directory.CreateDirectory(Path.Combine(root, "knowledge"));
        File.WriteAllText(Path.Combine(root, "agent.yaml"), "name: jean\nllm: openai/gpt-5.6\ntags:\n  team: docs\n");
        File.WriteAllText(Path.Combine(root, "instructions.md"), "You are Jean.\n");
        File.WriteAllText(Path.Combine(root, "knowledge", "pricing.md"), "A penny.\n");
        File.WriteAllText(Path.Combine(root, "knowledge", "urls.yaml"), "- url: https://example.com/plans\n  title: Plans\n");

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
        Assert.Equal("pricing.md", synced["knowledge"]![0].Text("source"));
        Assert.Equal(("https://example.com/plans", "Plans"), (synced["knowledge_urls"]![0].Text("url"), synced["knowledge_urls"]![0].Text("title")));
        Assert.Equal(synced.Text("hash"), Folder.Load(root).ReadStamp());
        Assert.Equal("cfg-1", router.Only("POST", "/v1/agents/sessions").Body.Text("config_id"));
        Assert.Equal("jean", router.Only("GET", "/v1/agents/configs").Query["name"]);
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

        var page = await agent.Knowledge.AddUrlAsync("https://example.com/plans", "Plans", cancellationToken: TestContext.Current.CancellationToken);

        Assert.Equal(("ready", 12), (page.State, page.Passages));
        var added = router.Only("POST", "/v1/agents/knowledge/urls").Body;
        Assert.Equal(("support", "https://example.com/plans", "Plans"), (added.Text("namespace"), added.Text("url"), added.Text("title")));
        Assert.Null(added?["description"]);
    }

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

    private sealed record City(string Name);

    private sealed record Forecast(string Summary);
}
