using System.Collections.Concurrent;
using System.ComponentModel;
using System.Text.Json;
using System.Text.Json.Nodes;

namespace GetStream.VisionAgents.Tests;

public sealed class ToolCallTests
{
    private const string Events = "/v1/agents/sessions/s1/events";
    private const string SessionTools = "/v1/agents/sessions/s1/tools";

    [Fact]
    public async Task AFunctionIsToldWhoItsCallAnswersAndCanSayHowItIsGoing()
    {
        await using var router = await TestRouter.StartAsync();
        router.On("POST", "/v1/agents/sessions", 201, Fixtures.Session());
        var sent = new List<JsonObject>();
        router.OnSocket(Events, async peer =>
        {
            await peer.SendAsync(new
            {
                type = "tool_call",
                id = "t1",
                name = "book",
                arguments = "{}",
                turn_id = "turn-1",
                approved = true,
                user = new { id = "u1", name = "Ada", custom = new { tier = "gold" } },
                conversation = new { cid = "messaging:c1", custom = new { } },
            });
            sent.Add(await peer.ReceiveAsync("tool_result"));
            await Finish(peer);
        });
        await using var agent = Agent(router);
        agent.Tools.Register("book", "Books a flight", [], (ToolCallContext toolCallContext, JsonElement _, CancellationToken _) =>
        {
            var user = toolCallContext.User;
            return Task.FromResult<object?>($"{toolCallContext.Id} {toolCallContext.SessionId} {toolCallContext.TurnId} {user.Id} {user.Name} " +
                $"{user.Custom.Text("tier")} {toolCallContext.Conversation.Cid} {toolCallContext.Approved}");
        });

        await Converse(agent);

        Assert.Equal("t1 s1 turn-1 u1 Ada gold messaging:c1 True", sent[0].Text("output"));
    }

    [Fact]
    public async Task ACallTheRouterRunsElsewhereIsLeftAlone()
    {
        await using var router = await TestRouter.StartAsync();
        router.On("POST", "/v1/agents/sessions", 201, Fixtures.Session());
        JsonObject? answered = null;
        router.OnSocket(Events, async peer =>
        {
            await peer.SendAsync(new { type = "tool_call", id = "f1", name = "fast", arguments = "{}", run = false });
            await peer.SendAsync(new { type = "tool_call", id = "f2", name = "fast", arguments = "{}" });
            answered = await peer.ReceiveAsync("tool_result");
            await Finish(peer);
        });
        await using var agent = Agent(router);
        Payments(agent.Tools);

        await Converse(agent);

        Assert.Equal("f2", answered.Text("tool_call_id"));
    }

    [Fact]
    public async Task AHeldCallIsNotCancelledByThePersonTalkingOverTheAgent()
    {
        await using var router = await TestRouter.StartAsync();
        router.On("POST", "/v1/agents/sessions", 201, Fixtures.Session());
        var sent = new List<JsonObject>();
        router.OnSocket(Events, async peer =>
        {
            await peer.SendAsync(new { type = "tool_call", id = "p1", name = "pay", arguments = "{}" });
            sent.Add(await peer.ReceiveAsync());
            await peer.SendAsync(new { type = "tool_held", id = "p1" });
            await peer.SendAsync(new { type = "tool_cancel", id = "p1" });
            sent.Add(await peer.ReceiveAsync("tool_result"));
            await Finish(peer);
        });
        await using var agent = Agent(router);
        Payments(agent.Tools);

        var session = await agent.ChatAsync(cancellationToken: TestContext.Current.CancellationToken);
        var kinds = new List<string>();
        using var patience = new CancellationTokenSource(TestRouter.Patience);
        await foreach (var happened in session.EventsAsync(patience.Token))
        {
            kinds.Add(happened.Kind);
            if (happened.Kind == "responded")
            {
                break;
            }
        }
        await session.CloseAsync(TestContext.Current.CancellationToken);

        Assert.Equal("""{"type":"tool_hold","tool_call_id":"p1"}""", sent[0].ToJsonString());
        Assert.Equal(("p1", "paid for s1"), (sent[1].Text("tool_call_id"), sent[1].Text("output")));
        Assert.Equal(["tool_held", "responded"], kinds);
    }

    [Fact]
    public async Task AHoldAskedForTooLateIsCancelledWithTheCall()
    {
        await using var router = await TestRouter.StartAsync();
        router.On("POST", "/v1/agents/sessions", 201, Fixtures.Session());
        var answered = new List<JsonObject>();
        router.OnSocket(Events, async peer =>
        {
            await peer.SendAsync(new { type = "tool_call", id = "p1", name = "pay", arguments = "{}" });
            await peer.ReceiveAsync("tool_hold");
            await peer.SendAsync(new { type = "tool_cancel", id = "p1" });
            await peer.SendAsync(new { type = "tool_call", id = "f1", name = "fast", arguments = "{}" });
            answered.Add(await peer.ReceiveAsync("tool_result"));
            await Task.Delay(300);
            await peer.SendAsync(new { type = "responded", text = "Done.", pending_work = false });
            for (var frame = await peer.ReceiveAsync(); frame.Text("type") != "close"; frame = await peer.ReceiveAsync())
            {
                answered.Add(frame);
            }
        });
        await using var agent = Agent(router);
        Payments(agent.Tools);

        await Converse(agent);

        Assert.Equal("f1", Assert.Single(answered).Text("tool_call_id"));
    }

    [Fact]
    public async Task ACallThatOutlastsItsTimeoutFails()
    {
        await using var router = await TestRouter.StartAsync();
        router.On("POST", "/v1/agents/sessions", 201, Fixtures.Session());
        var answered = new List<JsonObject>();
        router.OnSocket(Events, async peer =>
        {
            await peer.SendAsync(new { type = "tool_call", id = "s1", name = "slow", arguments = "{}" });
            answered.Add(await peer.ReceiveAsync("tool_result"));
            await peer.SendAsync(new { type = "tool_call", id = "s2", name = "deaf", arguments = "{}" });
            answered.Add(await peer.ReceiveAsync("tool_result"));
            await Finish(peer);
        });
        await using var agent = Agent(router);
        agent.Tools
            .Register("slow", "Takes a while", [], async (JsonElement _, CancellationToken cancellationToken) =>
            {
                await Task.Delay(TimeSpan.FromSeconds(30), cancellationToken);
                return (object?)"done";
            }, timeout: TimeSpan.FromMilliseconds(50))
            .Register("deaf", "Ignores being told to stop", [], async (JsonElement _, CancellationToken _) =>
            {
                await Task.Delay(TimeSpan.FromSeconds(3));
                return (object?)"done";
            }, timeout: TimeSpan.FromMilliseconds(50));

        await Converse(agent);

        Assert.Equal(("s1", "slow took longer than 0.05s"), (answered[0].Text("tool_call_id"), answered[0].Text("error")));
        Assert.Equal(("s2", "deaf took longer than 0.05s"), (answered[1].Text("tool_call_id"), answered[1].Text("error")));
    }

    [Fact]
    public async Task AToolIsDeclaredWithOnlyThePoliciesItSets()
    {
        await using var router = await TestRouter.StartAsync();
        router.On("POST", "/v1/agents/sessions", 201, Fixtures.Session());
        router.OnSocket(Events, Finish);
        await using var agent = Agent(router);
        Payments(agent.Tools);
        agent.Tools.Register("refund", "Refunds", Amount(), (JsonElement _, CancellationToken _) => Task.FromResult<object?>("refunded"),
            interruptible: true, timeout: TimeSpan.FromSeconds(5), approval: "Refund {amount}?");

        await Converse(agent);

        var declared = router.Only("POST", "/v1/agents/sessions").Body!["tools"]!.AsArray()
            .ToDictionary(tool => tool.Text("name"), tool => tool!.AsObject());
        Assert.Equal("""{"title":"Refund {amount}?"}""", declared["refund"]["approval"]!.ToJsonString());
        Assert.Equal((true, 5000L), (declared["refund"]["interruptible"]!.GetValue<bool>(), declared["refund"]["timeout_ms"]!.GetValue<long>()));
        Assert.False(declared["fast"].ContainsKey("interruptible"));
        Assert.False(declared["fast"].ContainsKey("timeout_ms"));
        Assert.False(declared["fast"].ContainsKey("approval"));
    }

    [Fact]
    public void APolicyThatCannotWorkIsRefusedWhenItIsRegistered()
    {
        var tools = new Tools();
        string? Refused(TimeSpan? timeout = null, string? approval = null)
        {
            try
            {
                tools.Register("refund", "Refunds", Amount(), (JsonElement _, CancellationToken _) => Task.FromResult<object?>(""),
                    timeout: timeout, approval: approval);
                return null;
            }
            catch (ConfigurationException refused)
            {
                return refused.Message;
            }
        }

        Assert.Equal("refund's approval names {card}, which is not one of its parameters", Refused(approval: "Refund {amount} to {card}?"));
        Assert.Equal("refund's approval question cannot be blank", Refused(approval: "  "));
        Assert.Equal("refund has a timeout that is not positive", Refused(timeout: TimeSpan.Zero));
        Assert.Equal(0, tools.Count);
        Assert.Null(Refused(approval: "Refund {amount}? {not a placeholder}"));
        Assert.Equal(1, tools.Count);
    }

    [Fact]
    public void ARecordParametersDescriptionIsItsArgumentsDescription()
    {
        var schema = Schema<WeatherLookup>();

        Assert.Equal("""{"type":"string","description":"a city or region"}""", schema["properties"]!["location"]!.ToJsonString());
    }

    [Fact]
    public void APropertysDescriptionIsItsArgumentsDescription()
    {
        var schema = Schema<Booking>();

        Assert.Equal("how many people are coming", schema["properties"]!["guests"]!.Text("description"));
    }

    [Fact]
    public void ATypesDescriptionDescribesTheArgumentsAndAnyObjectOfThatType()
    {
        var schema = Schema<Trip>();

        Assert.Equal("a journey of one leg", schema.Text("description"));
        Assert.Equal("one hop between two places", schema["properties"]!["leg"]!.Text("description"));
    }

    [Fact]
    public void ArgumentsWithNoDescriptionAreDeclaredAsTheyAre()
    {
        var schema = Schema<Undescribed>();

        Assert.Equal("""{"type":["object","null"],"properties":{"location":{"type":"string"}},"required":["location"]}""",
            schema.ToJsonString());
    }

    private static JsonObject Schema<TArguments>()
    {
        var tools = new Tools().Register<TArguments, string>("tool", "A tool", (_, _) => Task.FromResult(""));
        return JsonSerializer.SerializeToNode(Assert.Single(tools.Declared()).Parameters)!.AsObject();
    }

    private sealed record WeatherLookup([Description("a city or region")] string Location);

    private sealed record Booking
    {
        [Description("how many people are coming")]
        public int Guests { get; init; }
    }

    [Description("a journey of one leg")]
    private sealed record Trip(Leg Leg);

    [Description("one hop between two places")]
    private sealed record Leg(string From);

    private sealed record Undescribed(string Location);

    [Fact]
    public async Task AHostedToolIsDeclaredWithItsPoliciesAndAnsweredByCallId()
    {
        await using var router = await TestRouter.StartAsync();
        JsonObject? declared = null;
        JsonObject? answered = null;
        router.OnSocket("/v1/dispatch", async peer =>
        {
            await peer.SendAsync(new { type = "ready", worker_id = "worker-1" });
            declared = await peer.ReceiveAsync("host_tools");
            await peer.SendAsync(new { type = "tool_call", id = "c1", session_id = "s1", name = "fast", arguments = "{}" });
            answered = await peer.ReceiveAsync("tool_result");
        });
        using var client = Fixtures.Client(router);
        Payments(client.Tools);

        await new Dispatch(new DispatchOptions { Client = client }).Host(client, TimeSpan.FromMinutes(1))
            .RunAsync(TestContext.Current.CancellationToken);

        var tools = declared!["tools"]!.AsArray().ToDictionary(tool => tool.Text("name"), tool => tool!.AsObject());
        Assert.Equal("""{"title":"Pay {amount}?"}""", tools["pay"]["approval"]!.ToJsonString());
        Assert.True(tools["slow"]["interruptible"]!.GetValue<bool>());
        Assert.False(tools["fast"].ContainsKey("interruptible"));
        Assert.Equal(("c1", "c1", "done"), (answered.Text("tool_call_id"), answered.Text("id"), answered.Text("output")));
    }

    [Fact]
    public async Task AHostedCallCancelledByAnInterruptionIsNotAnswered()
    {
        await using var router = await TestRouter.StartAsync();
        var answered = new List<JsonObject>();
        router.OnSocket("/v1/dispatch", async peer =>
        {
            await peer.SendAsync(new { type = "ready", worker_id = "worker-1" });
            await peer.ReceiveAsync("host_tools");
            await peer.SendAsync(new { type = "tool_call", id = "c1", session_id = "s1", name = "slow", arguments = "{}" });
            await peer.SendAsync(new { type = "tool_call", id = "c1", session_id = "s2", name = "fast", arguments = "{}", run = false });
            // Ids are only unique within a session: this names a call that is not running.
            await peer.SendAsync(new { type = "tool_cancel", id = "c1", session_id = "s3" });
            await peer.SendAsync(new { type = "tool_cancel", id = "c1", session_id = "s1" });
            await peer.SendAsync(new { type = "tool_call", id = "c2", session_id = "s1", name = "fast", arguments = "{}" });
            answered.Add(await peer.ReceiveAsync("tool_result"));
            await Task.Delay(300);
            while (await Next(peer) is { } frame)
            {
                answered.Add(frame);
            }
        });
        using var client = Fixtures.Client(router);
        Payments(client.Tools);

        await new Dispatch(new DispatchOptions { Client = client }).Host(client, TimeSpan.FromMinutes(1))
            .RunAsync(TestContext.Current.CancellationToken);

        Assert.Equal("c2", Assert.Single(answered, frame => frame.Text("type") == "tool_result").Text("id"));
    }

    [Fact]
    public async Task AHostedCallHeldThroughInterruptionsIsAnswered()
    {
        await using var router = await TestRouter.StartAsync();
        var sent = new List<JsonObject>();
        router.OnSocket("/v1/dispatch", async peer =>
        {
            await peer.SendAsync(new { type = "ready", worker_id = "worker-1" });
            await peer.ReceiveAsync("host_tools");
            await peer.SendAsync(new { type = "tool_call", id = "c1", session_id = "s1", name = "pay", arguments = "{}" });
            sent.Add(await peer.ReceiveAsync("tool_hold"));
            await peer.SendAsync(new { type = "tool_held", id = "c1", session_id = "s1" });
            await peer.SendAsync(new { type = "tool_cancel", id = "c1", session_id = "s1" });
            sent.Add(await peer.ReceiveAsync("tool_result"));
        });
        using var client = Fixtures.Client(router);
        Payments(client.Tools);

        await new Dispatch(new DispatchOptions { Client = client }).Host(client, TimeSpan.FromMinutes(1))
            .RunAsync(TestContext.Current.CancellationToken);

        Assert.Equal("""{"type":"tool_hold","tool_call_id":"c1"}""", sent[0].ToJsonString());
        Assert.Equal(("c1", "paid for s1"), (sent[1].Text("id"), sent[1].Text("output")));
    }

    [Fact]
    public async Task AHostedCallsUpdatesReachTheRouterBeforeItsResult()
    {
        await using var router = await TestRouter.StartAsync();
        var sent = new List<JsonObject>();
        router.OnSocket("/v1/dispatch", async peer =>
        {
            await peer.SendAsync(new { type = "ready", worker_id = "worker-1" });
            await peer.ReceiveAsync("host_tools");
            await peer.SendAsync(new { type = "tool_call", id = "c1", session_id = "s1", name = "search_flights", arguments = "{}" });
            for (var i = 0; i < 3; i++)
            {
                sent.Add(await peer.ReceiveAsync("tool_update"));
            }
            sent.Add(await peer.ReceiveAsync("tool_result"));
        });
        using var client = Fixtures.Client(router);
        client.Tools.Register("search_flights", "Searches", [], async (ToolCallContext toolCallContext, JsonElement _, CancellationToken cancellationToken) =>
        {
            await toolCallContext.UpdateAsync("Comparing 40 flights", cancellationToken);
            await toolCallContext.UpdateAsync(new ToolCallUpdate("Booking the cheapest", new JsonObject { ["cheapest"] = 129 }), cancellationToken);
            await toolCallContext.UpdateAsync(new ToolCallUpdate(Custom: new JsonObject { ["seat"] = "12A" }), cancellationToken);
            return (object?)"found 3";
        });

        await new Dispatch(new DispatchOptions { Client = client }).Host(client, TimeSpan.FromMinutes(1))
            .RunAsync(TestContext.Current.CancellationToken);

        Assert.Equal("""{"type":"tool_update","tool_call_id":"c1","text":"Comparing 40 flights"}""", sent[0].ToJsonString());
        Assert.Equal("""{"type":"tool_update","tool_call_id":"c1","text":"Booking the cheapest","custom":{"cheapest":129}}""", sent[1].ToJsonString());
        Assert.Equal("""{"type":"tool_update","tool_call_id":"c1","custom":{"seat":"12A"}}""", sent[2].ToJsonString());
        Assert.Equal("found 3", sent[3].Text("output"));
    }

    [Fact]
    public async Task AToolAddedWhileTheSessionRunsIsDeclaredAndAnsweredHere()
    {
        await using var router = await TestRouter.StartAsync();
        router.On("POST", "/v1/agents/sessions", 201, Fixtures.Session());
        router.On("POST", SessionTools, 204);
        var changed = new TaskCompletionSource();
        JsonObject? answered = null;
        router.OnSocket(Events, async peer =>
        {
            await changed.Task.WaitAsync(TestRouter.Patience);
            await peer.SendAsync(new { type = "tool_call", id = "r1", name = "reserve_table", arguments = "{}" });
            answered = await peer.ReceiveAsync("tool_result");
            await Finish(peer);
        });
        await using var agent = Agent(router);
        var session = await agent.ChatAsync(cancellationToken: TestContext.Current.CancellationToken);

        await session.AddToolsAsync(Answering("reserve_table", "Reserved."), TestContext.Current.CancellationToken);
        await Talk(session, changed);

        Assert.Equal("reserve_table", Assert.Single(router.Only("POST", SessionTools).Body!["tools"]!.AsArray()).Text("name"));
        Assert.Equal(("r1", "Reserved."), (answered.Text("tool_call_id"), answered.Text("output")));
    }

    [Fact]
    public async Task AToolTheRouterRefusedIsNotAnsweredHere()
    {
        await using var router = await TestRouter.StartAsync();
        router.On("POST", "/v1/agents/sessions", 201, Fixtures.Session());
        router.On("POST", SessionTools, 409, Fixtures.Failure("conflict", "conflict", "the session offers reserve_table from a connector"));
        var changed = new TaskCompletionSource();
        router.OnSocket(Events, async peer =>
        {
            await changed.Task.WaitAsync(TestRouter.Patience);
            await peer.SendAsync(new { type = "tool_call", id = "r1", name = "reserve_table", arguments = "{}" });
            await peer.SendAsync(new { type = "tool_call", id = "f1", name = "fast", arguments = "{}" });
            await AnsweredThenFinish(peer, "f1");
        });
        var ran = new ConcurrentQueue<string>();
        await using var agent = Agent(router);
        Answering("fast", "done", agent.Tools, ran);
        var session = await agent.ChatAsync(cancellationToken: TestContext.Current.CancellationToken);

        var refused = await Assert.ThrowsAsync<RouterException>(() =>
            session.AddToolsAsync(Answering("reserve_table", "Reserved.", ran: ran), TestContext.Current.CancellationToken));
        await Talk(session, changed);

        Assert.Equal(409, refused.Status);
        Assert.Equal(["f1"], ran);
    }

    [Fact]
    public async Task AToolWhoseAddingWasNeverAnsweredIsStillAnsweredHere()
    {
        await using var router = await TestRouter.StartAsync();
        router.On("POST", "/v1/agents/sessions", 201, Fixtures.Session());
        router.On("POST", SessionTools, -1);
        var changed = new TaskCompletionSource();
        JsonObject? answered = null;
        router.OnSocket(Events, async peer =>
        {
            await changed.Task.WaitAsync(TestRouter.Patience);
            await peer.SendAsync(new { type = "tool_call", id = "r1", name = "reserve_table", arguments = "{}" });
            answered = await peer.ReceiveAsync("tool_result");
            await Finish(peer);
        });
        await using var agent = Agent(router);
        var session = await agent.ChatAsync(cancellationToken: TestContext.Current.CancellationToken);

        var lost = await Assert.ThrowsAsync<RouterException>(() =>
            session.AddToolsAsync(Answering("reserve_table", "Reserved."), TestContext.Current.CancellationToken));
        await Talk(session, changed);

        Assert.Equal(0, lost.Status);
        // The adding whose answer was lost was not sent again.
        router.Only("POST", SessionTools);
        Assert.Equal(("r1", "Reserved."), (answered.Text("tool_call_id"), answered.Text("output")));
    }

    [Fact]
    public async Task AToolWhoseAddingWasNeverAnsweredIsWithdrawnBeforeItsNameIsAddedAgain()
    {
        await using var router = await TestRouter.StartAsync();
        router.On("POST", "/v1/agents/sessions", 201, Fixtures.Session());
        var drop = true;
        router.On("POST", SessionTools, _ => new Reply(drop ? -1 : 204));
        // The first adding never reached the router, which so has no tool by the name.
        router.On("DELETE", SessionTools + "/reserve_table", 404, NotFound());
        var changed = new TaskCompletionSource();
        JsonObject? answered = null;
        router.OnSocket(Events, async peer =>
        {
            await changed.Task.WaitAsync(TestRouter.Patience);
            await peer.SendAsync(new { type = "tool_call", id = "r1", name = "reserve_table", arguments = "{}" });
            answered = await peer.ReceiveAsync("tool_result");
            await Finish(peer);
        });
        await using var agent = Agent(router);
        var session = await agent.ChatAsync(cancellationToken: TestContext.Current.CancellationToken);
        await Assert.ThrowsAsync<RouterException>(() =>
            session.AddToolsAsync(Answering("reserve_table", "First."), TestContext.Current.CancellationToken));
        drop = false;

        await session.AddToolsAsync(Answering("reserve_table", "Second."), TestContext.Current.CancellationToken);
        await Talk(session, changed);

        var changes = router.Requests.Where(seen => seen.Path.StartsWith(SessionTools, StringComparison.Ordinal)).Select(seen => $"{seen.Method} {seen.Path}");
        Assert.Equal([$"DELETE {SessionTools}/reserve_table", $"POST {SessionTools}"], changes.TakeLast(2));
        Assert.Equal("Second.", answered.Text("output"));
    }

    [Fact]
    public async Task AToolWhoseAddingAGatewayFailedIsStillAnsweredHereAndWithdrawnBeforeItsNameIsAddedAgain()
    {
        await using var router = await TestRouter.StartAsync();
        router.On("POST", "/v1/agents/sessions", 201, Fixtures.Session());
        // The router adds the tool and a gateway in front of it answers the first adding 502. It
        // refuses a name it already has, and withdraws one it has.
        var offered = false;
        var adds = 0;
        router.On("POST", SessionTools, _ =>
        {
            if (offered)
            {
                return new Reply(409, Fixtures.Failure("conflict", "conflict", "the session already has a tool called reserve_table"));
            }
            offered = true;
            return Interlocked.Increment(ref adds) == 1 ? new Reply(502, "Bad Gateway") : new Reply(204);
        });
        router.On("DELETE", SessionTools + "/reserve_table", _ =>
        {
            if (!offered)
            {
                return new Reply(404, NotFound());
            }
            offered = false;
            return new Reply(204);
        });
        var failed = new TaskCompletionSource();
        var kept = new TaskCompletionSource<JsonObject>();
        var changed = new TaskCompletionSource();
        JsonObject? answered = null;
        router.OnSocket(Events, async peer =>
        {
            await failed.Task.WaitAsync(TestRouter.Patience);
            await peer.SendAsync(new { type = "tool_call", id = "r1", name = "reserve_table", arguments = "{}" });
            kept.SetResult(await peer.ReceiveAsync("tool_result"));
            await changed.Task.WaitAsync(TestRouter.Patience);
            await peer.SendAsync(new { type = "tool_call", id = "r2", name = "reserve_table", arguments = "{}" });
            answered = await peer.ReceiveAsync("tool_result");
            await Finish(peer);
        });
        await using var agent = Agent(router);
        var session = await agent.ChatAsync(cancellationToken: TestContext.Current.CancellationToken);
        var failure = await Assert.ThrowsAsync<RouterException>(() =>
            session.AddToolsAsync(Answering("reserve_table", "First."), TestContext.Current.CancellationToken));
        failed.SetResult();
        var first = await kept.Task.WaitAsync(TestRouter.Patience, TestContext.Current.CancellationToken);

        await session.AddToolsAsync(Answering("reserve_table", "Second."), TestContext.Current.CancellationToken);
        await Talk(session, changed);

        Assert.Equal(502, failure.Status);
        Assert.Equal(("r1", "First."), (first.Text("tool_call_id"), first.Text("output")));
        Assert.Equal([$"POST {SessionTools}", $"DELETE {SessionTools}/reserve_table", $"POST {SessionTools}"],
            router.Requests.Where(seen => seen.Path.StartsWith(SessionTools, StringComparison.Ordinal)).Select(seen => $"{seen.Method} {seen.Path}"));
        Assert.Equal(("r2", "Second."), (answered.Text("tool_call_id"), answered.Text("output")));
    }

    [Fact]
    public async Task RemovingAToolWhoseAddingWasNeverAnsweredTakesNotFoundAsDone()
    {
        await using var router = await TestRouter.StartAsync();
        router.On("POST", "/v1/agents/sessions", 201, Fixtures.Session());
        router.On("POST", SessionTools, -1);
        router.On("DELETE", SessionTools + "/reserve_table", 404, NotFound());
        var changed = new TaskCompletionSource();
        router.OnSocket(Events, async peer =>
        {
            await changed.Task.WaitAsync(TestRouter.Patience);
            await peer.SendAsync(new { type = "tool_call", id = "r1", name = "reserve_table", arguments = "{}" });
            await peer.SendAsync(new { type = "tool_call", id = "f1", name = "fast", arguments = "{}" });
            await AnsweredThenFinish(peer, "f1");
        });
        var ran = new ConcurrentQueue<string>();
        await using var agent = Agent(router);
        Answering("fast", "done", agent.Tools, ran);
        var session = await agent.ChatAsync(cancellationToken: TestContext.Current.CancellationToken);
        await Assert.ThrowsAsync<RouterException>(() =>
            session.AddToolsAsync(Answering("reserve_table", "Reserved.", ran: ran), TestContext.Current.CancellationToken));

        await session.RemoveToolAsync("reserve_table", TestContext.Current.CancellationToken);
        await Talk(session, changed);

        Assert.Equal(["f1"], ran);
    }

    [Theory]
    [InlineData(-1)]
    [InlineData(502)]
    public async Task AToolWhoseRemovalWasNeverAnsweredRunsUntilItIsRemovedAgain(int lostWith)
    {
        await using var router = await TestRouter.StartAsync();
        router.On("POST", "/v1/agents/sessions", 201, Fixtures.Session());
        // The first removal is made and its answer lost, or a gateway's; the router has no tool by the name after it.
        var removals = 0;
        router.On("DELETE", SessionTools + "/reserve_table", _ => Interlocked.Increment(ref removals) == 1 ? new Reply(lostWith, "Bad Gateway") : new Reply(404, NotFound()));
        var lost = new TaskCompletionSource();
        var answered = new TaskCompletionSource<JsonObject>();
        router.OnSocket(Events, async peer =>
        {
            await lost.Task.WaitAsync(TestRouter.Patience);
            await peer.SendAsync(new { type = "tool_call", id = "r1", name = "reserve_table", arguments = "{}" });
            answered.SetResult(await peer.ReceiveAsync("tool_result"));
            await Finish(peer);
        });
        await using var agent = Agent(router);
        Answering("reserve_table", "Reserved.", agent.Tools);
        var session = await agent.ChatAsync(cancellationToken: TestContext.Current.CancellationToken);
        var failure = await Assert.ThrowsAsync<RouterException>(() =>
            session.RemoveToolAsync("reserve_table", TestContext.Current.CancellationToken));
        lost.SetResult();
        var kept = await answered.Task.WaitAsync(TestRouter.Patience, TestContext.Current.CancellationToken);

        await session.RemoveToolAsync("reserve_table", TestContext.Current.CancellationToken);
        await session.CloseAsync(TestContext.Current.CancellationToken);

        Assert.Equal(Math.Max(lostWith, 0), failure.Status);
        Assert.Equal("Reserved.", kept.Text("output"));
        // One request a removal: the one whose answer was lost was not sent again.
        Assert.Equal(2, router.To("DELETE", SessionTools + "/reserve_table").Count);
    }

    [Fact]
    public async Task ARemovalWhoseAnswerWasLostIsNotSentAgainByAnHttpClientThatWaitsToContinue()
    {
        await using var router = await TestRouter.StartAsync();
        router.On("POST", "/v1/agents/sessions", 201, Fixtures.Session());
        // Made and its answer lost; the router has no tool by the name after it.
        var removals = 0;
        router.On("DELETE", SessionTools + "/reserve_table", _ => Interlocked.Increment(ref removals) == 1 ? new Reply(-1) : new Reply(404, NotFound()));
        router.OnSocket(Events, peer => peer.ReceiveAsync("close"));
        // Every request waits for 100 Continue before its body, an empty one too, and one whose
        // connection closes before it began its body is sent again.
        using var http = new HttpClient();
        http.DefaultRequestHeaders.ExpectContinue = true;
        await using var agent = new Agent(new AgentOptions
        {
            Name = "jean",
            Client = new VisionAgentsClient(http, new VisionAgentsOptions { Url = router.Url, CustomerId = "examples" }),
        });
        Answering("reserve_table", "Reserved.", agent.Tools);
        var session = await agent.ChatAsync(cancellationToken: TestContext.Current.CancellationToken);

        var failure = await Assert.ThrowsAsync<RouterException>(() =>
            session.RemoveToolAsync("reserve_table", TestContext.Current.CancellationToken));
        await session.CloseAsync(TestContext.Current.CancellationToken);

        Assert.Equal(0, failure.Status);
        router.Only("DELETE", SessionTools + "/reserve_table");
    }

    [Fact]
    public async Task ANameWhoseRemovalWasNeverAnsweredIsWithdrawnBeforeItIsAddedAgain()
    {
        await using var router = await TestRouter.StartAsync();
        router.On("POST", "/v1/agents/sessions", 201, Fixtures.Session());
        router.On("POST", SessionTools, 204);
        var removals = 0;
        router.On("DELETE", SessionTools + "/reserve_table", _ => Interlocked.Increment(ref removals) == 1 ? new Reply(-1) : new Reply(404, NotFound()));
        var changed = new TaskCompletionSource();
        JsonObject? answered = null;
        router.OnSocket(Events, async peer =>
        {
            await changed.Task.WaitAsync(TestRouter.Patience);
            await peer.SendAsync(new { type = "tool_call", id = "r1", name = "reserve_table", arguments = "{}" });
            answered = await peer.ReceiveAsync("tool_result");
            await Finish(peer);
        });
        await using var agent = Agent(router);
        var session = await agent.ChatAsync(cancellationToken: TestContext.Current.CancellationToken);
        await session.AddToolsAsync(Answering("reserve_table", "First."), TestContext.Current.CancellationToken);
        await Assert.ThrowsAsync<RouterException>(() => session.RemoveToolAsync("reserve_table", TestContext.Current.CancellationToken));

        await session.AddToolsAsync(Answering("reserve_table", "Second."), TestContext.Current.CancellationToken);
        await Talk(session, changed);

        Assert.Equal([$"POST {SessionTools}", $"DELETE {SessionTools}/reserve_table", $"DELETE {SessionTools}/reserve_table", $"POST {SessionTools}"],
            router.Requests.Where(seen => seen.Path.StartsWith(SessionTools, StringComparison.Ordinal)).Select(seen => $"{seen.Method} {seen.Path}"));
        Assert.Equal("Second.", answered.Text("output"));
    }

    [Fact]
    public async Task ARemovalCancelledInFlightIsSettledByTheNextOne()
    {
        await using var router = await TestRouter.StartAsync();
        router.On("POST", "/v1/agents/sessions", 201, Fixtures.Session());
        var arrived = new TaskCompletionSource(TaskCreationOptions.RunContinuationsAsynchronously);
        using var release = new ManualResetEventSlim();
        var removals = 0;
        router.On("DELETE", SessionTools + "/reserve_table", _ =>
        {
            if (Interlocked.Increment(ref removals) > 1)
            {
                return new Reply(404, NotFound());
            }
            // Made, and answered only after the caller stopped waiting.
            arrived.SetResult();
            release.Wait(TestRouter.Patience);
            return new Reply(204);
        });
        router.OnSocket(Events, peer => peer.ReceiveAsync("close"));
        await using var agent = Agent(router);
        Answering("reserve_table", "Reserved.", agent.Tools);
        var session = await agent.ChatAsync(cancellationToken: TestContext.Current.CancellationToken);
        using var cancel = CancellationTokenSource.CreateLinkedTokenSource(TestContext.Current.CancellationToken);
        var removing = session.RemoveToolAsync("reserve_table", cancel.Token);
        await arrived.Task.WaitAsync(TestRouter.Patience, TestContext.Current.CancellationToken);
        await cancel.CancelAsync();
        await Assert.ThrowsAnyAsync<OperationCanceledException>(() => removing);
        release.Set();

        await session.RemoveToolAsync("reserve_table", TestContext.Current.CancellationToken);
        await session.CloseAsync(TestContext.Current.CancellationToken);

        Assert.Equal(2, removals);
    }

    [Fact]
    public async Task AnAddingCancelledInFlightIsWithdrawnBeforeItsNameIsAddedAgain()
    {
        await using var router = await TestRouter.StartAsync();
        router.On("POST", "/v1/agents/sessions", 201, Fixtures.Session());
        var arrived = new TaskCompletionSource(TaskCreationOptions.RunContinuationsAsynchronously);
        using var release = new ManualResetEventSlim();
        var adds = 0;
        router.On("POST", SessionTools, _ =>
        {
            if (Interlocked.Increment(ref adds) == 1)
            {
                // Made, and answered only after the caller stopped waiting.
                arrived.SetResult();
                release.Wait(TestRouter.Patience);
            }
            return new Reply(204);
        });
        router.On("DELETE", SessionTools + "/reserve_table", 204);
        var changed = new TaskCompletionSource();
        JsonObject? answered = null;
        router.OnSocket(Events, async peer =>
        {
            await changed.Task.WaitAsync(TestRouter.Patience);
            await peer.SendAsync(new { type = "tool_call", id = "r1", name = "reserve_table", arguments = "{}" });
            answered = await peer.ReceiveAsync("tool_result");
            await Finish(peer);
        });
        await using var agent = Agent(router);
        var session = await agent.ChatAsync(cancellationToken: TestContext.Current.CancellationToken);
        using var cancel = CancellationTokenSource.CreateLinkedTokenSource(TestContext.Current.CancellationToken);
        var adding = session.AddToolsAsync(Answering("reserve_table", "First."), cancel.Token);
        await arrived.Task.WaitAsync(TestRouter.Patience, TestContext.Current.CancellationToken);
        await cancel.CancelAsync();
        await Assert.ThrowsAnyAsync<OperationCanceledException>(() => adding);
        release.Set();

        await session.AddToolsAsync(Answering("reserve_table", "Second."), TestContext.Current.CancellationToken);
        await Talk(session, changed);

        Assert.Equal([$"POST {SessionTools}", $"DELETE {SessionTools}/reserve_table", $"POST {SessionTools}"],
            router.Requests.Where(seen => seen.Path.StartsWith(SessionTools, StringComparison.Ordinal)).Select(seen => $"{seen.Method} {seen.Path}"));
        Assert.Equal("Second.", answered.Text("output"));
    }

    [Fact]
    public async Task AToolRemovedFromTheSessionIsNoLongerAnsweredHere()
    {
        await using var router = await TestRouter.StartAsync();
        router.On("POST", "/v1/agents/sessions", 201, Fixtures.Session());
        router.On("POST", SessionTools, 204);
        router.On("DELETE", SessionTools + "/*", 204);
        var changed = new TaskCompletionSource();
        router.OnSocket(Events, async peer =>
        {
            await changed.Task.WaitAsync(TestRouter.Patience);
            await peer.SendAsync(new { type = "tool_call", id = "r1", name = "reserve_table", arguments = "{}" });
            await peer.SendAsync(new { type = "tool_call", id = "w1", name = "order_wine", arguments = "{}" });
            await peer.SendAsync(new { type = "tool_call", id = "f1", name = "fast", arguments = "{}" });
            await AnsweredThenFinish(peer, "f1");
        });
        var ran = new ConcurrentQueue<string>();
        await using var agent = Agent(router);
        Answering("fast", "done", agent.Tools, ran);
        Answering("reserve_table", "Reserved.", agent.Tools, ran);
        var session = await agent.ChatAsync(cancellationToken: TestContext.Current.CancellationToken);
        await session.AddToolsAsync(Answering("order_wine", "Ordered.", ran: ran), TestContext.Current.CancellationToken);

        await session.RemoveToolAsync("reserve_table", TestContext.Current.CancellationToken);
        await session.RemoveToolAsync("order_wine", TestContext.Current.CancellationToken);
        await Talk(session, changed);

        Assert.Equal([$"{SessionTools}/reserve_table", $"{SessionTools}/order_wine"],
            router.Requests.Where(seen => seen.Method == "DELETE").Select(seen => seen.Path));
        Assert.Equal(["f1"], ran);
    }

    [Fact]
    public async Task AForkRunsTheToolsTheSessionHasWhenItForks()
    {
        await using var router = await TestRouter.StartAsync();
        router.On("POST", "/v1/agents/sessions", 201, Fixtures.Session());
        router.On("POST", SessionTools, 204);
        router.On("DELETE", SessionTools + "/*", 204);
        router.On("POST", "/v1/agents/sessions/s1/fork", 201, Fixtures.Session("s2"));
        router.OnSocket(Events, peer => peer.ReceiveAsync("close"));
        var answered = new TaskCompletionSource<Dictionary<string, string>>();
        router.OnSocket("/v1/agents/sessions/s2/events", async peer =>
        {
            await peer.SendAsync(new { type = "tool_call", id = "r1", name = "reserve_table", arguments = "{}" });
            await peer.SendAsync(new { type = "tool_call", id = "w1", name = "order_wine", arguments = "{}" });
            await peer.SendAsync(new { type = "tool_call", id = "f1", name = "fast", arguments = "{}" });
            answered.SetResult(await AnsweredThenFinish(peer, "w1", "f1"));
        });
        var ran = new ConcurrentQueue<string>();
        await using var agent = Agent(router);
        Answering("fast", "done", agent.Tools, ran);
        Answering("reserve_table", "Reserved.", agent.Tools, ran);
        var session = await agent.ChatAsync(cancellationToken: TestContext.Current.CancellationToken);
        await session.AddToolsAsync(Answering("order_wine", "Ordered.", ran: ran), TestContext.Current.CancellationToken);
        await session.RemoveToolAsync("reserve_table", TestContext.Current.CancellationToken);

        var forked = await session.ForkAsync(cancellationToken: TestContext.Current.CancellationToken);
        await FirstAsync(forked);
        await forked.CloseAsync(TestContext.Current.CancellationToken);
        await session.CloseAsync(TestContext.Current.CancellationToken);

        Assert.Equal("Ordered.", (await answered.Task.WaitAsync(TestRouter.Patience, TestContext.Current.CancellationToken))["w1"]);
        Assert.Equal(["f1", "w1"], ran.Order());
    }

    [Fact]
    public async Task AForkWithdrawsANameWhoseAddingWasNeverAnsweredBeforeAddingItAgain()
    {
        const string forkTools = "/v1/agents/sessions/s2/tools";
        await using var router = await TestRouter.StartAsync();
        router.On("POST", "/v1/agents/sessions", 201, Fixtures.Session());
        // The adding never reached the router, so neither the session nor its fork has the tool.
        router.On("POST", SessionTools, -1);
        router.On("POST", "/v1/agents/sessions/s1/fork", 201, Fixtures.Session("s2"));
        router.On("DELETE", forkTools + "/reserve_table", 404, NotFound());
        router.On("POST", forkTools, 204);
        router.OnSocket(Events, peer => peer.ReceiveAsync("close"));
        var changed = new TaskCompletionSource();
        JsonObject? answered = null;
        router.OnSocket("/v1/agents/sessions/s2/events", async peer =>
        {
            await changed.Task.WaitAsync(TestRouter.Patience);
            await peer.SendAsync(new { type = "tool_call", id = "r1", name = "reserve_table", arguments = "{}" });
            answered = await peer.ReceiveAsync("tool_result");
            await Finish(peer);
        });
        await using var agent = Agent(router);
        var session = await agent.ChatAsync(cancellationToken: TestContext.Current.CancellationToken);
        await Assert.ThrowsAsync<RouterException>(() =>
            session.AddToolsAsync(Answering("reserve_table", "First."), TestContext.Current.CancellationToken));
        var forked = await session.ForkAsync(cancellationToken: TestContext.Current.CancellationToken);

        await forked.AddToolsAsync(Answering("reserve_table", "Second."), TestContext.Current.CancellationToken);
        await Talk(forked, changed);
        await session.CloseAsync(TestContext.Current.CancellationToken);

        Assert.Equal([$"DELETE {forkTools}/reserve_table", $"POST {forkTools}"],
            router.Requests.Where(seen => seen.Path.StartsWith(forkTools, StringComparison.Ordinal)).Select(seen => $"{seen.Method} {seen.Path}"));
        Assert.Equal(("r1", "Second."), (answered.Text("tool_call_id"), answered.Text("output")));
    }

    [Fact]
    public async Task AFunctionRegisteredLaterUnderAnAddedNameDoesNotReplaceIt()
    {
        await using var router = await TestRouter.StartAsync();
        router.On("POST", "/v1/agents/sessions", 201, Fixtures.Session());
        router.On("POST", SessionTools, 204);
        var changed = new TaskCompletionSource();
        JsonObject? answered = null;
        router.OnSocket(Events, async peer =>
        {
            await changed.Task.WaitAsync(TestRouter.Patience);
            await peer.SendAsync(new { type = "tool_call", id = "r1", name = "reserve_table", arguments = "{}" });
            answered = await peer.ReceiveAsync("tool_result");
            await Finish(peer);
        });
        await using var agent = Agent(router);
        var session = await agent.ChatAsync(cancellationToken: TestContext.Current.CancellationToken);
        var more = Answering("reserve_table", "First.");
        await session.AddToolsAsync(more, TestContext.Current.CancellationToken);

        Answering("reserve_table", "Second.", more);
        await Talk(session, changed);

        Assert.Equal("First.", answered.Text("output"));
    }

    [Fact]
    public async Task ANameWithdrawnFromTheAgentsToolsRunsTheFunctionAddedInItsPlace()
    {
        await using var router = await TestRouter.StartAsync();
        router.On("POST", "/v1/agents/sessions", 201, Fixtures.Session());
        router.On("POST", SessionTools, 204);
        router.On("DELETE", SessionTools + "/*", 204);
        var changed = new TaskCompletionSource();
        JsonObject? answered = null;
        router.OnSocket(Events, async peer =>
        {
            await changed.Task.WaitAsync(TestRouter.Patience);
            await peer.SendAsync(new { type = "tool_call", id = "r1", name = "reserve_table", arguments = "{}" });
            answered = await peer.ReceiveAsync("tool_result");
            await Finish(peer);
        });
        await using var agent = Agent(router);
        Answering("reserve_table", "Reserved.", agent.Tools);
        var session = await agent.ChatAsync(cancellationToken: TestContext.Current.CancellationToken);
        await session.RemoveToolAsync("reserve_table", TestContext.Current.CancellationToken);

        await session.AddToolsAsync(Answering("reserve_table", "Replaced."), TestContext.Current.CancellationToken);
        await Talk(session, changed);

        Assert.Equal("Replaced.", answered.Text("output"));
    }

    [Fact]
    public async Task AddingANameTheSessionRunsKeepsItsFunction()
    {
        await using var router = await TestRouter.StartAsync();
        router.On("POST", "/v1/agents/sessions", 201, Fixtures.Session());
        // Lost, so nothing here says whether the router took it; it never replaces a tool the session has.
        router.On("POST", SessionTools, -1);
        var changed = new TaskCompletionSource();
        JsonObject? answered = null;
        router.OnSocket(Events, async peer =>
        {
            await changed.Task.WaitAsync(TestRouter.Patience);
            await peer.SendAsync(new { type = "tool_call", id = "f1", name = "fast", arguments = "{}" });
            answered = await peer.ReceiveAsync("tool_result");
            await Finish(peer);
        });
        await using var agent = Agent(router);
        Payments(agent.Tools);
        var session = await agent.ChatAsync(cancellationToken: TestContext.Current.CancellationToken);

        await Assert.ThrowsAsync<RouterException>(() =>
            session.AddToolsAsync(Answering("fast", "Replaced."), TestContext.Current.CancellationToken));
        await Talk(session, changed);

        Assert.Equal("done", answered.Text("output"));
    }

    [Fact]
    public async Task ASessionChangesItsToolsOneAtATime()
    {
        await using var router = await TestRouter.StartAsync();
        router.On("POST", "/v1/agents/sessions", 201, Fixtures.Session());
        var inFlight = 0;
        var overlapped = false;
        using var second = new ManualResetEventSlim();
        Func<Seen, Reply> change = _ =>
        {
            if (Interlocked.Increment(ref inFlight) > 1)
            {
                overlapped = true;
                second.Set();
            }
            // Held a while, so a change sent without waiting for this one arrives during it.
            second.Wait(TimeSpan.FromMilliseconds(500));
            Interlocked.Decrement(ref inFlight);
            return new Reply(204);
        };
        router.On("POST", SessionTools, change).On("DELETE", SessionTools + "/fast", change);
        router.OnSocket(Events, peer => peer.ReceiveAsync("close"));
        await using var agent = Agent(router);
        Payments(agent.Tools);
        var session = await agent.ChatAsync(cancellationToken: TestContext.Current.CancellationToken);

        await Task.WhenAll(
            session.AddToolsAsync(Answering("reserve_table", "Reserved."), TestContext.Current.CancellationToken),
            session.RemoveToolAsync("fast", TestContext.Current.CancellationToken));
        await session.CloseAsync(TestContext.Current.CancellationToken);

        Assert.Equal(2, router.Requests.Count(seen => seen.Path.StartsWith(SessionTools, StringComparison.Ordinal)));
        Assert.False(overlapped);
    }

    [Fact]
    public async Task AToolAddedToOneSessionIsNotTheAgentsNextSessions()
    {
        await using var router = await TestRouter.StartAsync();
        var opened = 0;
        router.On("POST", "/v1/agents/sessions", _ => new Reply(201, Fixtures.Session(Interlocked.Increment(ref opened) == 1 ? "s1" : "s2")));
        router.On("POST", SessionTools, 204);
        router.OnSocket(Events, peer => peer.ReceiveAsync("close"));
        router.OnSocket("/v1/agents/sessions/s2/events", async peer =>
        {
            await peer.SendAsync(new { type = "tool_call", id = "r1", name = "reserve_table", arguments = "{}" });
            await peer.SendAsync(new { type = "tool_call", id = "f1", name = "fast", arguments = "{}" });
            await AnsweredThenFinish(peer, "f1");
        });
        var ran = new ConcurrentQueue<string>();
        await using var agent = Agent(router);
        Answering("fast", "done", agent.Tools, ran);
        var first = await agent.ChatAsync(cancellationToken: TestContext.Current.CancellationToken);
        await first.AddToolsAsync(Answering("reserve_table", "Reserved.", ran: ran), TestContext.Current.CancellationToken);

        await Converse(agent);
        await first.CloseAsync(TestContext.Current.CancellationToken);

        Assert.Equal(["f1"], ran);
    }

    /// <summary>
    /// <c>pay</c>, which holds itself through interruptions and asks first, <c>slow</c>,
    /// which may be interrupted, and <c>fast</c>.
    /// </summary>
    private static void Payments(Tools tools) => tools
        .Register("pay", "Pays", Amount(), async (ToolCallContext toolCallContext, JsonElement _, CancellationToken cancellationToken) =>
        {
            await toolCallContext.DisallowInterruptionsAsync(cancellationToken);
            await Task.Delay(200, cancellationToken);
            return (object?)("paid for " + toolCallContext.SessionId);
        }, approval: "Pay {amount}?")
        .Register("slow", "Takes a while", [], async (JsonElement _, CancellationToken cancellationToken) =>
        {
            await Task.Delay(TimeSpan.FromSeconds(30), cancellationToken);
            return (object?)"done";
        }, interruptible: true)
        .Register("fast", "Does not", [], (JsonElement _, CancellationToken _) => Task.FromResult<object?>("done"));

    /// <summary>
    /// A function called <paramref name="name"/> that answers <paramref name="output"/>, so a test can tell which one ran,
    /// and puts the id of each call it runs in <paramref name="ran"/>.
    /// </summary>
    private static Tools Answering(string name, string output, Tools? into = null, ConcurrentQueue<string>? ran = null) =>
        (into ?? new Tools()).Register(name, "Answers " + output, [], (ToolCallContext call, JsonElement _, CancellationToken _) =>
        {
            ran?.Enqueue(call.Id);
            return Task.FromResult<object?>(output);
        });

    private static object NotFound() => Fixtures.Failure("not_found", "not_found", "the session's caller has no tool by that name");

    /// <summary>Lets the router carry on once the session's tools changed, then chats until the agent says something, and closes.</summary>
    private static async Task Talk(Session session, TaskCompletionSource changed)
    {
        changed.SetResult();
        await FirstAsync(session);
        await session.CloseAsync(TestContext.Current.CancellationToken);
    }

    /// <summary>
    /// What each of the calls named was answered with, by id, once all of them are; then the agent
    /// responds and the session is closed. A call that must not run is not looked for here: its
    /// function runs off the session's read loop, so its answer could come after these, or not
    /// before the close. A test reads what ran from the functions once the session has closed,
    /// when every call it was sent has been handed to one or to none.
    /// </summary>
    private static async Task<Dictionary<string, string>> AnsweredThenFinish(Peer peer, params string[] ids)
    {
        Dictionary<string, string> answered = [];
        while (!ids.All(answered.ContainsKey))
        {
            var result = await peer.ReceiveAsync("tool_result");
            answered[result.Text("tool_call_id")] = result.Text("output");
        }
        await Finish(peer);
        return answered;
    }

    private static JsonObject Amount() => new()
    {
        ["type"] = "object",
        ["properties"] = new JsonObject { ["amount"] = new JsonObject { ["type"] = "number" } },
    };

    private static Agent Agent(TestRouter router) =>
        new(new AgentOptions { Name = "jean", Client = Fixtures.Client(router) });

    private static async Task Finish(Peer peer)
    {
        await peer.SendAsync(new { type = "responded", text = "Done.", pending_work = false });
        await peer.ReceiveAsync("close");
    }

    /// <summary>Chats until the agent first says something, then closes.</summary>
    private static async Task Converse(Agent agent)
    {
        var session = await agent.ChatAsync(cancellationToken: TestContext.Current.CancellationToken);
        await FirstAsync(session);
        await session.CloseAsync(TestContext.Current.CancellationToken);
    }

    private static async Task<SessionEvent> FirstAsync(Session session)
    {
        using var patience = new CancellationTokenSource(TestRouter.Patience);
        await foreach (var happened in session.EventsAsync(patience.Token))
        {
            return happened;
        }
        throw new InvalidOperationException("the session ended with nothing said");
    }

    /// <summary>The next frame, or null when nothing more comes for a moment.</summary>
    private static async Task<JsonObject?> Next(Peer peer)
    {
        try
        {
            return await peer.ReceiveAsync().WaitAsync(TimeSpan.FromMilliseconds(200));
        }
        catch (TimeoutException)
        {
            return null;
        }
    }
}
