using System.Text.Json;
using System.Text.Json.Nodes;

namespace GetStream.VisionAgents.Tests;

public sealed class ToolCallTests
{
    private const string Events = "/v1/agents/sessions/s1/events";

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
            sent.Add(await peer.ReceiveAsync());
            sent.Add(await peer.ReceiveAsync());
            sent.Add(await peer.ReceiveAsync("tool_result"));
            await Finish(peer);
        });
        await using var agent = Agent(router);
        agent.Tools.Register("book", "Books a flight", [], async (JsonElement _, ToolCall toolCallContext, CancellationToken cancellationToken) =>
        {
            await toolCallContext.UpdateAsync("Looking at flights", cancellationToken);
            await toolCallContext.SayAsync("One moment.", cancellationToken);
            var user = toolCallContext.User;
            return (object?)$"{toolCallContext.Id} {toolCallContext.SessionId} {toolCallContext.TurnId} {user.Id} {user.Name} " +
                $"{user.Custom.Text("tier")} {toolCallContext.Conversation.Cid} {toolCallContext.Approved}";
        });

        await Converse(agent);

        Assert.Equal("""{"type":"tool_update","tool_call_id":"t1","text":"Looking at flights","verbatim":false}""", sent[0].ToJsonString());
        Assert.Equal("""{"type":"tool_update","tool_call_id":"t1","text":"One moment.","verbatim":true}""", sent[1].ToJsonString());
        Assert.Equal("t1 s1 turn-1 u1 Ada gold messaging:c1 True", sent[2].Text("output"));
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
        using var patience = new CancellationTokenSource(TimeSpan.FromSeconds(5));
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
        var agent = client.Agent("payments");
        Payments(agent.Tools);

        await new Dispatch(new DispatchOptions { Client = client }).Host(agent, TimeSpan.FromMinutes(1))
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
        var agent = client.Agent("payments");
        Payments(agent.Tools);

        await new Dispatch(new DispatchOptions { Client = client }).Host(agent, TimeSpan.FromMinutes(1))
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
        var agent = client.Agent("payments");
        Payments(agent.Tools);

        await new Dispatch(new DispatchOptions { Client = client }).Host(agent, TimeSpan.FromMinutes(1))
            .RunAsync(TestContext.Current.CancellationToken);

        Assert.Equal("""{"type":"tool_hold","tool_call_id":"c1"}""", sent[0].ToJsonString());
        Assert.Equal(("c1", "paid for s1"), (sent[1].Text("id"), sent[1].Text("output")));
    }

    /// <summary>
    /// <c>pay</c>, which holds itself through interruptions and asks first, <c>slow</c>,
    /// which may be interrupted, and <c>fast</c>.
    /// </summary>
    private static void Payments(Tools tools) => tools
        .Register("pay", "Pays", Amount(), async (JsonElement _, ToolCall toolCallContext, CancellationToken cancellationToken) =>
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
        using var patience = new CancellationTokenSource(TimeSpan.FromSeconds(5));
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
