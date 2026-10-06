using System.Collections.Concurrent;
using System.Net.WebSockets;
using System.Text.Json.Nodes;

namespace GetStream.VisionAgents.Tests;

public sealed class DispatchTests
{
    [Fact]
    public async Task ACallIsHandedOverAndTheRouterToldWhenItIsDone()
    {
        await using var router = await TestRouter.StartAsync();
        var told = new List<JsonObject>();
        router.OnSocket("/v1/dispatch", async peer =>
        {
            await peer.SendAsync(new { type = "ready", worker_id = "worker-1" });
            await peer.SendAsync(new
            {
                type = "call", work_id = "w1", call_id = "c1", call_type = "agent", called_number = "+15552223333", caller_number = "+15550001111",
                custom = new { campaign = "spring", ignored = 3 }, at = "2026-09-24T10:00:00Z",
            });
            told.Add(await peer.ReceiveAsync("done"));
            await peer.SendAsync(new { type = "call", work_id = "w2", call_id = "c2", call_type = "agent", called_number = "+15552223333" });
            told.Add(await peer.ReceiveAsync("done"));
        });
        using var client = Fixtures.Client(router);
        var dispatch = new Dispatch(new DispatchOptions { Client = client, Capacity = 2 });
        var answered = new ConcurrentQueue<InboundCall>();
        dispatch.WaitForCall(call =>
        {
            answered.Enqueue(call);
            return call.CallId == "c2" ? throw new InvalidOperationException("nobody is free") : Task.CompletedTask;
        });

        await dispatch.RunAsync(TestContext.Current.CancellationToken);

        Assert.Equal("worker-1", dispatch.WorkerId);
        var query = router.Only("GET", "/v1/dispatch").Query;
        Assert.Equal(("2", "0", "call"), (query["capacity"], query["active"], query["handles"]));
        var call = answered.First();
        Assert.Equal(("c1", "+15552223333", "+15550001111"), (call.CallId, call.CalledNumber, call.CallerNumber));
        Assert.Equal(new Dictionary<string, string> { ["campaign"] = "spring" }, call.Custom);
        Assert.Equal(new DateTimeOffset(2026, 9, 24, 10, 0, 0, TimeSpan.Zero), call.At);
        Assert.Equal("w1", told[0].Text("work_id"));
        Assert.Null(told[0]["error"]);
        Assert.Equal(("w2", "nobody is free"), (told[1].Text("work_id"), told[1].Text("error")));
    }

    [Fact]
    public async Task AMessageNoHandlerAnswersIsDoneWithAnError()
    {
        await using var router = await TestRouter.StartAsync();
        JsonObject? done = null;
        router.OnSocket("/v1/dispatch", async peer =>
        {
            await peer.SendAsync(new { type = "message", work_id = "w1", channel_id = "ch1", text = "hi" });
            done = await peer.ReceiveAsync("done");
        });
        using var client = Fixtures.Client(router);
        var dispatch = new Dispatch(new DispatchOptions { Client = client });
        dispatch.WaitForCall(_ => Task.CompletedTask);

        await dispatch.RunAsync(TestContext.Current.CancellationToken);

        Assert.Equal(("w1", "this worker answers no messages"), (done.Text("work_id"), done.Text("error")));
    }

    [Fact]
    public async Task AMessageWrittenToARunningSessionCarriesItsSessionAndCommand()
    {
        await using var router = await TestRouter.StartAsync();
        JsonObject? done = null;
        router.OnSocket("/v1/dispatch", async peer =>
        {
            await peer.SendAsync(new
            {
                type = "message", work_id = "w1", agent_id = "support", session_id = "s1", command_id = "cmd-1", user_id = "ada", text = "hi",
            });
            done = await peer.ReceiveAsync("done");
        });
        using var client = Fixtures.Client(router);
        var dispatch = new Dispatch(new DispatchOptions { Client = client });
        InboundMessage? handed = null;
        dispatch.WaitForMessage(message =>
        {
            handed = message;
            return Task.CompletedTask;
        });

        await dispatch.RunAsync(TestContext.Current.CancellationToken);

        Assert.Equal("message", router.Only("GET", "/v1/dispatch").Query["handles"]);
        Assert.Equal(("s1", "cmd-1", "", "hi"), (handed!.SessionId, handed.CommandId, handed.ChannelId, handed.Text));
        Assert.Equal("w1", done.Text("work_id"));
        Assert.Null(done!["error"]);
    }

    [Fact]
    public async Task AnsweringAMessageCreatesAResponseWithTheServerCredentialActingForItsWriter()
    {
        await using var router = await TestRouter.StartAsync();
        router.On("POST", "/v1/agents/sessions/s1/responses", 202, new { id = "r1", session_id = "s1", status = "running" });
        using var client = new VisionAgentsClient(new VisionAgentsOptions { Url = router.Url, ApiKey = "key", ApiSecret = Fixtures.Secret });
        var dispatch = new Dispatch(new DispatchOptions { Client = client });

        var response = await dispatch.AnswerAsync(
            new InboundMessage { ChannelId = "", SessionId = "s1", CommandId = "cmd-1", UserId = "ada", Text = "what does it cost?" },
            TestContext.Current.CancellationToken);

        Assert.Equal("r1", response.Id);
        var seen = router.Only("POST", "/v1/agents/sessions/s1/responses");
        Assert.Equal(("what does it cost?", "cmd-1"), (seen.Body.Text("text"), seen.Body.Text("command_id")));
        Assert.Equal(("ada", "server"), (seen.Headers["X-Stream-User-Id"], seen.Headers["Stream-Auth-Type"]));
        Assert.True(Fixtures.Claims(seen.Headers["Authorization"])["server"]!.GetValue<bool>());
    }

    [Fact]
    public async Task AnsweringBehindTheProxyKeepsTheServerToken()
    {
        await using var router = await TestRouter.StartAsync();
        router.On("POST", "/v1/agents/sessions/s1/responses", 202, new { id = "r1", session_id = "s1", status = "running" });
        using var client = new VisionAgentsClient(new VisionAgentsOptions { Url = router.Url, ApiKey = "key", ApiSecret = Fixtures.Secret, Authenticate = true });
        var dispatch = new Dispatch(new DispatchOptions { Client = client });

        await dispatch.AnswerAsync(new InboundMessage { ChannelId = "", SessionId = "s1", UserId = "ada", Text = "hi" },
            TestContext.Current.CancellationToken);

        var seen = router.Only("POST", "/v1/agents/sessions/s1/responses");
        Assert.Null(seen.Body!["command_id"]);
        Assert.Equal("ada", seen.Headers["X-Stream-User-Id"]);
        var claims = Fixtures.Claims(seen.Headers["Authorization"]);
        Assert.True(claims["server"]!.GetValue<bool>());
        Assert.Null(claims["user_id"]);
    }

    [Fact]
    public async Task AMessageASessionIsHoldingIsAnsweredThereRatherThanByANewAgent()
    {
        using var client = new VisionAgentsClient(new VisionAgentsOptions { Url = "http://x", CustomerId = "examples" });
        var dispatch = new Dispatch(new DispatchOptions { Client = client });

        var refused = await Assert.ThrowsAsync<InvalidOperationException>(() => dispatch.GetOrCreateAgentAsync(
            new InboundMessage { ChannelId = "ch1", SessionId = "s1" }, () => throw new InvalidOperationException("never built"),
            TestContext.Current.CancellationToken));
        await Assert.ThrowsAsync<InvalidOperationException>(() => dispatch.AnswerAsync(
            new InboundMessage { ChannelId = "ch1" }, TestContext.Current.CancellationToken));

        Assert.Contains("AnswerAsync", refused.Message);
    }

    [Fact]
    public async Task AWorkerReconnectingSaysWhatItIsStillHandling()
    {
        await using var router = await TestRouter.StartAsync();
        var opened = 0;
        var started = new TaskCompletionSource();
        var release = new TaskCompletionSource();
        JsonObject? done = null;
        router.OnSocket("/v1/dispatch", async peer =>
        {
            if (Interlocked.Increment(ref opened) == 1)
            {
                await peer.SendAsync(new { type = "call", work_id = "w1", call_id = "c1" });
                await started.Task;
                peer.Abort();
                return;
            }
            await peer.ReceiveAsync("ping");
            release.SetResult();
            done = await peer.ReceiveAsync("done");
        });
        using var client = Fixtures.Client(router);
        var dispatch = new Dispatch(new DispatchOptions { Client = client, ReportEvery = TimeSpan.FromMilliseconds(50) })
        {
            FirstRetry = TimeSpan.FromMilliseconds(10),
        };
        dispatch.WaitForCall(_ =>
        {
            started.SetResult();
            return release.Task;
        });

        await dispatch.RunAsync(TestContext.Current.CancellationToken).WaitAsync(TimeSpan.FromSeconds(10), TestContext.Current.CancellationToken);

        Assert.Equal(["0", "1"], router.To("GET", "/v1/dispatch").Select(seen => seen.Query["active"]));
        Assert.Equal("w1", done.Text("work_id"));
    }

    [Fact]
    public async Task TheSecondMessageOnAChannelGoesToTheAgentThatAnsweredTheFirst()
    {
        await using var router = await TestRouter.StartAsync();
        router.On("POST", "/v1/agents/sessions", 201, Fixtures.Session("s1", persisted: true));
        var closed = new TaskCompletionSource();
        router.OnSocket("/v1/agents/sessions/s1/events", async peer =>
        {
            await peer.ReceiveAsync("close");
            closed.SetResult();
        });
        var handled = new SemaphoreSlim(0);
        router.OnSocket("/v1/dispatch", async peer =>
        {
            await peer.SendAsync(new { type = "message", channel_type = "messaging", channel_id = "ch1", agent_id = "support-ch1", text = "hi" });
            await handled.WaitAsync();
            await peer.SendAsync(new { type = "message", channel_type = "messaging", channel_id = "ch1", agent_id = "support-ch1", text = "again" });
            await handled.WaitAsync();
        });
        using var client = Fixtures.Client(router);
        var dispatch = new Dispatch(new DispatchOptions { Client = client });
        var built = 0;
        var agents = new ConcurrentBag<Agent>();
        dispatch.WaitForMessage(async message =>
        {
            try
            {
                agents.Add(await dispatch.GetOrCreateAgentAsync(message, () =>
                {
                    built++;
                    return new Agent(new AgentOptions { Config = "support", Client = client });
                }));
            }
            finally
            {
                handled.Release();
            }
        });

        await dispatch.RunAsync(TestContext.Current.CancellationToken);
        await closed.Task.WaitAsync(TimeSpan.FromSeconds(5), TestContext.Current.CancellationToken);

        Assert.Equal(1, built);
        Assert.Single(agents.Distinct());
        var opened = router.Only("POST", "/v1/agents/sessions").Body;
        Assert.Equal(("support", "support-ch1"), (opened.Text("agent"), opened.Text("agent_id")));
        Assert.Null(opened!["incognito"]);
        Assert.True(opened["text"]!.GetValue<bool>());
    }

    [Fact]
    public async Task AWorkerReportsItsLoadAndTheRoundTrip()
    {
        await using var router = await TestRouter.StartAsync();
        JsonObject? load = null;
        router.OnSocket("/v1/dispatch", async peer =>
        {
            var ping = await peer.ReceiveAsync("ping");
            await peer.SendAsync(new { type = "pong", at = ping["at"]!.GetValue<double>() });
            load = await peer.ReceiveAsync("load");
        });
        using var client = Fixtures.Client(router);
        var dispatch = new Dispatch(new DispatchOptions { Client = client, ReportEvery = TimeSpan.FromMilliseconds(50) });
        dispatch.WaitForCall(_ => Task.CompletedTask);

        await dispatch.RunAsync(TestContext.Current.CancellationToken);

        Assert.Equal(0, load!["active_agents"]!.GetValue<int>());
        Assert.InRange(load["latency_ms"]!.GetValue<double>(), 0, 5000);
    }

    [Fact]
    public async Task CancellingStopsWaitingAndFinishesWhatIsRunning()
    {
        await using var router = await TestRouter.StartAsync();
        var finished = false;
        var started = new TaskCompletionSource();
        router.OnSocket("/v1/dispatch", async peer =>
        {
            await peer.SendAsync(new { type = "call", call_id = "c1" });
            await peer.Closed;
        });
        using var client = Fixtures.Client(router);
        var dispatch = new Dispatch(new DispatchOptions { Client = client });
        dispatch.WaitForCall(async _ =>
        {
            started.SetResult();
            await Task.Delay(200);
            finished = true;
        });
        using var stop = CancellationTokenSource.CreateLinkedTokenSource(TestContext.Current.CancellationToken);

        var running = dispatch.RunAsync(stop.Token);
        await started.Task.WaitAsync(TimeSpan.FromSeconds(5), TestContext.Current.CancellationToken);
        await stop.CancelAsync();
        await running.WaitAsync(TimeSpan.FromSeconds(5), TestContext.Current.CancellationToken);

        Assert.True(finished);
        Assert.Equal(0, dispatch.Active);
    }

    [Fact]
    public async Task CancellingBeforeTheSocketOpensStillStops()
    {
        await using var router = await TestRouter.StartAsync();
        using var client = Fixtures.Client(router);
        var dispatch = new Dispatch(new DispatchOptions { Client = client });
        dispatch.WaitForCall(_ => Task.CompletedTask);

        await dispatch.RunAsync(new CancellationToken(canceled: true)).WaitAsync(TimeSpan.FromSeconds(5), TestContext.Current.CancellationToken);

        Assert.Empty(router.Requests);
    }

    [Fact]
    public async Task AHostedToolIsDeclaredAndAnsweredOverTheDispatchSocket()
    {
        await using var router = await TestRouter.StartAsync();
        JsonObject? declared = null;
        var answered = new List<JsonObject>();
        router.OnSocket("/v1/dispatch", async peer =>
        {
            await peer.SendAsync(new { type = "ready", worker_id = "worker-1" });
            declared = await peer.ReceiveAsync("host_tools");
            await peer.SendAsync(new { type = "hosting", agent_id = "stream-support" });
            await peer.SendAsync(new { type = "tool_call", id = "call-1", session_id = "s", name = "investigate_sdk", arguments = """{"sdk":"android"}""" });
            answered.Add(await peer.ReceiveAsync("tool_result"));
            await peer.SendAsync(new { type = "tool_call", id = "call-2", session_id = "s", name = "delete_everything", arguments = "{}" });
            answered.Add(await peer.ReceiveAsync("tool_result"));
        });
        using var client = Fixtures.Client(router);
        var tools = new Tools().Register<Investigation, string>("investigate_sdk", "Read SDK source",
            (investigation, _) => Task.FromResult("read " + investigation.Sdk));
        var dispatch = new Dispatch(new DispatchOptions { Client = client }).Host("stream-support", tools, TimeSpan.FromMinutes(1));

        await dispatch.RunAsync(TestContext.Current.CancellationToken);

        Assert.Equal("", router.Only("GET", "/v1/dispatch").Query["handles"]);
        Assert.Equal(("stream-support", 60000), (declared.Text("agent_id"), declared!["timeout_ms"]!.GetValue<int>()));
        var tool = Assert.Single(declared["tools"]!.AsArray())!;
        Assert.Equal(("investigate_sdk", "Read SDK source"), (tool["name"]!.GetValue<string>(), tool["description"]!.GetValue<string>()));
        Assert.NotNull(tool["parameters"]!["properties"]!["sdk"]);
        Assert.Equal(("call-1", "read android"), (answered[0].Text("id"), answered[0].Text("output")));
        Assert.Equal(("call-2", "this worker does not run delete_everything"), (answered[1].Text("id"), answered[1].Text("error")));
        Assert.Equal(0, dispatch.Active);
    }

    [Fact]
    public async Task AWorkerTheRouterDropsReconnectsAndHostsAgain()
    {
        // The first connection is cut without a close frame, the way a router pod being
        // replaced ends it, and the second goes away with one. A worker that stopped at
        // either would leave every session naming the agent without its tools.
        await using var router = await TestRouter.StartAsync();
        var opened = 0;
        var hosted = new ConcurrentQueue<JsonObject>();
        var third = new TaskCompletionSource();
        router.OnSocket("/v1/dispatch", async peer =>
        {
            var connection = Interlocked.Increment(ref opened);
            await peer.SendAsync(new { type = "ready", worker_id = $"worker-{connection}" });
            hosted.Enqueue(await peer.ReceiveAsync("host_tools"));
            switch (connection)
            {
                case 1:
                    peer.Abort();
                    break;
                case 2:
                    await peer.CloseAsync(WebSocketCloseStatus.EndpointUnavailable);
                    break;
                default:
                    third.TrySetResult();
                    await peer.Closed;
                    break;
            }
        });
        using var client = Fixtures.Client(router);
        var tools = new Tools().Register<Investigation, string>("investigate_sdk", "Read SDK source", (_, _) => Task.FromResult(""));
        var dispatch = new Dispatch(new DispatchOptions { Client = client }) { FirstRetry = TimeSpan.FromMilliseconds(10) }
            .Host("stream-support", tools);
        using var stop = CancellationTokenSource.CreateLinkedTokenSource(TestContext.Current.CancellationToken);

        var running = dispatch.RunAsync(stop.Token);
        await third.Task.WaitAsync(TimeSpan.FromSeconds(5), TestContext.Current.CancellationToken);
        await stop.CancelAsync();
        await running.WaitAsync(TimeSpan.FromSeconds(5), TestContext.Current.CancellationToken);

        Assert.Equal("worker-3", dispatch.WorkerId);
        Assert.Equal(["stream-support", "stream-support", "stream-support"], hosted.Select(frame => frame.Text("agent_id")));
        Assert.Equal(0, hosted.First()["timeout_ms"]!.GetValue<int>());
    }

    [Fact]
    public async Task AWorkerWhoseToolsAreRefusedStopsWaiting()
    {
        await using var router = await TestRouter.StartAsync();
        router.OnSocket("/v1/dispatch", async peer =>
        {
            await peer.SendAsync(new { type = "ready", worker_id = "worker-1" });
            await peer.ReceiveAsync("host_tools");
            await peer.SendAsync(new { type = "hosting_refused", agent_id = "stream-support", reason = "hosting no tools is not hosting" });
            await peer.Closed;
        });
        using var client = Fixtures.Client(router);
        var dispatch = new Dispatch(new DispatchOptions { Client = client }).Host("stream-support", new Tools());

        var refused = await Assert.ThrowsAsync<InvalidOperationException>(
            () => dispatch.RunAsync(TestContext.Current.CancellationToken).WaitAsync(TimeSpan.FromSeconds(5), TestContext.Current.CancellationToken));

        Assert.Contains("stream-support", refused.Message);
        Assert.Contains("hosting no tools is not hosting", refused.Message);
    }

    [Fact]
    public async Task AWorkerThatCanHoldNothingOrDoesNothingIsRefused()
    {
        using var client = new VisionAgentsClient(new VisionAgentsOptions { Url = "http://x", CustomerId = "examples" });

        Assert.Throws<ConfigurationException>(() => new Dispatch(new DispatchOptions { Client = client, Capacity = 0 }));
        await Assert.ThrowsAsync<InvalidOperationException>(() => new Dispatch(new DispatchOptions { Client = client }).RunAsync(TestContext.Current.CancellationToken));
    }

    private sealed record Investigation(string Sdk);
}
