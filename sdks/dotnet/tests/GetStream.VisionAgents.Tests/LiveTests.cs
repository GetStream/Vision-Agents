using System.Text.Json;
using GetStream.VisionAgents.Models;

namespace GetStream.VisionAgents.Tests;

/// <summary>
/// The SDK against a real router, run only when <c>VISION_AGENTS_LIVE_URL</c> names one.
/// </summary>
/// <remarks>
/// A local router with auth off takes a customer id at face value, which is what makes this
/// runnable without secrets. What it proves is what the test server cannot: that the requests
/// built here are ones a real router accepts.
/// </remarks>
public sealed class LiveTests : IAsyncLifetime
{
    private static readonly string? Url = Environment.GetEnvironmentVariable("VISION_AGENTS_LIVE_URL");
    private static readonly string Customer = Environment.GetEnvironmentVariable("VISION_AGENTS_LIVE_CUSTOMER") ?? "examples";

    private readonly List<string> _opened = [];
    private VisionAgentsClient _client = null!;
    private string _model = "";

    public async ValueTask InitializeAsync()
    {
        Assert.SkipWhen(Url is null, "VISION_AGENTS_LIVE_URL names no router");
        _client = new VisionAgentsClient(new VisionAgentsOptions { Url = Url, CustomerId = Customer });
        _model = await ConversationModelAsync();
    }

    public async ValueTask DisposeAsync()
    {
        if (Url is null)
        {
            return;
        }
        foreach (var id in _opened)
        {
            try
            {
                await _client.DeleteAsync($"/v1/agents/sessions/{id}");
            }
            catch (RouterException)
            {
            }
        }
        _client.Dispose();
    }

    [Fact]
    public async Task IsHealthyAndSaysWhatItDependsOn()
    {
        var health = await _client.GetAsync<JsonElement>("/health", cancellationToken: Cancel);

        Assert.Equal("ok", health.GetProperty("dependencies").GetProperty("llm").GetString());
        Assert.True(health.GetProperty("status").GetString() == "ok", $"the router is degraded: {health}");
    }

    [Fact]
    public async Task NamesTheTurnItAnswersAndWritesDownWhatItWasMadeOf()
    {
        await using var agent = Agent();
        var session = Remember(await agent.ChatAsync(cancellationToken: Cancel));

        var answering = await agent.Responses.CreateAsync("Reply with the single word: pong.", cancellationToken: Cancel);
        Assert.NotEqual("", answering.Id);
        await AnsweredAsync(session);

        var items = await EventuallyAsync(() => answering.Items.AllAsync(Cancel), items => items.Exists(item => item.Kind == "answer"));

        Assert.Equal("said", items[0].Kind);
        Assert.Contains(items, item => item.Kind == "answer");
    }

    [Fact]
    public async Task RewindsToATurnAndForksFromIt()
    {
        await using var agent = Agent();
        var session = Remember(await agent.ChatAsync(cancellationToken: Cancel));
        foreach (var question in new[] { "Reply with the single word: one.", "Reply with the single word: two." })
        {
            await session.Responses.CreateAsync(question, cancellationToken: Cancel);
            await AnsweredAsync(session);
        }

        var kept = (await session.Responses.ListAsync(cancellationToken: Cancel))[0];
        await session.Responses.RewindAsync(kept.Id, Cancel);

        Assert.Equal([kept.Id], (await session.Responses.ListAsync(cancellationToken: Cancel)).Select(response => response.Id));
        var forked = Remember(await session.ForkAsync(new ForkOptions { ResponseId = kept.Id }, Cancel));
        Assert.Equal(session.Id, forked.Created.ForkedFrom);
    }

    [Fact]
    public async Task RefusesToRewindAConversationKeptInStreamChat()
    {
        await using var agent = Agent();
        var session = Remember(await agent.ChatAsync(new SessionOptions { Persist = true }, Cancel));

        var refused = await Assert.ThrowsAsync<RouterException>(() => session.Responses.RewindAsync("anything", Cancel));

        Assert.Equal(400, refused.Status);
        Assert.Contains("fork it at the response", refused.Message);
    }

    [Fact]
    public async Task FindsAConversationAgainByItsTitle()
    {
        await using var agent = Agent();
        var title = $"dotnet {Guid.NewGuid():N}";
        var session = Remember(await agent.ChatAsync(new SessionOptions { Persist = true, Title = title }, Cancel));
        await session.CloseAsync(Cancel);

        var found = await EventuallyAsync(() => _client.Sessions.SearchAsync(title, new SessionQuery { Limit = 50 }, Cancel),
            found => found.Exists(each => each.Id == session.Id));

        Assert.Contains(found, each => each.Id == session.Id);
    }

    [Fact]
    public async Task SyncsAFolderAndOnlyReadsItBackWhenNothingChanged()
    {
        var root = Directory.CreateTempSubdirectory("dotnet-live-").FullName;
        var name = $"dotnet-live-{Guid.NewGuid():N}"[..24];
        var folder = Path.Combine(root, name);
        Directory.CreateDirectory(folder);
        await File.WriteAllTextAsync(Path.Combine(folder, Folder.AgentFile), $"llm: {_model}\n", Cancel);
        await File.WriteAllTextAsync(Path.Combine(folder, Folder.InstructionsFile), "Answer in one word.\n", Cancel);
        await using var agent = new Agent(new AgentOptions { Config = folder, Client = _client });
        try
        {
            var synced = await agent.SyncAsync(Cancel);
            var again = await agent.SyncAsync(Cancel);

            Assert.Equal(name, synced.Name);
            Assert.Equal(synced.Id, again.Id);
            Assert.True(File.Exists(Path.Combine(folder, Folder.StampFile)));
            await _client.DeleteAsync($"/v1/agents/configs/{synced.Id}", Cancel);
        }
        finally
        {
            Directory.Delete(root, recursive: true);
        }
    }

    [Fact]
    public async Task AnswersOverTheCompletionsSocket()
    {
        await using var llm = await _client.Router().Llm.RealtimeAsync(new LlmOptions { Target = _model }, Cancel);

        await llm.AskAsync(new Question { Messages = [new Said("user", "Reply with the single word: pong.")] }, Cancel);

        await foreach (var answer in llm.ReadAllAsync(Cancel))
        {
            Assert.SkipWhen(answer.Error.Contains("quota"), $"nothing left to answer with: {answer.Error}");
            Assert.Equal("", answer.Error);
            if (answer.Done)
            {
                Assert.NotEqual("", answer.Text);
                return;
            }
        }
        Assert.Fail("the socket closed without an answer");
    }

    private static CancellationToken Cancel => TestContext.Current.CancellationToken;

    private Agent Agent() => new(new AgentOptions
    {
        Name = "dotnet-live",
        Instructions = "Answer in one word.",
        Pipeline = new Pipeline { Llm = _model },
        Client = _client,
    });

    private Session Remember(Session session)
    {
        _opened.Add(session.Id);
        return session;
    }

    /// <summary>Waits out one turn, stepping around a deployment that has spent its allowance.</summary>
    private static async Task AnsweredAsync(Session session)
    {
        await foreach (var happened in session.EventsAsync(Cancel))
        {
            if (happened.Kind == "responded")
            {
                return;
            }
            if (happened.Kind == "error")
            {
                Assert.SkipWhen(happened.Error.Contains("quota"), $"nothing left to answer with: {happened.Error}");
                Assert.Fail($"the session reported: {happened.Error}");
            }
        }
        Assert.Fail("the session ended before answering");
    }

    /// <summary>
    /// Reads until what the router writes behind a request has arrived, since a read taken
    /// the instant the request returns legitimately finds nothing yet.
    /// </summary>
    private static async Task<T> EventuallyAsync<T>(Func<Task<T>> read, Func<T, bool> holds)
    {
        var deadline = DateTime.UtcNow.AddSeconds(20);
        var last = await read();
        while (!holds(last) && DateTime.UtcNow < deadline)
        {
            await Task.Delay(250, Cancel);
            last = await read();
        }
        return last;
    }

    /// <summary>A conversation model this deployment resolves, since its catalogue is its own.</summary>
    private async Task<string> ConversationModelAsync()
    {
        foreach (var target in new[] { "llm-fast", "llm-thinking", "llm-flow" })
        {
            try
            {
                await _client.GetAsync<JsonElement>($"/v1/llm/routes/{target}");
                return target;
            }
            catch (RouterException missing) when (missing.Status == 404)
            {
            }
        }
        throw new InvalidOperationException("this deployment resolves none of the conversation models");
    }
}
