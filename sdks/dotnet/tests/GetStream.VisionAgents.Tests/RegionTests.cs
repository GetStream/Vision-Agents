using System.Net;
using System.Text;
using GetStream.VisionAgents.Models;

namespace GetStream.VisionAgents.Tests;

/// <summary>
/// Changes the process's environment, which every client reads, so nothing else runs alongside it.
/// </summary>
[CollectionDefinition(nameof(ProcessEnvironment), DisableParallelization = true)]
public sealed class ProcessEnvironment;

[Collection(nameof(ProcessEnvironment))]
public sealed class RegionTests : IDisposable
{
    private const string RegionalHost = "accelerate-us-east4.gcp.stream-io-api.com";

    private static readonly Dictionary<string, string?> Set = new()
    {
        [Backend.UrlEnv] = null,
        [Backend.RegionEnv] = null,
        [Backend.CustomerEnv] = null,
        [Backend.AuthenticateEnv] = null,
        [Backend.ApiKeyEnv] = "test-key",
        [Backend.ApiSecretEnv] = "test-secret",
    };

    private readonly Dictionary<string, string?> _before = [];
    private readonly Recorder _recorder = new();
    private readonly HttpClient _http;

    public RegionTests()
    {
        foreach (var (name, value) in Set)
        {
            _before[name] = Environment.GetEnvironmentVariable(name);
            Environment.SetEnvironmentVariable(name, value);
        }
        _http = new HttpClient(_recorder);
    }

    public void Dispose()
    {
        _http.Dispose();
        foreach (var (name, value) in _before)
        {
            Environment.SetEnvironmentVariable(name, value);
        }
    }

    [Fact]
    public async Task TheDefaultClientReadsFromTheRegionalHost()
    {
        using var client = new VisionAgentsClient(_http);

        await ReadConfigs(client);
    }

    [Fact]
    public async Task AnExplicitRegionReadsFromTheRegionalHost()
    {
        using var client = new VisionAgentsClient(_http, new VisionAgentsOptions { Region = "us-east4" });

        await ReadConfigs(client);
    }

    [Fact]
    public async Task ARegionFromTheEnvironmentReadsFromTheRegionalHost()
    {
        Environment.SetEnvironmentVariable(Backend.RegionEnv, "us-east4");
        using var client = new VisionAgentsClient(_http);

        await ReadConfigs(client);
    }

    [Fact]
    public void AConflictingUrlAndRegionAreRefused()
    {
        Assert.Throws<ConfigurationException>(() =>
            new VisionAgentsClient(_http, new VisionAgentsOptions { Region = "europe-west4", Url = Backend.DefaultUrl }));
    }

    [Fact]
    public void AnInvalidRegionWithAUrlIsRefused()
    {
        Assert.Throws<ConfigurationException>(() =>
            new VisionAgentsClient(_http, new VisionAgentsOptions { Region = "us-east4.evil.com/x", Url = Backend.DefaultUrl }));
    }

    [Fact]
    public async Task ARegionalClientCanActForAUser()
    {
        using var client = new VisionAgentsClient(_http, new VisionAgentsOptions { Region = "us-east4" });

        var user = client.AsUser("jean", "user-token");

        Assert.Equal(Backend.DefaultUrl, user.Url);
        Assert.Equal("jean", user.Backend.UserId);
        Assert.Equal("user-token", user.Backend.Token);
        await ReadConfigs(user);
        Assert.Equal("Bearer user-token", _recorder.Seen.Single().Headers.GetValues("Authorization").Single());
    }

    private async Task ReadConfigs(VisionAgentsClient client)
    {
        await client.GetAsync<List<AgentConfig>>("/v1/agents/configs", cancellationToken: TestContext.Current.CancellationToken);

        var seen = Assert.Single(_recorder.Seen);
        Assert.Equal(HttpMethod.Get, seen.Method);
        Assert.Equal(RegionalHost, seen.RequestUri!.Host);
        Assert.Equal("/v1/agents/configs", seen.RequestUri.AbsolutePath);
        Assert.Equal("test-key", seen.Headers.GetValues("api_key").Single());
        Assert.Equal("jwt", seen.Headers.GetValues("stream-auth-type").Single());
        Assert.NotEmpty(seen.Headers.GetValues("Authorization").Single());
    }

    /// <summary>Answers every request itself, so the regional URL is kept but never dialled.</summary>
    private sealed class Recorder : HttpMessageHandler
    {
        public List<HttpRequestMessage> Seen { get; } = [];

        protected override Task<HttpResponseMessage> SendAsync(HttpRequestMessage request, CancellationToken cancellationToken)
        {
            Seen.Add(request);
            return Task.FromResult(new HttpResponseMessage(HttpStatusCode.OK)
            {
                Content = new StringContent("[]", Encoding.UTF8, "application/json"),
            });
        }
    }
}
