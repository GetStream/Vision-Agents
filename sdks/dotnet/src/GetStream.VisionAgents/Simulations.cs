using GetStream.VisionAgents.Models;

namespace GetStream.VisionAgents;

/// <summary>
/// Conversations to put an agent through, and something that has to be true at the end of each.
/// </summary>
/// <remarks>
/// Server side only: what an agent is tested on is the app's business.
/// <code>
/// var simulation = await api.Simulations.CreateAsync(new SimulationRequest { Name = name, ConfigId = configId, Scenario = scenario, Assertion = assertion });
/// var run = await api.Simulations.RunAsync(simulation.Id);
/// run = await api.Simulations.Runs.GetAsync(run.Id);
/// </code>
/// </remarks>
public sealed class Simulations
{
    private const string Path = "/v1/agents/simulations";

    private readonly VisionAgentsClient _client;

    internal Simulations(VisionAgentsClient client)
    {
        _client = client;
        Runs = new SimulationRuns(client);
    }

    /// <summary>What the simulations have come to.</summary>
    public SimulationRuns Runs { get; }

    /// <summary>Writes a simulation down. Nothing is run until <see cref="RunAsync"/> is called.</summary>
    public Task<Simulation> CreateAsync(SimulationRequest simulation, CancellationToken cancellationToken = default) =>
        _client.PostAsync<Simulation>(Path, simulation, cancellationToken);

    /// <summary>One simulation.</summary>
    public Task<Simulation> GetAsync(string id, CancellationToken cancellationToken = default) =>
        _client.GetAsync<Simulation>($"{Path}/{VisionAgentsClient.Escape(id)}", cancellationToken: cancellationToken);

    /// <summary>The customer's simulations, newest first.</summary>
    public Task<List<Simulation>> ListAsync(CancellationToken cancellationToken = default) =>
        _client.GetAsync<List<Simulation>>(Path, cancellationToken: cancellationToken);

    /// <summary>
    /// Replaces a simulation: every field is written, so pass what it now asks rather than what
    /// changed. The runs it already has keep their own copy of what they tested.
    /// </summary>
    public Task<Simulation> UpdateAsync(string id, SimulationRequest simulation, CancellationToken cancellationToken = default) =>
        _client.PutAsync<Simulation>($"{Path}/{VisionAgentsClient.Escape(id)}", simulation, cancellationToken);

    /// <summary>Deletes a simulation. The runs that named it are kept.</summary>
    public Task DeleteAsync(string id, CancellationToken cancellationToken = default) =>
        _client.DeleteAsync($"{Path}/{VisionAgentsClient.Escape(id)}", cancellationToken);

    /// <summary>
    /// Starts a run. It answers once the run is written rather than once it is over, so read
    /// it again with <see cref="SimulationRuns.GetAsync"/> until its state is no longer <c>running</c>.
    /// </summary>
    public Task<SimulationRun> RunAsync(string id, CancellationToken cancellationToken = default) =>
        _client.PostAsync<SimulationRun>($"{Path}/{VisionAgentsClient.Escape(id)}/run", null, cancellationToken);
}

/// <summary>Which runs to list. Every field narrows.</summary>
public sealed record SimulationRunQuery
{
    /// <summary>Only the runs of this simulation; null, every simulation's.</summary>
    public string? SimulationId { get; init; }

    /// <summary>Only the runs in this state: running, passed, failed, cancelled or errored.</summary>
    public string? State { get; init; }

    /// <summary>How many to return.</summary>
    public int? Limit { get; init; }
}

/// <summary>The runs of every simulation, newest first.</summary>
public sealed class SimulationRuns
{
    private const string Path = "/v1/agents/simulation-runs";

    private readonly VisionAgentsClient _client;

    internal SimulationRuns(VisionAgentsClient client) => _client = client;

    /// <summary>One run, with the conversations it had.</summary>
    public Task<SimulationRun> GetAsync(string id, CancellationToken cancellationToken = default) =>
        _client.GetAsync<SimulationRun>($"{Path}/{VisionAgentsClient.Escape(id)}", cancellationToken: cancellationToken);

    /// <summary>What the simulations have come to, newest first.</summary>
    public Task<List<SimulationRun>> ListAsync(SimulationRunQuery? query = null, CancellationToken cancellationToken = default) =>
        _client.GetAsync<List<SimulationRun>>(Path, new Dictionary<string, string?>
        {
            ["simulation_id"] = VisionAgentsClient.Blank(query?.SimulationId),
            ["state"] = VisionAgentsClient.Blank(query?.State),
            ["limit"] = VisionAgentsClient.Number(query?.Limit),
        }, cancellationToken);

    /// <summary>Stops a run in progress, ending the conversations still in flight.</summary>
    public Task<SimulationRun> CancelAsync(string id, CancellationToken cancellationToken = default) =>
        _client.PostAsync<SimulationRun>($"{Path}/{VisionAgentsClient.Escape(id)}/cancel", null, cancellationToken);
}
