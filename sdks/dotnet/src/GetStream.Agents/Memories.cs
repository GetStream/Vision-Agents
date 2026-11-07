namespace GetStream.Agents;

/// <summary>What agents remember about the app's users between conversations.</summary>
public sealed class Memories
{
    private readonly StreamAgents _client;

    internal Memories(StreamAgents client) => _client = client;

    /// <summary>
    /// Deletes everything remembered about one user: every session's and every agent's,
    /// whatever memory filter it was written under. Only a backend may ask.
    /// </summary>
    /// <param name="userId">The <c>user_id</c> of the memory filter the sessions were opened with.</param>
    /// <param name="cancellationToken">Stops waiting.</param>
    public Task TruncateAsync(string userId, CancellationToken cancellationToken = default) =>
        string.IsNullOrEmpty(userId)
            ? throw new ConfigurationException("truncating memories needs a user id")
            : _client.DeleteAsync($"/v1/agents/users/{StreamAgents.Escape(userId)}/memories", cancellationToken);
}
