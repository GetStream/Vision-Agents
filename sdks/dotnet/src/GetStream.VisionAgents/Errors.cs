namespace GetStream.VisionAgents;

/// <summary>
/// What the router said it would not do, raised where it was asked.
/// </summary>
/// <remarks>
/// A request that never arrived is <see cref="Status"/> 0, because a caller retrying a
/// network failure and one retrying a 500 are doing different things.
/// </remarks>
public class RouterException(int status, string operation, string message, Exception? inner = null)
    : Exception($"{operation}: {message}", inner)
{
    /// <summary>The HTTP status, or 0 when nothing was answered.</summary>
    public int Status { get; } = status;

    /// <summary>What was being asked, as the method and path.</summary>
    public string Operation { get; } = operation;

    /// <summary>What the router said, without the operation in front of it.</summary>
    public string Said { get; } = message;

    /// <summary>
    /// The seconds until a daily limit resets, from <c>Retry-After</c>, when the router
    /// answered 429.
    /// </summary>
    public TimeSpan? RetryAfter { get; init; }
}

/// <summary>
/// Something about how the SDK was set up that cannot work, reported before any request.
/// </summary>
public class ConfigurationException(string message) : Exception(message);

/// <summary>A send on a socket that is no longer open.</summary>
public class SocketClosedException(string message) : Exception(message);
