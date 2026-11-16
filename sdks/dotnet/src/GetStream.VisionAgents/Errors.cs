using System.Net.Http.Headers;
using System.Text.Json;
using GetStream.VisionAgents.Models;

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

    /// <summary>
    /// The kind of failure, which decides the status: <c>invalid_request</c>,
    /// <c>not_found</c>, <c>rate_limited</c>, <c>internal</c> and the rest of
    /// <see cref="ErrorDetail.Type"/>. Null when the body was not the router's.
    /// </summary>
    public string? Type { get; init; }

    /// <summary>
    /// What went wrong, for a program to branch on: <c>not_configured</c>,
    /// <c>validation_failed</c>, <c>session_not_found</c> and more as the router learns them,
    /// so expect one this SDK does not know. Null when the body was not the router's.
    /// </summary>
    public string? Code { get; init; }

    /// <summary>Where <see cref="Code"/> is explained.</summary>
    public Uri? DocUrl { get; init; }

    /// <summary>
    /// The response's <c>X-Request-Id</c>, which is what to quote to support: a 500 says only
    /// "something went wrong", and this is how the rest of it is found.
    /// </summary>
    public string? RequestId { get; init; }

    /// <summary>
    /// The failure a response that was not a success reports.
    /// </summary>
    /// <remarks>
    /// The router answers every failure with an <see cref="ErrorResponse"/>, but a 502 from
    /// something in front of it, an empty body or an older router's <c>{"error": "..."}</c> is
    /// not one, so those keep their text as <see cref="Said"/> rather than fail to parse: a
    /// JSON error in place of the router's would hide what went wrong. The request id is read
    /// either way, since a proxy may still pass it on.
    /// </remarks>
    internal static RouterException Answered(
        int status,
        string operation,
        string body,
        IEnumerable<KeyValuePair<string, IEnumerable<string>>>? headers,
        Exception? inner = null)
    {
        string? requestId = null;
        TimeSpan? retryAfter = null;
        foreach (var (name, values) in headers ?? [])
        {
            if (name.Equals("X-Request-Id", StringComparison.OrdinalIgnoreCase))
            {
                requestId = VisionAgentsClient.Blank(values.FirstOrDefault());
            }
            else if (name.Equals("Retry-After", StringComparison.OrdinalIgnoreCase)
                && RetryConditionHeaderValue.TryParse(values.FirstOrDefault(), out var retry))
            {
                retryAfter = retry.Delta;
            }
        }

        ErrorDetail? detail = null;
        try
        {
            detail = JsonSerializer.Deserialize<ErrorResponse>(body, Json.Options)?.Error;
        }
        catch (JsonException)
        {
            // Not the envelope, so what arrived is reported as it is below.
        }
        if (detail is { Message.Length: > 0 })
        {
            return new RouterException(status, operation, detail.Message, inner)
            {
                Type = VisionAgentsClient.Blank(detail.Type),
                Code = VisionAgentsClient.Blank(detail.Code),
                DocUrl = detail.DocUrl,
                RequestId = requestId,
                RetryAfter = retryAfter,
            };
        }
        return new RouterException(status, operation, body.Trim() is { Length: > 0 } raw ? raw : $"the router answered {status}", inner)
        {
            RequestId = requestId,
            RetryAfter = retryAfter,
        };
    }
}

/// <summary>
/// Something about how the SDK was set up that cannot work, reported before any request.
/// </summary>
public class ConfigurationException(string message) : Exception(message);

/// <summary>A send on a socket that is no longer open.</summary>
public class SocketClosedException(string message) : Exception(message);
