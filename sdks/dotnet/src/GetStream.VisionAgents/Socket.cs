using System.Buffers;
using System.Net.WebSockets;
using System.Text.Json;
using System.Text.Json.Nodes;

namespace GetStream.VisionAgents;

/// <summary>
/// One JSON message on a socket. Every frame carries a <c>type</c> and that event's fields.
/// </summary>
/// <remarks>
/// Loosely typed on purpose: an event a router learnt after this SDK shipped still reaches
/// a caller reading <see cref="Type"/>, whole, rather than being dropped here.
/// </remarks>
public sealed class Frame(JsonObject json)
{
    /// <summary>The whole frame as it arrived.</summary>
    public JsonObject Json { get; } = json;

    /// <summary>The frame's kind, or empty if it has none.</summary>
    public string Type => Text("type");

    /// <summary>A string field, or empty if it is absent or another type.</summary>
    public string Text(string name) =>
        Json[name] is JsonValue value && value.TryGetValue(out string? text) ? text : "";

    /// <summary>A boolean field, false if it is absent or another type.</summary>
    public bool Flag(string name) =>
        Json[name] is JsonValue value && value.TryGetValue(out bool flag) && flag;

    /// <summary>A numeric field, zero if it is absent or another type.</summary>
    public double Number(string name) =>
        Json[name] is JsonValue value && value.TryGetValue(out double number) ? number : 0;

    /// <summary>A nested object, or null if it is absent or another type.</summary>
    public Frame? Nested(string name) => Json[name] is JsonObject nested ? new Frame(nested) : null;

    /// <inheritdoc />
    public override string ToString() => Json.ToJsonString();
}

/// <summary>
/// One WebSocket to the router: JSON one way, JSON and audio the other.
/// </summary>
/// <remarks>
/// OpenAPI stops at the upgrade, so this is written by hand. One reader at a time, which is
/// what <see cref="ClientWebSocket"/> allows; sends are serialised, because tool results are
/// answered from a task per call. There is no reconnection: <c>respond</c> and
/// <c>tool_result</c> are not idempotent and the protocol has nothing to resume from.
/// </remarks>
internal sealed class Socket : IAsyncDisposable
{
    // A socket gone wrong at the far end would otherwise buffer without bound.
    private const int MaxMessage = 16 << 20;

    private readonly ClientWebSocket _socket;
    private readonly SemaphoreSlim _sending = new(1, 1);
    private int _closed;

    private Socket(ClientWebSocket socket) => _socket = socket;

    public bool Open => _socket.State == WebSocketState.Open && Volatile.Read(ref _closed) == 0;

    /// <summary>Dials the router, reporting what it said if it refused the upgrade.</summary>
    public static async Task<Socket> ConnectAsync(Backend backend, Uri url, CancellationToken cancellationToken)
    {
        var socket = new ClientWebSocket();
        socket.Options.CollectHttpResponseDetails = true;
        foreach (var (name, value) in backend.Headers())
        {
            socket.Options.SetRequestHeader(name, value);
        }
        try
        {
            await socket.ConnectAsync(url, cancellationToken).ConfigureAwait(false);
        }
        catch (WebSocketException failure)
        {
            var status = (int)socket.HttpStatusCode;
            socket.Dispose();
            throw new RouterException(status, $"GET {url.AbsolutePath}",
                status == 0 ? failure.Message : $"the router refused the socket with {status}", failure);
        }
        return new Socket(socket);
    }

    /// <summary>Writes one JSON frame.</summary>
    public Task SendAsync(object frame, CancellationToken cancellationToken = default) =>
        WriteAsync(JsonSerializer.SerializeToUtf8Bytes(frame, frame.GetType(), Json.Options),
            WebSocketMessageType.Text, cancellationToken);

    /// <summary>Writes one binary frame.</summary>
    public Task SendAudioAsync(ReadOnlyMemory<byte> payload, CancellationToken cancellationToken = default) =>
        WriteAsync(payload, WebSocketMessageType.Binary, cancellationToken);

    /// <summary>
    /// The next message: a frame, or bytes for a binary one. Null once the socket closes.
    /// </summary>
    /// <remarks>
    /// A text frame that is not JSON is skipped rather than ending the stream, which would
    /// lose everything said after it. A read cancelled by <see cref="CloseAsync"/> is a
    /// close that was asked for, so it ends the stream rather than throwing.
    /// </remarks>
    public async Task<(Frame? Frame, byte[]? Audio)?> ReceiveAsync(CancellationToken cancellationToken = default)
    {
        var buffer = ArrayPool<byte>.Shared.Rent(64 << 10);
        try
        {
            while (true)
            {
                using var message = new MemoryStream();
                WebSocketReceiveResult received;
                do
                {
                    try
                    {
                        received = await _socket.ReceiveAsync(buffer, cancellationToken).ConfigureAwait(false);
                    }
                    catch (WebSocketException) when (!Open)
                    {
                        return null;
                    }
                    catch (OperationCanceledException) when (!cancellationToken.IsCancellationRequested)
                    {
                        return null;
                    }
                    catch (ObjectDisposedException)
                    {
                        return null;
                    }
                    catch (WebSocketException)
                    {
                        // The far end went away without a close handshake: the same thing,
                        // from here, as a close.
                        return null;
                    }

                    if (received.MessageType == WebSocketMessageType.Close)
                    {
                        await CloseAsync().ConfigureAwait(false);
                        return null;
                    }
                    if (message.Length + received.Count > MaxMessage)
                    {
                        await CloseAsync(WebSocketCloseStatus.MessageTooBig).ConfigureAwait(false);
                        return null;
                    }
                    message.Write(buffer, 0, received.Count);
                }
                while (!received.EndOfMessage);

                if (received.MessageType == WebSocketMessageType.Binary)
                {
                    return (null, message.ToArray());
                }
                try
                {
                    if (JsonNode.Parse(message.ToArray()) is JsonObject json)
                    {
                        return (new Frame(json), null);
                    }
                }
                catch (JsonException)
                {
                    // Skipped, as the remarks say.
                }
            }
        }
        finally
        {
            ArrayPool<byte>.Shared.Return(buffer);
        }
    }

    /// <summary>Closes the socket. Safe to call twice.</summary>
    public async Task CloseAsync(WebSocketCloseStatus status = WebSocketCloseStatus.NormalClosure)
    {
        if (Interlocked.Exchange(ref _closed, 1) == 1)
        {
            return;
        }
        if (_socket.State is WebSocketState.Open or WebSocketState.CloseReceived)
        {
            using var bounded = new CancellationTokenSource(TimeSpan.FromSeconds(2));
            try
            {
                await _socket.CloseOutputAsync(status, "", bounded.Token).ConfigureAwait(false);
            }
            catch (Exception failure) when (failure is WebSocketException or OperationCanceledException or ObjectDisposedException)
            {
                // Already going; there is nobody left to say goodbye to.
            }
        }
        _socket.Abort();
    }

    public async ValueTask DisposeAsync()
    {
        await CloseAsync().ConfigureAwait(false);
        _socket.Dispose();
        _sending.Dispose();
    }

    private async Task WriteAsync(ReadOnlyMemory<byte> payload, WebSocketMessageType type, CancellationToken cancellationToken)
    {
        if (!Open)
        {
            throw new SocketClosedException("the socket is not open");
        }
        await _sending.WaitAsync(cancellationToken).ConfigureAwait(false);
        try
        {
            await _socket.SendAsync(payload, type, endOfMessage: true, cancellationToken).ConfigureAwait(false);
        }
        catch (Exception failure) when (failure is WebSocketException or ObjectDisposedException)
        {
            throw new SocketClosedException($"the socket closed while sending: {failure.Message}");
        }
        finally
        {
            _sending.Release();
        }
    }
}

internal static class Frames
{
    /// <summary>A frame the SDK sends, written with the router's own key spelling.</summary>
    public static JsonObject Of(string type, params (string Key, object? Value)[] fields)
    {
        var frame = new JsonObject { ["type"] = type };
        foreach (var (key, value) in fields)
        {
            if (value is not null)
            {
                frame[key] = JsonSerializer.SerializeToNode(value, value.GetType(), Json.Options);
            }
        }
        return frame;
    }
}
