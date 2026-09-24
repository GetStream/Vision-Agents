using System.Collections.Concurrent;
using System.Net;
using System.Net.WebSockets;
using System.Text;
using System.Text.Json;
using System.Text.Json.Nodes;
using System.Threading.Channels;
using Microsoft.AspNetCore.Builder;
using Microsoft.AspNetCore.Hosting;
using Microsoft.AspNetCore.Hosting.Server;
using Microsoft.AspNetCore.Hosting.Server.Features;
using Microsoft.AspNetCore.Http;
using Microsoft.Extensions.DependencyInjection;
using Microsoft.Extensions.Logging;

namespace GetStream.VisionAgents.Tests;

/// <summary>A request the server received, whole.</summary>
public sealed record Seen(string Method, string Path, IReadOnlyDictionary<string, string> Query, IReadOnlyDictionary<string, string> Headers, JsonNode? Body);

/// <summary>What to answer with.</summary>
public sealed record Reply(int Status, object? Body = null);

/// <summary>
/// A real HTTP and WebSocket server on a loopback port, answering the way the router's
/// contract says it does. What the SDK sent is recorded for the test to read back.
/// </summary>
public sealed class TestRouter : IAsyncDisposable
{
    private readonly WebApplication _app;
    private readonly ConcurrentDictionary<string, Func<Seen, Reply>> _routes = new();
    private readonly ConcurrentDictionary<string, Func<Peer, Task>> _sockets = new();
    private readonly ConcurrentQueue<Seen> _seen = new();

    private TestRouter(WebApplication app) => _app = app;

    /// <summary>The server's base URL.</summary>
    public string Url { get; private set; } = "";

    /// <summary>Every request, in the order it arrived.</summary>
    public IReadOnlyList<Seen> Requests => [.. _seen];

    public static async Task<TestRouter> StartAsync()
    {
        var builder = WebApplication.CreateSlimBuilder();
        builder.Logging.ClearProviders();
        builder.WebHost.UseKestrel(kestrel => kestrel.Listen(IPAddress.Loopback, 0));
        var app = builder.Build();
        var router = new TestRouter(app);
        app.UseWebSockets();
        app.Run(router.HandleAsync);
        await app.StartAsync();
        router.Url = app.Services.GetRequiredService<IServer>().Features.Get<IServerAddressesFeature>()!.Addresses.First();
        return router;
    }

    /// <summary>Answers a method and path.</summary>
    public TestRouter On(string method, string path, Func<Seen, Reply> answer)
    {
        _routes[$"{method} {path}"] = answer;
        return this;
    }

    /// <summary>Answers a method and path, always the same way.</summary>
    public TestRouter On(string method, string path, int status, object? body = null) =>
        On(method, path, _ => new Reply(status, body));

    /// <summary>Accepts a socket on a path and hands it to the test.</summary>
    public TestRouter OnSocket(string path, Func<Peer, Task> peer)
    {
        _sockets[path] = peer;
        return this;
    }

    /// <summary>The requests to a method and path.</summary>
    public List<Seen> To(string method, string path) =>
        [.. _seen.Where(seen => seen.Method == method && seen.Path == path)];

    /// <summary>The one request to a method and path.</summary>
    public Seen Only(string method, string path) => Assert.Single(To(method, path));

    public async ValueTask DisposeAsync()
    {
        await _app.StopAsync();
        await _app.DisposeAsync();
    }

    private async Task HandleAsync(HttpContext context)
    {
        var request = context.Request;
        JsonNode? body = null;
        if (request.ContentLength is > 0 || request.Headers.ContainsKey("Transfer-Encoding"))
        {
            using var reader = new StreamReader(request.Body, Encoding.UTF8);
            var text = await reader.ReadToEndAsync();
            body = text.Length > 0 ? JsonNode.Parse(text) : null;
        }
        var seen = new Seen(request.Method, request.Path.Value ?? "",
            request.Query.ToDictionary(pair => pair.Key, pair => pair.Value.ToString()),
            request.Headers.ToDictionary(pair => pair.Key, pair => pair.Value.ToString(), StringComparer.OrdinalIgnoreCase),
            body);
        _seen.Enqueue(seen);

        if (context.WebSockets.IsWebSocketRequest)
        {
            if (!_sockets.TryGetValue(seen.Path, out var socket))
            {
                context.Response.StatusCode = 404;
                return;
            }
            using var accepted = await context.WebSockets.AcceptWebSocketAsync();
            await using var peer = new Peer(accepted, seen);
            await socket(peer);
            await peer.CloseAsync();
            return;
        }

        // A path ending in * answers everything under it.
        var key = $"{seen.Method} {seen.Path}";
        var answer = _routes.TryGetValue(key, out var exact) ? exact
            : _routes.FirstOrDefault(route => route.Key.EndsWith('*') && key.StartsWith(route.Key[..^1], StringComparison.Ordinal)).Value;
        var reply = answer?.Invoke(seen) ?? new Reply(404, new { error = $"nothing answers {seen.Method} {seen.Path}" });
        context.Response.StatusCode = reply.Status;
        if (reply.Body is not null)
        {
            context.Response.ContentType = "application/json";
            await context.Response.WriteAsync(reply.Body as string ?? JsonSerializer.Serialize(reply.Body));
        }
    }
}

/// <summary>The server's end of one socket.</summary>
public sealed class Peer : IAsyncDisposable
{
    private readonly WebSocket _socket;
    private readonly Channel<JsonObject> _frames = Channel.CreateUnbounded<JsonObject>();
    private readonly Channel<byte[]> _audio = Channel.CreateUnbounded<byte[]>();
    private readonly Task _reading;

    internal Peer(WebSocket socket, Seen request)
    {
        _socket = socket;
        Request = request;
        _reading = Task.Run(ReadAsync);
    }

    /// <summary>The upgrade request.</summary>
    public Seen Request { get; }

    /// <summary>Whether the client has closed its end.</summary>
    public Task Closed => _reading;

    public Task SendAsync(object frame) =>
        _socket.SendAsync(JsonSerializer.SerializeToUtf8Bytes(frame), WebSocketMessageType.Text, true, CancellationToken.None);

    public Task SendAudioAsync(byte[] payload) =>
        _socket.SendAsync(payload, WebSocketMessageType.Binary, true, CancellationToken.None);

    /// <summary>The next JSON frame the client sent, or a failed test after five seconds.</summary>
    public async Task<JsonObject> ReceiveAsync()
    {
        using var patience = new CancellationTokenSource(TimeSpan.FromSeconds(5));
        return await _frames.Reader.ReadAsync(patience.Token);
    }

    /// <summary>The next frame of a type, skipping others.</summary>
    public async Task<JsonObject> ReceiveAsync(string type)
    {
        while (true)
        {
            var frame = await ReceiveAsync();
            if (frame["type"]?.GetValue<string>() == type)
            {
                return frame;
            }
        }
    }

    /// <summary>The next binary frame the client sent.</summary>
    public async Task<byte[]> ReceiveAudioAsync()
    {
        using var patience = new CancellationTokenSource(TimeSpan.FromSeconds(5));
        return await _audio.Reader.ReadAsync(patience.Token);
    }

    public async Task CloseAsync()
    {
        if (_socket.State is WebSocketState.Open or WebSocketState.CloseReceived)
        {
            try
            {
                await _socket.CloseAsync(WebSocketCloseStatus.NormalClosure, "", CancellationToken.None);
            }
            catch (WebSocketException)
            {
                // The client is already gone.
            }
        }
    }

    public async ValueTask DisposeAsync()
    {
        await CloseAsync();
        await _reading;
    }

    private async Task ReadAsync()
    {
        var buffer = new byte[1 << 20];
        try
        {
            while (_socket.State is WebSocketState.Open or WebSocketState.CloseSent)
            {
                using var message = new MemoryStream();
                WebSocketReceiveResult received;
                do
                {
                    received = await _socket.ReceiveAsync(buffer, CancellationToken.None);
                    message.Write(buffer, 0, received.Count);
                }
                while (!received.EndOfMessage);
                if (received.MessageType == WebSocketMessageType.Close)
                {
                    return;
                }
                if (received.MessageType == WebSocketMessageType.Binary)
                {
                    await _audio.Writer.WriteAsync(message.ToArray());
                }
                else if (JsonNode.Parse(message.ToArray()) is JsonObject frame)
                {
                    await _frames.Writer.WriteAsync(frame);
                }
            }
        }
        catch (WebSocketException)
        {
            // Gone without a close, which ends the read the same way.
        }
        finally
        {
            _frames.Writer.TryComplete();
            _audio.Writer.TryComplete();
        }
    }
}
