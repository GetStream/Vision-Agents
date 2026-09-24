using System.Buffers.Binary;
using System.Runtime.CompilerServices;
using System.Text.Json.Nodes;
using System.Threading.Channels;
using GetStream.VisionAgents.Models;

namespace GetStream.VisionAgents;

/// <summary>
/// Everything the backend routes, on its own rather than inside an agent.
/// </summary>
/// <remarks>
/// Each streaming modality has a realtime socket and a recording job, and search has
/// neither, because a question and its answer are one round trip. What the named config
/// holds is a default that a per-call option overrides.
/// </remarks>
public sealed class Router
{
    internal static readonly TimeSpan Poll = TimeSpan.FromSeconds(1);

    internal Router(VisionAgentsClient client, string? config, IReadOnlyDictionary<string, string>? tags)
    {
        Client = client;
        Config = config ?? "";
        Tags = tags is { Count: > 0 } ? new Dictionary<string, string>(tags) : null;
        Stt = new Transcribing(this);
        Tts = new Speaking(this);
        Llm = new Answering(this);
    }

    /// <summary>A stored router config, by name or id. Empty means every call says what it wants.</summary>
    public string Config { get; }

    /// <summary>Transcription, live or from a recording.</summary>
    public Transcribing Stt { get; }

    /// <summary>A voice, live or recorded.</summary>
    public Speaking Tts { get; }

    /// <summary>The model that answers.</summary>
    public Answering Llm { get; }

    internal VisionAgentsClient Client { get; }

    internal Dictionary<string, string>? Tags { get; }

    /// <summary>Answers one question out of what is true now.</summary>
    public Task<SearchAnswer> SearchAsync(string query, SearchOptions? options = null, CancellationToken cancellationToken = default) =>
        Client.PostAsync<SearchAnswer>("/v1/search", new SearchRequest
        {
            Query = query,
            Options = options,
            ConfigId = VisionAgentsClient.Blank(Config),
            Tags = Tags,
        }, cancellationToken);

    /// <summary>Dials one modality's socket and sends the start frame that says what it is for.</summary>
    internal async Task<Socket> OpenAsync(string modality, object? options, string? target, CancellationToken cancellationToken)
    {
        if (Config == "" && string.IsNullOrEmpty(target))
        {
            throw new ConfigurationException("routing needs a target, either in the options or in a config");
        }
        var socket = await Socket.ConnectAsync(Client.Backend, Client.Backend.SocketUrl($"/v1/{modality}/stream"), cancellationToken)
            .ConfigureAwait(false);
        try
        {
            // An empty block is a socket that takes everything from its config. The router
            // refuses a frame whose target is only inside the block, so it is sent beside it too.
            await socket.SendAsync(Frames.Of("start", ("config_id", Config), ("target", VisionAgentsClient.Blank(target)), ("tags", Tags),
                (modality, options ?? new JsonObject())),
                cancellationToken).ConfigureAwait(false);
        }
        catch
        {
            await socket.DisposeAsync().ConfigureAwait(false);
            throw;
        }
        return socket;
    }

    internal async Task<T> SettleAsync<T>(T job, Func<T, string> status, string path, CancellationToken cancellationToken)
    {
        while (status(job) is "queued" or "running")
        {
            await Task.Delay(Poll, cancellationToken).ConfigureAwait(false);
            job = await Client.GetAsync<T>(path, cancellationToken: cancellationToken).ConfigureAwait(false);
        }
        return job;
    }
}

/// <summary>A whole recording to transcribe: a URL the provider fetches, or the bytes themselves.</summary>
public sealed record Recorded
{
    /// <summary>A fetchable audio or video file, which is what anything longer than a clip should be.</summary>
    public string? Url { get; init; }

    /// <summary>The file, for a clip with nowhere to host it.</summary>
    public byte[]? Audio { get; init; }

    /// <summary>Completes a short recording synchronously, storing nothing.</summary>
    public bool Inline { get; init; }

    /// <summary>A URL the finished job is POSTed to. Set, the job is returned as soon as it is accepted.</summary>
    public string? Callback { get; init; }

    /// <summary>A local recording, for a clip small enough to send.</summary>
    public static Recorded File(string path) => new() { Audio = System.IO.File.ReadAllBytes(path) };
}

/// <summary>Routes transcription.</summary>
public sealed class Transcribing
{
    private readonly Router _router;

    internal Transcribing(Router router) => _router = router;

    /// <summary>Opens a transcription socket, configured and ready for audio.</summary>
    public async Task<Transcriber> RealtimeAsync(SttOptions? options = null, CancellationToken cancellationToken = default) =>
        new(await _router.OpenAsync("stt", options, options?.Target, cancellationToken).ConfigureAwait(false));

    /// <summary>
    /// Transcribes a whole recording and waits for the transcript.
    /// </summary>
    /// <remarks>
    /// Served by the batch half of a vendor rather than the streaming one, which is both
    /// cheaper and more accurate. A job that failed is returned with its status and error.
    /// </remarks>
    public async Task<Transcription> RecordingAsync(Recorded recording, SttOptions? options = null, CancellationToken cancellationToken = default)
    {
        if (recording.Url is { Length: > 0 } == recording.Audio is { Length: > 0 })
        {
            throw new ConfigurationException("a recording is either a url or the audio itself, and one of them");
        }
        var job = await _router.Client.PostAsync<Transcription>("/v1/stt/recordings", new TranscriptionRequest
        {
            Source = new RecordingSource { Url = VisionAgentsClient.Blank(recording.Url), Audio = recording.Audio is { Length: > 0 } ? recording.Audio : null },
            Options = options,
            Inline = recording.Inline ? true : null,
            Callback = VisionAgentsClient.Blank(recording.Callback),
            ConfigId = VisionAgentsClient.Blank(_router.Config),
            Tags = _router.Tags,
        }, cancellationToken).ConfigureAwait(false);
        return recording.Callback is { Length: > 0 }
            ? job
            : await _router.SettleAsync(job, done => done.Status, $"/v1/stt/recordings/{VisionAgentsClient.Escape(job.Id)}", cancellationToken)
                .ConfigureAwait(false);
    }
}

/// <summary>Routes a voice.</summary>
public sealed class Speaking
{
    private readonly Router _router;

    internal Speaking(Router router) => _router = router;

    /// <summary>Opens a speech socket, configured and ready for text.</summary>
    public async Task<Voice> RealtimeAsync(TtsOptions? options = null, CancellationToken cancellationToken = default) =>
        new(await _router.OpenAsync("tts", options, options?.Target, cancellationToken).ConfigureAwait(false));

    /// <summary>
    /// Speaks a whole text into one file and waits for it. Nothing is listening while it is
    /// made, which is what lets a codec and a bitrate be chosen.
    /// </summary>
    /// <param name="text">What to say.</param>
    /// <param name="options">The voice and the file.</param>
    /// <param name="inline">Synthesizes a short reply synchronously, storing nothing.</param>
    /// <param name="cancellationToken">Stops waiting.</param>
    public async Task<Speech> RecordingAsync(string text, TtsOptions? options = null, bool inline = false, CancellationToken cancellationToken = default)
    {
        var job = await _router.Client.PostAsync<Speech>("/v1/tts/recordings", new SpeechRequest
        {
            Text = text,
            Options = options,
            Inline = inline ? true : null,
            ConfigId = VisionAgentsClient.Blank(_router.Config),
            Tags = _router.Tags,
        }, cancellationToken).ConfigureAwait(false);
        return await _router.SettleAsync(job, done => done.Status, $"/v1/tts/recordings/{VisionAgentsClient.Escape(job.Id)}", cancellationToken)
            .ConfigureAwait(false);
    }
}

/// <summary>Routes completions.</summary>
public sealed class Answering
{
    private readonly Router _router;

    internal Answering(Router router) => _router = router;

    /// <summary>Opens a completions socket, configured and ready for a question.</summary>
    public async Task<LanguageModel> RealtimeAsync(LlmOptions? options = null, CancellationToken cancellationToken = default) =>
        new(await _router.OpenAsync("llm", options, options?.Target, cancellationToken).ConfigureAwait(false));
}

/// <summary>An open socket whose frames are read into a stream of <typeparamref name="T"/>.</summary>
public abstract class Realtime<T> : IAsyncDisposable
{
    // Anybody keeping up never fills it; somebody who is not is told by the socket slowing.
    private readonly Channel<T> _read = Channel.CreateBounded<T>(new BoundedChannelOptions(64) { SingleReader = true });
    private readonly CancellationTokenSource _stop = new();
    private readonly Task _reading;

    private protected Realtime(Socket socket)
    {
        Socket = socket;
        _reading = Task.Run(ReadAsync);
    }

    private protected Socket Socket { get; }

    /// <summary>What arrives, until the socket closes.</summary>
    public async IAsyncEnumerable<T> ReadAllAsync([EnumeratorCancellation] CancellationToken cancellationToken = default)
    {
        await foreach (var item in _read.Reader.ReadAllAsync(cancellationToken).ConfigureAwait(false))
        {
            yield return item;
        }
    }

    /// <summary>Closes the socket. Safe to call twice.</summary>
    public async ValueTask DisposeAsync()
    {
        // Stopped first, since a reader nobody drains is waiting for room that never comes.
        await _stop.CancelAsync().ConfigureAwait(false);
        await Socket.DisposeAsync().ConfigureAwait(false);
        await _reading.ConfigureAwait(false);
        _stop.Dispose();
        GC.SuppressFinalize(this);
    }

    private protected abstract T? Read(Frame? frame, byte[]? audio);

    private async Task ReadAsync()
    {
        try
        {
            while (await Socket.ReceiveAsync(_stop.Token).ConfigureAwait(false) is { } message)
            {
                if (Read(message.Frame, message.Audio) is { } item)
                {
                    await _read.Writer.WriteAsync(item, _stop.Token).ConfigureAwait(false);
                }
            }
        }
        catch (OperationCanceledException)
        {
            // Closed.
        }
        finally
        {
            _read.Writer.TryComplete();
        }
    }
}

/// <summary>One thing the transcriber heard.</summary>
/// <param name="Text">What was said.</param>
/// <param name="Final">Whether it is settled, rather than a guess that will be revised.</param>
/// <param name="Speaker">Who said it, when diarization was asked for.</param>
/// <param name="Language">The language it was heard in.</param>
/// <param name="Provider">What answered. A config names several and routing picks per session.</param>
/// <param name="Model">The provider's model.</param>
/// <param name="Error">Why nothing was heard, when that is what happened.</param>
/// <param name="Frame">The frame as it arrived.</param>
public sealed record Transcript(string Text, bool Final, string Speaker, string Language, string Provider, string Model, string Error, Frame Frame);

/// <summary>One open transcription socket.</summary>
public sealed class Transcriber : Realtime<Transcript>
{
    internal Transcriber(Socket socket) : base(socket)
    {
    }

    /// <summary>Hands over 16 kHz mono 16-bit PCM to be transcribed.</summary>
    public Task SendAsync(ReadOnlyMemory<byte> pcm, CancellationToken cancellationToken = default) =>
        Socket.SendAudioAsync(pcm, cancellationToken);

    private protected override Transcript? Read(Frame? frame, byte[]? audio) =>
        frame is { Type.Length: > 0 }
            ? new Transcript(frame.Text("text"), frame.Flag("final"), frame.Text("speaker"), frame.Text("language"),
                frame.Text("provider"), frame.Text("model"), frame.Text("error"), frame)
            : null;
}

/// <summary>One piece of speech, as the provider produced it.</summary>
/// <param name="Samples">Signed 16-bit little-endian PCM.</param>
/// <param name="SampleRate">How many samples a second.</param>
/// <param name="Channels">How many channels are interleaved.</param>
/// <param name="Done">The utterance is finished; there are no samples.</param>
/// <param name="Error">Why nothing was spoken, when that is what happened.</param>
public sealed record Audio(ReadOnlyMemory<byte> Samples, int SampleRate, int Channels, bool Done = false, string Error = "");

/// <summary>One open speech socket.</summary>
public sealed class Voice : Realtime<Audio>
{
    internal Voice(Socket socket) : base(socket)
    {
    }

    /// <summary>
    /// Says text, whose audio arrives on <see cref="Realtime{T}.ReadAllAsync"/>. One utterance
    /// at a time: the frames come back bare, so two overlapping ones would be indistinguishable.
    /// </summary>
    public Task SpeakAsync(string text, CancellationToken cancellationToken = default) =>
        Socket.SendAsync(Frames.Of("speak", ("text", text), ("final", true)), cancellationToken);

    /// <summary>Abandons what is being spoken.</summary>
    public Task InterruptAsync(CancellationToken cancellationToken = default) =>
        Socket.SendAsync(Frames.Of("interrupt"), cancellationToken);

    private protected override Audio? Read(Frame? frame, byte[]? audio)
    {
        // Each binary frame starts with a header saying how to play what follows it.
        if (audio is { Length: >= 8 })
        {
            return new Audio(audio.AsMemory(8), (int)BinaryPrimitives.ReadUInt32LittleEndian(audio),
                BinaryPrimitives.ReadUInt16LittleEndian(audio.AsSpan(4)));
        }
        return frame?.Type switch
        {
            "synthesis_complete" => new Audio(ReadOnlyMemory<byte>.Empty, 0, 0, Done: true),
            "error" => new Audio(ReadOnlyMemory<byte>.Empty, 0, 0, Error: frame.Text("error")),
            _ => null,
        };
    }
}

/// <summary>One turn of a conversation handed to the model.</summary>
/// <param name="Role">user, assistant or system.</param>
/// <param name="Content">What was said.</param>
public sealed record Said(string Role, string Content);

/// <summary>
/// What to answer and how, in the response parameters the router speaks. What the config
/// holds fills in whatever is left null.
/// </summary>
public sealed record Question
{
    /// <summary>The conversation so far, oldest first.</summary>
    public required IReadOnlyList<Said> Messages { get; init; }

    /// <summary>What the model answers under.</summary>
    public string? Instructions { get; init; }

    /// <summary>What the model may ask to have run.</summary>
    public IReadOnlyList<SessionTool>? Tools { get; init; }

    /// <summary>auto, none, required, or the name of a tool it must call.</summary>
    public string? ToolChoice { get; init; }

    /// <summary>Caps the reply.</summary>
    public int? MaxOutputTokens { get; init; }

    /// <summary>How varied the answer is.</summary>
    public double? Temperature { get; init; }

    /// <summary>How considered the answer is.</summary>
    public string? ReasoningEffort { get; init; }

    /// <summary>text or json_object.</summary>
    public string? Format { get; init; }

    /// <summary>How much of it there is.</summary>
    public string? Verbosity { get; init; }

    /// <summary>Continues from an earlier answer the provider still holds.</summary>
    public string? PreviousResponseId { get; init; }

    /// <summary>What a cached prompt prefix is keyed by.</summary>
    public string? PromptCacheKey { get; init; }

    /// <summary>Passed to the provider untouched.</summary>
    public IReadOnlyDictionary<string, string>? Metadata { get; init; }
}

/// <summary>The reply arriving as it is written.</summary>
/// <param name="Delta">The next piece of text, on a delta.</param>
/// <param name="Text">The whole answer, on the frame that finishes it.</param>
/// <param name="Done">Whether this finishes it.</param>
/// <param name="Error">Why there is no answer, when that is what happened.</param>
/// <param name="Frame">The frame as it arrived.</param>
public sealed record Answer(string Delta, string Text, bool Done, string Error, Frame Frame);

/// <summary>One open completions socket.</summary>
public sealed class LanguageModel : Realtime<Answer>
{
    internal LanguageModel(Socket socket) : base(socket)
    {
    }

    /// <summary>Sends a question, whose answer arrives on <see cref="Realtime{T}.ReadAllAsync"/>.</summary>
    public Task AskAsync(Question question, CancellationToken cancellationToken = default) =>
        Socket.SendAsync(Frames.Of("respond",
            ("messages", question.Messages.Select(said => new JsonObject { ["role"] = said.Role, ["content"] = said.Content }).ToList()),
            ("instructions", VisionAgentsClient.Blank(question.Instructions)),
            ("tools", question.Tools is { Count: > 0 } ? question.Tools : null),
            ("tool_choice", VisionAgentsClient.Blank(question.ToolChoice)),
            ("max_output_tokens", question.MaxOutputTokens is > 0 ? question.MaxOutputTokens : null),
            ("temperature", question.Temperature),
            ("reasoning_effort", VisionAgentsClient.Blank(question.ReasoningEffort)),
            ("format", VisionAgentsClient.Blank(question.Format)),
            ("verbosity", VisionAgentsClient.Blank(question.Verbosity)),
            ("previous_response_id", VisionAgentsClient.Blank(question.PreviousResponseId)),
            ("prompt_cache_key", VisionAgentsClient.Blank(question.PromptCacheKey)),
            ("metadata", question.Metadata is { Count: > 0 } ? question.Metadata : null)), cancellationToken);

    /// <summary>Abandons the answer in flight.</summary>
    public Task InterruptAsync(CancellationToken cancellationToken = default) =>
        Socket.SendAsync(Frames.Of("interrupt"), cancellationToken);

    private protected override Answer? Read(Frame? frame, byte[]? audio) => frame?.Type switch
    {
        "delta" => new Answer(frame.Text("text"), "", false, "", frame),
        "complete" => new Answer("", frame.Text("text"), true, "", frame),
        "error" => new Answer("", "", false, frame.Text("error"), frame),
        _ => null,
    };
}
