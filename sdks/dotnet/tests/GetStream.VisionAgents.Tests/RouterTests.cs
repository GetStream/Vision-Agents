using System.Buffers.Binary;
using System.Text.Json.Nodes;
using GetStream.VisionAgents.Models;

namespace GetStream.VisionAgents.Tests;

public sealed class RouterTests
{
    [Fact]
    public async Task SearchIsAskedUnderTheConfigWithItsLabels()
    {
        await using var router = await TestRouter.StartAsync();
        router.On("POST", "/v1/search", 200, new { provider = "exa", model = "exa", answer = "Paris", results = Array.Empty<object>() });
        using var client = Fixtures.Client(router);

        var answer = await client.Router("research", new Dictionary<string, string> { ["team"] = "docs" })
            .SearchAsync("capital of France", cancellationToken: TestContext.Current.CancellationToken);

        Assert.Equal("Paris", answer.Answer);
        var asked = router.Only("POST", "/v1/search").Body;
        Assert.Equal(("capital of France", "research", "docs"), (asked.Text("query"), asked.Text("config_id"), asked?["tags"].Text("team")));
    }

    [Fact]
    public async Task ARecordingIsTranscribedAndWaitedFor()
    {
        await using var router = await TestRouter.StartAsync();
        router.On("POST", "/v1/stt/recordings", 202, Job("queued"));
        router.On("GET", "/v1/stt/recordings/job-1", 200, Job("completed", "hello there"));
        using var client = Fixtures.Client(router);

        var transcript = await client.Router().Stt.RecordingAsync(new Recorded { Audio = [1, 2, 3] },
            new SttOptions { Target = "stt-batch", Diarize = true }, TestContext.Current.CancellationToken);

        Assert.Equal(("completed", "hello there"), (transcript.Status, transcript.Text));
        var sent = router.Only("POST", "/v1/stt/recordings").Body;
        Assert.Equal("AQID", sent?["source"].Text("audio"));
        Assert.Equal("stt-batch", sent?["options"].Text("target"));
        Assert.Null(sent?["config_id"]);
    }

    [Fact]
    public async Task ARecordingIsAUrlOrTheAudioButNotBoth()
    {
        await using var router = await TestRouter.StartAsync();
        using var client = Fixtures.Client(router);
        var stt = client.Router().Stt;

        await Assert.ThrowsAsync<ConfigurationException>(() => stt.RecordingAsync(new Recorded(), cancellationToken: TestContext.Current.CancellationToken));
        await Assert.ThrowsAsync<ConfigurationException>(() =>
            stt.RecordingAsync(new Recorded { Url = "https://x/a.mp3", Audio = [1] }, cancellationToken: TestContext.Current.CancellationToken));
    }

    [Fact]
    public async Task ARealtimeTranscriberIsStartedThenFedAudio()
    {
        await using var router = await TestRouter.StartAsync();
        JsonObject? start = null;
        byte[]? heard = null;
        router.OnSocket("/v1/stt/stream", async peer =>
        {
            start = await peer.ReceiveAsync("start");
            heard = await peer.ReceiveAudioAsync();
            await peer.SendAsync(new { type = "transcript", text = "hello", final = true, provider = "deepgram", model = "nova-3" });
        });
        using var client = Fixtures.Client(router);
        var cancel = TestContext.Current.CancellationToken;

        await using var transcriber = await client.Router().Stt.RealtimeAsync(new SttOptions { Target = "stt-fast" }, cancel);
        await transcriber.SendAsync(new byte[] { 0, 1, 0, 1 }, cancel);
        var transcripts = new List<Transcript>();
        await foreach (var transcript in transcriber.ReadAllAsync(cancel))
        {
            transcripts.Add(transcript);
        }

        Assert.Equal("stt-fast", start?["stt"].Text("target"));
        Assert.Equal("stt-fast", start.Text("target"));
        Assert.Equal("", start.Text("config_id"));
        Assert.Equal([0, 1, 0, 1], heard);
        var only = Assert.Single(transcripts);
        Assert.Equal(("hello", true, "deepgram"), (only.Text, only.Final, only.Provider));
    }

    [Fact]
    public async Task ASocketWithNothingToRouteToIsRefusedBeforeItOpens()
    {
        await using var router = await TestRouter.StartAsync();
        using var client = Fixtures.Client(router);

        await Assert.ThrowsAsync<ConfigurationException>(() => client.Router().Tts.RealtimeAsync(cancellationToken: TestContext.Current.CancellationToken));
        Assert.Empty(router.Requests);
    }

    [Fact]
    public async Task AVoiceSpeaksAndItsAudioCarriesHowToPlayIt()
    {
        await using var router = await TestRouter.StartAsync();
        JsonObject? spoken = null;
        router.OnSocket("/v1/tts/stream", async peer =>
        {
            await peer.ReceiveAsync("start");
            spoken = await peer.ReceiveAsync("speak");
            var frame = new byte[12];
            BinaryPrimitives.WriteUInt32LittleEndian(frame, 24_000);
            BinaryPrimitives.WriteUInt16LittleEndian(frame.AsSpan(4), 1);
            frame[8] = 7;
            await peer.SendAudioAsync(frame);
            await peer.SendAsync(new { type = "synthesis_complete" });
        });
        using var client = Fixtures.Client(router);
        var cancel = TestContext.Current.CancellationToken;

        await using var voice = await client.Router("voices").Tts.RealtimeAsync(cancellationToken: cancel);
        await voice.SpeakAsync("Hello.", cancel);
        var audio = new List<Audio>();
        await foreach (var piece in voice.ReadAllAsync(cancel))
        {
            audio.Add(piece);
        }

        Assert.Equal(("Hello.", true), (spoken.Text("text"), spoken!["final"]!.GetValue<bool>()));
        Assert.Equal((24_000, 1, 4, (byte)7), (audio[0].SampleRate, audio[0].Channels, audio[0].Samples.Length, audio[0].Samples.Span[0]));
        Assert.True(audio[1].Done);
    }

    [Fact]
    public async Task ASpeechRecordingIsWaitedFor()
    {
        await using var router = await TestRouter.StartAsync();
        router.On("POST", "/v1/tts/recordings", 202, Job("running"));
        router.On("GET", "/v1/tts/recordings/job-1", 200, new { id = "job-1", status = "completed", url = "https://x/speech.mp3", created_at = "2026-09-24T10:00:00Z", updated_at = "2026-09-24T10:00:00Z" });
        using var client = Fixtures.Client(router);

        var speech = await client.Router().Tts.RecordingAsync("Hello.", new TtsOptions { Target = "tts-fast" }, cancellationToken: TestContext.Current.CancellationToken);

        Assert.Equal("https://x/speech.mp3", speech.Url);
        Assert.Equal("Hello.", router.Only("POST", "/v1/tts/recordings").Body.Text("text"));
    }

    [Fact]
    public async Task AModelIsAskedAndItsAnswerArrivesAsItIsWritten()
    {
        await using var router = await TestRouter.StartAsync();
        JsonObject? asked = null;
        router.OnSocket("/v1/llm/stream", async peer =>
        {
            await peer.ReceiveAsync("start");
            asked = await peer.ReceiveAsync("respond");
            await peer.SendAsync(new { type = "delta", text = "Hel" });
            await peer.SendAsync(new { type = "delta", text = "lo." });
            await peer.SendAsync(new { type = "complete", text = "Hello." });
        });
        using var client = Fixtures.Client(router);
        var cancel = TestContext.Current.CancellationToken;

        await using var model = await client.Router().Llm.RealtimeAsync(new LlmOptions { Target = "llm-fast" }, cancel);
        await model.AskAsync(new Question { Messages = [new Said("user", "Say hello.")], Instructions = "Be brief.", Temperature = 0.2 }, cancel);
        var answers = new List<Answer>();
        await foreach (var answer in model.ReadAllAsync(cancel))
        {
            answers.Add(answer);
        }

        Assert.Equal(("user", "Say hello."), (asked?["messages"]![0].Text("role"), asked?["messages"]![0].Text("content")));
        Assert.Equal(("Be brief.", 0.2), (asked.Text("instructions"), asked!["temperature"]!.GetValue<double>()));
        Assert.Null(asked["tools"]);
        Assert.Equal("Hello.", string.Concat(answers.Select(answer => answer.Delta)));
        Assert.True(answers[^1].Done);
    }

    private static object Job(string status, string? text = null) => new
    {
        id = "job-1", status, text, created_at = "2026-09-24T10:00:00Z", updated_at = "2026-09-24T10:00:00Z",
    };
}
