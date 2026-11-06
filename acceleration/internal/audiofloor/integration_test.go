//go:build integration

package audiofloor

import (
	"bytes"
	"context"
	"encoding/binary"
	"log/slog"
	"os"
	"slices"
	"testing"
	"time"

	"github.com/stretchr/testify/require"

	"github.com/GetStream/Vision-Agents/acceleration/internal/audio"
	"github.com/GetStream/Vision-Agents/acceleration/internal/harness"
	"github.com/GetStream/Vision-Agents/acceleration/internal/llm"
	"github.com/GetStream/Vision-Agents/acceleration/internal/stt"
)

// recording is the server ROUTER_CHORUS_ASR_URL names and CHORUS_ASR_WAV's audio, a 16 kHz
// mono 16-bit WAV of a caller asking something.
func recording(t *testing.T) (string, []int16) {
	url, wav := os.Getenv("ROUTER_CHORUS_ASR_URL"), os.Getenv("CHORUS_ASR_WAV")
	if url == "" || wav == "" {
		t.Skip("needs ROUTER_CHORUS_ASR_URL and CHORUS_ASR_WAV")
	}
	raw, err := os.ReadFile(wav)
	require.NoError(t, err)
	at := bytes.Index(raw, []byte("data"))
	require.Positive(t, at, "not a WAV")
	pcm := make([]int16, binary.LittleEndian.Uint32(raw[at+4:])/2)
	require.NoError(t, binary.Read(bytes.NewReader(raw[at+8:]), binary.LittleEndian, pcm))
	return url, pcm
}

// streamed is pcm as a stream of format sends its last 16 s.
func streamed(t *testing.T, format Format, pcm []int16) harness.Audio {
	stream, err := newStream(format, Window)
	require.NoError(t, err)
	for frame := range slices.Chunk(pcm, frameSamples) {
		require.NoError(t, stream.Write(frame))
	}
	return stream.Last(Window)
}

// A real chorus-asr server rules on a caller's question, in Opus as in PCM, and writes it
// down with no decision.
func TestARealServerRulesAndTranscribes(t *testing.T) {
	url, pcm := recording(t)
	// The caller has been quiet a while when the turn is put.
	pcm = append(pcm, make([]int16, SampleRate*6/10)...)

	ctx, cancel := context.WithTimeout(context.Background(), 2*time.Minute)
	defer cancel()
	client := New(url, Opus, nil).WithAPIKey(os.Getenv("ROUTER_CHORUS_ASR_API_KEY"))
	hearer := NewHearer(client, EndOfTurn{}, slog.New(slog.DiscardHandler))
	for _, format := range []Format{PCM, Opus} {
		turn := harness.FlowTurn{
			Instructions: "You answer the phone for a weather service.",
			Tools:        []llm.Tool{{Name: "get_weather", Description: "Get the forecast for a city.", Parameters: map[string]any{"required": []any{"city"}}}},
			Audio:        streamed(t, format, pcm),
			Paused:       600 * time.Millisecond,
			FirstPause:   true,
		}
		_, err := hearer.Check(ctx, turn) // registers the agent and evaluates its prompt
		require.NoError(t, err)
		var took []time.Duration
		var check Check
		for range 5 {
			start := time.Now()
			check, err = hearer.Check(ctx, turn)
			require.NoError(t, err)
			took = append(took, time.Since(start))
		}
		slices.Sort(took)
		body, _, err := decisionBody(turnOf(turn), turn.Audio)
		require.NoError(t, err)
		t.Logf("%s decision: median %v over %d, server %.0f ms, %d bytes; answer %s %.3f, p_eot %.4f, bar %.3f, ruling %+v",
			format, took[len(took)/2], len(took), check.Answer.DecideMs, len(body), check.Answer.Answer, check.Answer.AnswerProbability,
			*check.Answer.PEOT, check.Bar, check.Ruling)
	}

	took := []time.Duration{}
	var said Words
	for range 3 {
		start := time.Now()
		var err error
		said, err = client.Transcribe(ctx, streamed(t, Opus, pcm), []string{"en"})
		require.NoError(t, err)
		took = append(took, time.Since(start))
	}
	slices.Sort(took)
	t.Logf("transcript: median %v over %d, of %.2f s: %q in %s", took[len(took)/2], len(took), float64(len(pcm))/SampleRate, said.Text, said.Language)
	require.NotEmpty(t, said.Text)
}

// The transcriber writes the caller's question at its pauses and ends it at the cap.
func TestARealServerTranscribesAnUtterance(t *testing.T) {
	url, pcm := recording(t)
	transcriber := NewTranscriber(New(url, Opus, nil).WithAPIKey(os.Getenv("ROUTER_CHORUS_ASR_API_KEY")), "", []string{"en"}, slog.New(slog.DiscardHandler))
	require.NoError(t, transcriber.Start(context.Background()))
	defer transcriber.Close()
	pcm = append(append(make([]int16, SampleRate), pcm...), make([]int16, 3*SampleRate)...)
	for frame := range slices.Chunk(pcm, SampleRate/50) {
		require.NoError(t, transcriber.ProcessAudio(audio.PcmData{Samples: frame, SampleRate: SampleRate, Channels: 1}, stt.Participant{ID: "p"}))
	}
	for event := range transcriber.Events() {
		said, ok := event.(stt.Transcript)
		if !ok {
			require.IsType(t, stt.Connected{}, event)
			continue
		}
		t.Logf("%s %d: %.0f ms of audio written in %.0f ms: %q", said.Mode, said.Utterance, said.AudioDurationMs, said.ProcessingTimeMs, said.Text)
		if said.Final() {
			require.NotEmpty(t, said.Text)
			return
		}
	}
}
