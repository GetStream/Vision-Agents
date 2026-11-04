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

// caller is the server ROUTER_AUDIO_FLOOR_URL names and AUDIO_FLOOR_WAV's audio, a 16 kHz
// mono 16-bit WAV of a caller asking something.
func caller(t *testing.T) (string, []int16) {
	url, wav := os.Getenv("ROUTER_AUDIO_FLOOR_URL"), os.Getenv("AUDIO_FLOOR_WAV")
	if url == "" || wav == "" {
		t.Skip("needs ROUTER_AUDIO_FLOOR_URL and AUDIO_FLOOR_WAV")
	}
	raw, err := os.ReadFile(wav)
	require.NoError(t, err)
	at := bytes.Index(raw, []byte("data"))
	require.Positive(t, at, "not a WAV")
	pcm := make([]int16, binary.LittleEndian.Uint32(raw[at+4:])/2)
	require.NoError(t, binary.Read(bytes.NewReader(raw[at+8:]), binary.LittleEndian, pcm))
	return url, pcm
}

// A real audio floor server rules on a caller's question and writes it down.
func TestARealServerRulesAndTranscribes(t *testing.T) {
	url, pcm := caller(t)
	// The caller has been quiet a while when the turn is put.
	pcm = append(pcm, make([]int16, SampleRate*6/10)...)

	ctx, cancel := context.WithTimeout(context.Background(), 2*time.Minute)
	defer cancel()
	client := New(url, nil)
	turn := harness.FlowTurn{
		Instructions: "You answer the phone for a weather service.",
		Tools:        []llm.Tool{{Name: "get_weather", Description: "Get the forecast for a city.", Parameters: map[string]any{"required": []any{"city"}}}},
		Audio:        pcm,
		Paused:       600 * time.Millisecond,
		FirstPause:   true,
	}
	hearer := NewHearer(client)
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
	body, _, err := decisionBody(turnOf(turn), pcm)
	require.NoError(t, err)
	t.Logf("decision: median %v over %d, server %.0f ms, %d bytes; answer %s %.3f, p_eot %.3f, bar %.3f, ruling %+v",
		took[len(took)/2], len(took), check.Answer.DecideMs, len(body), check.Answer.Answer, check.Answer.AnswerProbability,
		*check.Answer.PEOT, check.Bar, check.Ruling)

	took = took[:0]
	var said Answer
	for range 3 {
		start := time.Now()
		said, err = client.Decide(ctx, Agent{}, Turn{}, pcm, true)
		require.NoError(t, err)
		took = append(took, time.Since(start))
	}
	slices.Sort(took)
	t.Logf("transcript: median %v over %d, of %.2f s: %q", took[len(took)/2], len(took), float64(len(pcm))/SampleRate, said.Transcript)
	require.NotEmpty(t, said.Transcript)
}

// The transcriber writes the caller's question at its pauses and ends it at the cap.
func TestARealServerTranscribesAnUtterance(t *testing.T) {
	url, pcm := caller(t)
	transcriber := NewTranscriber(New(url, nil), "", slog.New(slog.DiscardHandler))
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
