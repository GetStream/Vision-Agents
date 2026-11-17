package agent

import (
	"bufio"
	"bytes"
	"encoding/json"
	"fmt"
	"log/slog"
	"sync"
	"testing"

	"github.com/prometheus/client_golang/prometheus/testutil"
	"github.com/stretchr/testify/suite"

	"github.com/GetStream/Vision-Agents/acceleration/internal/audio"
	"github.com/GetStream/Vision-Agents/acceleration/internal/tts"
)

// overlapWarning is the message logged when two utterances are published at once.
const overlapWarning = "two utterances are being published at once, so they are heard spliced together"

// logLines keeps what an agent logs as JSON, one record a line.
type logLines struct {
	mu  sync.Mutex
	buf bytes.Buffer
}

func (l *logLines) Write(p []byte) (int, error) {
	l.mu.Lock()
	defer l.mu.Unlock()
	return l.buf.Write(p)
}

// records returns every record logged with message.
func (l *logLines) records(message string) []map[string]any {
	l.mu.Lock()
	defer l.mu.Unlock()
	var found []map[string]any
	lines := bufio.NewScanner(bytes.NewReader(l.buf.Bytes()))
	for lines.Scan() {
		var record map[string]any
		if json.Unmarshal(lines.Bytes(), &record) == nil && record["msg"] == message {
			found = append(found, record)
		}
	}
	return found
}

// keyCounts returns, for every record logged with message, how many times key appears in
// it: a record with a key twice decodes to one value and hides the other.
func (l *logLines) keyCounts(message, key string) []int {
	l.mu.Lock()
	defer l.mu.Unlock()
	var counts []int
	lines := bufio.NewScanner(bytes.NewReader(l.buf.Bytes()))
	for lines.Scan() {
		var record map[string]any
		if json.Unmarshal(lines.Bytes(), &record) != nil || record["msg"] != message {
			continue
		}
		decoder := json.NewDecoder(bytes.NewReader(lines.Bytes()))
		count := 0
		if _, err := decoder.Token(); err != nil {
			panic(err)
		}
		for decoder.More() {
			name, err := decoder.Token()
			if err != nil {
				panic(err)
			}
			var value json.RawMessage
			if err := decoder.Decode(&value); err != nil {
				panic(err)
			}
			if name == key {
				count++
			}
		}
		counts = append(counts, count)
	}
	return counts
}

// chunkOf is a piece of an utterance whose samples all hold value, so where it went is
// told by its length and its sum.
func chunkOf(synthesisID string, samples int, value int16) tts.AudioChunk {
	pcm := make([]int16, samples)
	for i := range pcm {
		pcm[i] = value
	}
	return tts.AudioChunk{SynthesisID: synthesisID, Audio: audio.PcmData{Samples: pcm, SampleRate: 16_000, Channels: 1}}
}

// voiceSends has the voice emit events in order and waits until the agent reports spoken
// utterances finished, so it has handled every event up to the last completion.
func (s *AgentSuite) voiceSends(spoken int, events ...tts.Event) {
	for _, event := range events {
		s.voice.emitter.Send(event)
	}
	s.eventually(func() bool { return countOf[Spoke](s.reported()) == spoken }, "the agent did not finish the voice's utterances")
}

// interleaved is a reply and a tool's reply whose audio arrives mixed, the way a streaming
// voice sends two open utterances.
func interleaved() []tts.Event {
	return []tts.Event{
		tts.SynthesisStarted{SynthesisID: "turn-a", Provider: "stub"},
		chunkOf("turn-a", 160, 1000),
		tts.SynthesisStarted{SynthesisID: "tool-b", Provider: "stub"},
		chunkOf("tool-b", 240, 2000),
		chunkOf("turn-a", 320, 3000),
		chunkOf("tool-b", 400, 4000),
		tts.SynthesisComplete{SynthesisID: "turn-a"},
		tts.SynthesisComplete{SynthesisID: "tool-b"},
	}
}

func (s *AgentSuite) TestTwoUtterancesPublishedAtOnceAreWarnedAboutOnce() {
	s.logs = &logLines{}
	s.join(true)
	pairs := testutil.ToFloat64(speechOverlaps.WithLabelValues("stub"))
	seconds := testutil.ToFloat64(speechOverlapSeconds.WithLabelValues("stub"))

	s.voiceSends(2, interleaved()...)

	warnings := s.logs.records(overlapWarning)
	s.Require().Len(warnings, 1, "one pair is one warning, however many chunks interleave")
	s.Equal("WARN", warnings[0]["level"])
	s.Equal("tool-b", warnings[0]["synthesis"])
	s.Equal("tool-b", warnings[0]["turn"])
	s.Equal("turn-a", warnings[0]["other_synthesis"])
	s.Equal("turn-a", warnings[0]["other_turn"])
	s.Equal("stub", warnings[0]["provider"])
	s.Equal(1.0, testutil.ToFloat64(speechOverlaps.WithLabelValues("stub"))-pairs)
	// The 15, 20 and 25 ms chunks went out while the other utterance was open.
	s.InDelta(0.060, testutil.ToFloat64(speechOverlapSeconds.WithLabelValues("stub"))-seconds, 1e-9)
}

func (s *AgentSuite) TestAudioPublishedAtOnceIsPublishedAsBefore() {
	s.join(true)

	s.voiceSends(2, interleaved()...)

	var heard []string
	for _, pcm := range s.edge.heard() {
		var sum int64
		for _, sample := range pcm.Samples {
			sum += int64(sample)
		}
		heard = append(heard, fmt.Sprintf("%d:%d", len(pcm.Samples), sum))
	}
	// Probed on eb44524c5, before overlaps were noticed.
	s.Equal([]string{"160:245858", "240:822549", "320:1099365", "400:1600000"}, heard)
}

func (s *AgentSuite) TestUtterancesPublishedOneAfterAnotherAreNoOverlap() {
	s.logs = &logLines{}
	s.join(true)
	pairs := testutil.ToFloat64(speechOverlaps.WithLabelValues("stub"))

	s.voiceSends(2,
		chunkOf("turn-a", 160, 1000),
		chunkOf("turn-a", 160, 1000),
		tts.SynthesisComplete{SynthesisID: "turn-a"},
		chunkOf("tool-b", 160, 1000),
		chunkOf("tool-b", 160, 1000),
		tts.SynthesisComplete{SynthesisID: "tool-b"},
	)

	s.Empty(s.logs.records(overlapWarning))
	s.Equal(0.0, testutil.ToFloat64(speechOverlaps.WithLabelValues("stub"))-pairs)
}

func (s *AgentSuite) TestASentenceStartedWhileTheLastIsStillHeardIsNoOverlap() {
	s.logs = &logLines{}
	s.join(false)

	s.voiceSends(2,
		tts.SynthesisStarted{SynthesisID: "turn-a#0", Provider: "stub"},
		chunkOf("turn-a#0", 160, 1000),
		tts.SynthesisStarted{SynthesisID: "turn-a#1", Provider: "stub"},
		chunkOf("turn-a#0", 160, 1000),
		tts.SynthesisComplete{SynthesisID: "turn-a#0"},
		chunkOf("turn-a#1", 160, 1000),
		tts.SynthesisComplete{SynthesisID: "turn-a#1"},
	)

	s.Empty(s.logs.records(overlapWarning), "a sentence is not heard until it publishes audio")
}

func (s *AgentSuite) TestAnUtteranceTheAgentLeftBehindIsNoOverlap() {
	s.logs = &logLines{}
	s.join(true)
	seconds := testutil.ToFloat64(speechOverlapSeconds.WithLabelValues("stub"))

	s.voice.emitter.Send(chunkOf("turn-a", 160, 1000))
	s.eventually(func() bool { return len(s.edge.heard()) == 1 }, "the reply was never published")
	s.agent.mu.Lock()
	s.agent.abandoned["turn-a"] = struct{}{}
	s.agent.mu.Unlock()
	s.voiceSends(1,
		chunkOf("turn-a", 160, 1000),
		chunkOf("tool-b", 160, 1000),
		tts.SynthesisComplete{SynthesisID: "tool-b"},
	)

	s.Empty(s.logs.records(overlapWarning), "a barge-in dropped the rest of turn-a, so nothing was spliced")
	s.Equal(0.0, testutil.ToFloat64(speechOverlapSeconds.WithLabelValues("stub"))-seconds)
}

func (s *AgentSuite) TestAWarningNamesTheSessionAndTheCall() {
	s.logs = &logLines{}
	s.callID = "call-1"
	s.join(true)

	s.voiceSends(2, interleaved()...)

	warnings := s.logs.records(overlapWarning)
	s.Require().Len(warnings, 1)
	s.Equal("session-1", warnings[0]["session"])
	s.Equal("call-1", warnings[0]["call"])
}

// OverlapSuite covers what noticing an overlap costs on the path every frame takes.
type OverlapSuite struct {
	suite.Suite
	overlap *overlap
}

func TestOverlapSuite(t *testing.T) {
	suite.Run(t, new(OverlapSuite))
}

func (s *OverlapSuite) SetupTest() {
	s.overlap = newOverlap("stub", func(string) bool { return false }, slog.New(slog.DiscardHandler))
}

func (s *OverlapSuite) TestAFrameOfTheOnlyUtterancePlayingAllocatesNothing() {
	s.overlap.published("turn-a", 10)

	s.Zero(testing.AllocsPerRun(100, func() { s.overlap.published("turn-a", 10) }))
}

func (s *OverlapSuite) TestAFrameOfAPairAlreadyWarnedAboutAllocatesNothing() {
	s.overlap.published("turn-a", 10)
	s.overlap.published("tool-b", 10)

	s.Zero(testing.AllocsPerRun(100, func() { s.overlap.published("tool-b", 10) }))
}

func (s *OverlapSuite) TestAFinishedUtteranceIsForgotten() {
	s.overlap.published("turn-a", 10)
	s.overlap.published("tool-b", 10)
	s.Require().NotEmpty(s.overlap.warned)

	s.overlap.finished("turn-a")
	s.overlap.finished("tool-b")

	s.Empty(s.overlap.playing)
	s.Empty(s.overlap.warned)
}
