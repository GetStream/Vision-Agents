//go:build integration

// Package stssuite is what every speech-to-speech provider is held to on a real call: the
// model hears the caller, answers in its own voice, writes down what it heard and said, can
// be cut off, and calls the tools it was given.
//
// A provider suite embeds Suite, says how to build a provider and what it may be held to,
// and inherits those tests. Anything only one provider does stays in that provider's own
// file. What a model cannot do by its own Capabilities is skipped rather than failed, since
// refusing is the honest answer and the contract says so.
package stssuite

import (
	"context"
	"os"
	"time"

	"github.com/stretchr/testify/suite"

	"github.com/GetStream/Vision-Agents/acceleration/internal/llm"
	"github.com/GetStream/Vision-Agents/acceleration/internal/sts"
	"github.com/GetStream/Vision-Agents/acceleration/internal/stt"
	"github.com/GetStream/Vision-Agents/acceleration/internal/testaudio"
	// The providers need their credentials, which live in the repository's .env rather
	// than in the environment an editor happens to run a test with.
	_ "github.com/GetStream/Vision-Agents/acceleration/internal/testenv"
)

// speaker is who the audio in these tests is attributed to.
var speaker = sts.Participant{ID: "test-user", UserID: "test-user"}

const (
	// defaultFixture is the clip the transcriber suite uses for the same purpose.
	defaultFixture = "mia.mp3"
	// defaultChunkMs is the size of the pieces a call arrives in.
	defaultChunkMs = 100
	// defaultSilenceMs is the quiet after the speech, which is what tells the model the
	// turn is over.
	defaultSilenceMs = 2000
	// defaultMinAccuracy is how much of the sentence has to be written down.
	defaultMinAccuracy = 0.9
	// defaultMaxTimeToFirstByte is the longest a caller should wait for the first audio
	// of a reply, in milliseconds. Loose enough that only a real regression trips it.
	defaultMaxTimeToFirstByte = 6_000.0
	// defaultTimeout bounds a session and the wait for the events it produces.
	defaultTimeout = 2 * time.Minute
)

// Ask is what a test wants the session opened with.
type Ask struct {
	Instructions string
	Tools        []llm.Tool
}

// Suite is the shared behaviour. The fields are set where the suite is constructed rather
// than in a SetupSuite of the provider's own, which would shadow this one.
type Suite struct {
	suite.Suite

	// New builds an unstarted provider for the ask, with both transcripts turned on.
	New func(ask Ask) sts.STS
	// Requires are the environment variables without which the provider cannot be
	// reached, and whose absence skips rather than fails.
	Requires []string

	// Fixture is the clip that gets spoken, and ChunkMs and SilenceMs the pace it and
	// the quiet after it are delivered at.
	Fixture   string
	ChunkMs   int
	SilenceMs int
	// MinAccuracy is how much of the sentence the model has to have heard.
	MinAccuracy float64
	// MaxTimeToFirstByte is how long a caller may wait to hear the reply begin.
	MaxTimeToFirstByte float64
	// Timeout is the context a session is opened with and the longest a test waits.
	Timeout time.Duration

	// Audio is the fixture and Reference is what is said in it. SetupSuite loads both.
	Audio     sts.PcmData
	Reference string
}

func (s *Suite) SetupSuite() {
	s.Require().NotNil(s.New, "a provider suite has to say how to build its provider")
	for _, name := range s.Requires {
		if os.Getenv(name) == "" {
			s.T().Skipf("%s not set", name)
		}
	}
	if !testaudio.HasFFmpeg() {
		s.T().Skip("ffmpeg not available to decode the audio fixture")
	}
	if s.Fixture == "" {
		s.Fixture = defaultFixture
	}
	if s.ChunkMs == 0 {
		s.ChunkMs = defaultChunkMs
	}
	if s.SilenceMs == 0 {
		s.SilenceMs = defaultSilenceMs
	}
	if s.MinAccuracy == 0 {
		s.MinAccuracy = defaultMinAccuracy
	}
	if s.MaxTimeToFirstByte == 0 {
		s.MaxTimeToFirstByte = defaultMaxTimeToFirstByte
	}
	if s.Timeout == 0 {
		s.Timeout = defaultTimeout
	}

	audio, err := testaudio.Load16kMono(s.Fixture)
	s.Require().NoError(err)
	s.Audio = audio

	reference, err := testaudio.Reference(s.Fixture)
	s.Require().NoError(err)
	s.Reference = reference
}

// Started builds and opens a provider for the ask.
func (s *Suite) Started(ask Ask) sts.STS {
	provider := s.New(ask)
	s.Start(provider)
	return provider
}

// Start opens a provider the caller built, so a test can use options of its own and still
// get the suite's timeout.
func (s *Suite) Start(provider sts.STS) {
	ctx, cancel := context.WithTimeout(context.Background(), s.Timeout)
	s.T().Cleanup(cancel)
	s.Require().NoError(provider.Start(ctx))
}

// Hangup ends the session the way the end of a call would.
func (s *Suite) Hangup(provider sts.STS) {
	s.Require().NoError(provider.Close())
}

// Speak streams the fixture at the pace a call delivers it, then the silence that tells the
// model the turn is over. Sending it any faster would have the model see the whole clip at
// once, which is not how it behaves on a call.
func (s *Suite) Speak(provider sts.STS) {
	s.stream(provider, testaudio.Chunks(s.Audio, s.ChunkMs))
	s.stream(provider, testaudio.Chunks(testaudio.Silence(s.SilenceMs), s.ChunkMs))
}

func (s *Suite) stream(provider sts.STS, chunks []stt.PcmData) {
	for _, chunk := range chunks {
		s.Require().NoError(provider.ProcessAudio(chunk, speaker))
		time.Sleep(time.Duration(s.ChunkMs) * time.Millisecond)
	}
}

// Collect reads events until the predicate is satisfied. A fatal provider error fails the
// test straight away, so a rejected session reports what went wrong instead of timing out.
func (s *Suite) Collect(provider sts.STS, until func(sts.Event) bool) []sts.Event {
	var events []sts.Event
	deadline := time.After(s.Timeout)

	for {
		select {
		case event, open := <-provider.Events():
			if !open {
				return events
			}
			events = append(events, event)
			if failure, failed := event.(sts.Error); failed && failure.Fatal {
				s.FailNowf("provider error", "%v", failure.Err)
			}
			if until(event) {
				return events
			}
		case <-deadline:
			s.FailNow("timed out waiting for the model")
			return events
		}
	}
}

// Reply is everything the model did in answering one turn.
type Reply struct {
	Complete  sts.ResponseComplete
	Chunks    []sts.AudioChunk
	Heard     []sts.InputTranscript
	Said      []sts.OutputTranscript
	ToolCalls []sts.ToolCall
	Started   []sts.ResponseStarted
	AllEvents []sts.Event
}

// Settled waits for the reply in flight to finish and returns what happened along the way.
func (s *Suite) Settled(provider sts.STS) Reply {
	events := s.Collect(provider, func(event sts.Event) bool {
		_, done := event.(sts.ResponseComplete)
		return done
	})
	return replyOf(events)
}

// Spoken waits for the next reply that carried audio. A reply that only asked for a tool
// settles without saying anything, and the one after the answer is the one that speaks.
func (s *Suite) Spoken(provider sts.STS) Reply {
	var spoke bool
	events := s.Collect(provider, func(event sts.Event) bool {
		switch event.(type) {
		case sts.AudioChunk:
			spoke = true
		case sts.ResponseComplete:
			return spoke
		}
		return false
	})
	return replyOf(events)
}

// Asked waits until the model asks for a tool, which on some vendors ends the turn and on
// others does not.
func (s *Suite) Asked(provider sts.STS) Reply {
	events := s.Collect(provider, func(event sts.Event) bool {
		_, asked := event.(sts.ToolCall)
		return asked
	})
	return replyOf(events)
}

func replyOf(events []sts.Event) Reply {
	reply := Reply{AllEvents: events}
	for _, event := range events {
		switch typed := event.(type) {
		case sts.ResponseComplete:
			reply.Complete = typed
		case sts.AudioChunk:
			reply.Chunks = append(reply.Chunks, typed)
		case sts.InputTranscript:
			reply.Heard = append(reply.Heard, typed)
		case sts.OutputTranscript:
			reply.Said = append(reply.Said, typed)
		case sts.ToolCall:
			reply.ToolCalls = append(reply.ToolCalls, typed)
		case sts.ResponseStarted:
			reply.Started = append(reply.Started, typed)
		}
	}
	return reply
}

// HeardText is what the model wrote down of the caller: the last settled transcript when
// there is one, the last restatement otherwise, and the deltas joined up failing both.
func (r Reply) HeardText() string {
	var settled, restated, deltas string
	for _, heard := range r.Heard {
		switch heard.Mode {
		case stt.ModeFinal:
			settled = heard.Text
		case stt.ModeReplacement:
			restated = heard.Text
		default:
			deltas += heard.Text
		}
	}
	if settled != "" {
		return settled
	}
	if restated != "" {
		return restated
	}
	return deltas
}

// SaidText is what the model wrote down of itself.
func (r Reply) SaidText() string {
	var deltas string
	for _, said := range r.Said {
		switch said.Mode {
		case stt.ModeFinal:
			return said.Text
		case stt.ModeReplacement:
			deltas = said.Text
		default:
			deltas += said.Text
		}
	}
	return deltas
}

// AudioMs is how much speech came back, by the chunks rather than by the model's own count.
func (r Reply) AudioMs() float64 {
	var total float64
	for _, chunk := range r.Chunks {
		total += chunk.Audio.DurationMs()
	}
	return total
}
