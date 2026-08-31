//go:build integration

// Package sttsuite is what every streaming speech-to-text provider is held to on a real
// call: the words come back, they come back while the caller is still talking, and the
// turn settles soon enough afterwards that a conversation does not drag.
//
// A provider suite embeds Suite, says how to build a provider and what it may be held to,
// and inherits those tests. Anything only one provider does, such as Deepgram's ping
// deadline or Gemini's transcription modes, stays in that provider's own file.
package sttsuite

import (
	"context"
	"os"
	"strings"
	"time"

	"github.com/stretchr/testify/suite"

	"github.com/GetStream/Vision-Agents/acceleration/internal/stt"
	"github.com/GetStream/Vision-Agents/acceleration/internal/testaudio"
	// The providers need their credentials, which live in the repository's .env rather
	// than in the environment an editor happens to run a test with.
	_ "github.com/GetStream/Vision-Agents/acceleration/internal/testenv"
)

// speaker is who the audio in these tests is attributed to.
var speaker = stt.Participant{ID: "test-user", UserID: "test-user"}

const (
	// defaultFixture is the clip the Python suite uses for the same purpose.
	defaultFixture = "mia.mp3"
	// defaultChunkMs is the size of the pieces a call arrives in.
	defaultChunkMs = 100
	// defaultSilenceMs is the quiet after the speech, which is what tells a server the
	// turn is over.
	defaultSilenceMs = 2000
	// defaultMinAccuracy is how much of the sentence has to come back. A provider that
	// gets one word in ten wrong is not one a conversation can be built on.
	defaultMinAccuracy = 0.9
	// defaultMaxSettle is the longest a caller should wait, after the last word, for the
	// turn to settle. It is not a target: it is the point past which a conversation
	// drags, and it is deliberately loose enough that only a real regression trips it.
	defaultMaxSettle = 2500 * time.Millisecond
	// defaultMaxToFirstWords is how long a caller may talk before anything they said is
	// written down. Tripping it means the hypotheses stopped arriving rather than that
	// they slowed down.
	defaultMaxToFirstWords = 2 * time.Second
	// defaultSessionTimeout is generous because it bounds the whole session rather than
	// the handshake: the longest test here idles for over a minute before it says
	// anything, and cutting a session short looks exactly like a provider dropping it.
	defaultSessionTimeout = 5 * time.Minute

	// patience is how long a test waits for a turn to settle before giving up on it.
	patience = 60 * time.Second
	// tailMs is how much of the fixture is spoken before hanging up mid-sentence, which
	// is enough to have said the opening words the tail is recognised by.
	tailMs = 4000
	// openingWords is how many of the reference's words the tail is checked against.
	openingWords = 3
)

// Suite is the shared behaviour. The fields are set where the suite is constructed rather
// than in a SetupSuite of the provider's own, which would shadow this one.
type Suite struct {
	suite.Suite

	// New builds an unstarted provider configured for an ordinary call.
	New func() stt.STT
	// Requires are the environment variables without which the provider cannot be
	// reached, and whose absence skips rather than fails.
	Requires []string

	// Fixture is the clip that gets spoken, and ChunkMs and SilenceMs the pace it and
	// the quiet after it are delivered at.
	Fixture   string
	ChunkMs   int
	SilenceMs int

	// MinAccuracy, MaxSettle and MaxToFirstWords are what a caller will put up with.
	MinAccuracy     float64
	MaxSettle       time.Duration
	MaxToFirstWords time.Duration
	// SessionTimeout is the context a session is opened with, which providers keep for
	// the life of the connection. A deployment that scales to zero needs a cold start's
	// worth on top.
	SessionTimeout time.Duration
	// SettlesOnClose marks a provider that transcribes what it is still holding when the
	// audio stream ends, rather than losing the tail of a call that was cut off.
	SettlesOnClose bool

	// Audio is the fixture and Reference is what is said in it. SetupSuite loads both.
	Audio     stt.PcmData
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
	s.fillInDefaults()

	audio, err := testaudio.Load16kMono(s.Fixture)
	s.Require().NoError(err)
	s.Audio = audio

	reference, err := testaudio.Reference(s.Fixture)
	s.Require().NoError(err)
	s.Reference = reference
}

// Started builds and opens an ordinary provider.
func (s *Suite) Started() stt.STT {
	provider := s.New()
	s.Start(provider)
	return provider
}

// Start opens a provider the caller built, so a test can use options of its own and
// still get the suite's timeout.
func (s *Suite) Start(provider stt.STT) {
	ctx, cancel := context.WithTimeout(context.Background(), s.SessionTimeout)
	s.T().Cleanup(cancel)
	s.Require().NoError(provider.Start(ctx))
}

// Hangup ends the session the way the end of a call would.
func (s *Suite) Hangup(provider stt.STT) {
	s.Require().NoError(provider.Close())
}

// Speak streams the fixture at the pace a call delivers it. Sending it any faster would
// have the model see the whole clip at once, which is not how it behaves on a call.
// chunks caps how much of it is spoken; zero speaks all of it.
func (s *Suite) Speak(provider stt.STT, chunks int) {
	spoken := testaudio.Chunks(s.Audio, s.ChunkMs)
	if chunks > 0 && chunks < len(spoken) {
		spoken = spoken[:chunks]
	}
	s.stream(provider, spoken)
}

// Quiet streams the silence that tells a provider the turn is over.
func (s *Suite) Quiet(provider stt.STT) {
	s.stream(provider, testaudio.Chunks(testaudio.Silence(s.SilenceMs), s.ChunkMs))
}

// Collect speaks the whole fixture, ends the turn, and returns everything the provider
// said along the way, up to and including the transcript it settled on.
func (s *Suite) Collect(provider stt.STT) []stt.Event {
	collected := make(chan []stt.Event, 1)
	go func() {
		var events []stt.Event
		for event := range provider.Events() {
			events = append(events, event)
			if transcript, ok := event.(stt.Transcript); ok && transcript.Final() {
				break
			}
		}
		collected <- events
	}()

	s.Speak(provider, 0)
	s.Quiet(provider)

	select {
	case events := <-collected:
		return events
	case <-time.After(patience):
		s.FailNow("timed out waiting for a settled transcript")
		return nil
	}
}

// SettledText speaks the fixture and returns the transcript the turn settles on.
func (s *Suite) SettledText(provider stt.STT) string {
	for _, event := range s.Collect(provider) {
		if transcript, ok := event.(stt.Transcript); ok && transcript.Final() {
			return transcript.Text
		}
	}
	s.FailNow("the turn never settled")
	return ""
}

// Measure times one turn the way the caller experiences it, on a session of its own.
func (s *Suite) Measure() testaudio.Timing {
	provider := s.Started()
	defer s.Hangup(provider)

	return s.MeasureOn(provider)
}

// MeasureOn times one turn on a session the caller opened, for a test comparing options
// the suite knows nothing about. Closing it is left to the caller.
func (s *Suite) MeasureOn(provider stt.STT) testaudio.Timing {
	timing, err := testaudio.Measure(provider, s.Audio, s.ChunkMs, s.SilenceMs, patience)
	s.Require().NoError(err)
	s.T().Logf("spoke for %.0fms, first words after %.0fms, settled %.0fms after the last "+
		"word, %d transcripts while still speaking: %q",
		ms(timing.SpokeFor), ms(timing.ToFirstWords), ms(timing.ToSettle),
		timing.WhileSpeaking, timing.Text)
	return timing
}

// Accuracy scores a transcript against what the fixture actually says.
func (s *Suite) Accuracy(heard string) float64 {
	return testaudio.Accuracy(s.Reference, heard)
}

// RequireAccurate fails unless the transcript says what the caller said.
func (s *Suite) RequireAccurate(heard string) {
	score := s.Accuracy(heard)
	s.T().Logf("%.0f%% of the words came back: %q", score*100, heard)
	s.GreaterOrEqualf(score, s.MinAccuracy,
		"the transcript should be most of what was said\n  said:  %q\n  heard: %q",
		s.Reference, heard)
}

func (s *Suite) stream(provider stt.STT, chunks []stt.PcmData) {
	started := time.Now()
	for i, chunk := range chunks {
		if wait := time.Until(started.Add(time.Duration(i*s.ChunkMs) * time.Millisecond)); wait > 0 {
			time.Sleep(wait)
		}
		s.Require().NoError(provider.ProcessAudio(chunk, speaker))
	}
}

// opening is the first few words of the fixture, which is what a transcript of the tail
// of a call that was cut off should still contain.
func (s *Suite) opening() string {
	words := strings.Fields(strings.ToLower(s.Reference))
	if len(words) > openingWords {
		words = words[:openingWords]
	}
	return strings.Join(words, " ")
}

func (s *Suite) fillInDefaults() {
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
	if s.MaxSettle == 0 {
		s.MaxSettle = defaultMaxSettle
	}
	if s.MaxToFirstWords == 0 {
		s.MaxToFirstWords = defaultMaxToFirstWords
	}
	if s.SessionTimeout == 0 {
		s.SessionTimeout = defaultSessionTimeout
	}
}

func ms(d time.Duration) float64 { return d.Seconds() * 1000 }
