//go:build integration

package assemblyai

import (
	"context"
	"strings"
	"testing"
	"time"

	"github.com/stretchr/testify/require"
	"github.com/stretchr/testify/suite"

	"github.com/GetStream/Vision-Agents/acceleration/internal/stt"
	"github.com/GetStream/Vision-Agents/acceleration/internal/stt/sttsuite"
)

// AssemblyAIIntegrationSuite inherits what every provider owes a call from sttsuite. It is
// fed 20 ms at a time, which is how a call arrives and less than the 50 ms the server
// takes in one frame, so the gathering of audio is held to account on the real socket.
// Terminate settles the turn in progress, so ending the audio stream keeps the tail. The
// model ends a turn wherever the words read as finished, which can be mid-sentence, so the
// clock-time clip holds it to returning every word across however many turns it makes.
type AssemblyAIIntegrationSuite struct {
	sttsuite.Suite
}

func TestAssemblyAIIntegrationSuite(t *testing.T) {
	suite.Run(t, &AssemblyAIIntegrationSuite{Suite: sttsuite.Suite{
		New: func() stt.STT {
			provider, err := New(Options{})
			require.NoError(t, err)
			return provider
		},
		Requires:       []string{"ASSEMBLYAI_API_KEY"},
		ChunkMs:        20,
		SettlesOnClose: true,
		ClockFixture:   "saturday_seven_thirty.wav",
	}})
}

// TestAMisspeltModelIsRefusedRatherThanServedByTheDefault is the server's habit of
// accepting a session it cannot serve as asked, which only Start can catch.
func (s *AssemblyAIIntegrationSuite) TestAMisspeltModelIsRefusedRatherThanServedByTheDefault() {
	provider, err := New(Options{Model: "universal-3-6-pr"})
	s.Require().NoError(err)

	ctx, cancel := context.WithTimeout(context.Background(), 30*time.Second)
	defer cancel()
	s.ErrorContains(provider.Start(ctx), "speech_model")
}

// TestEverythingThisPackageSendsIsAccepted opens a session with each setting the query
// string can carry. The server refuses a malformed one outright, so a session that opens
// on the preset that was asked for and still hears the caller is one that read them all.
// A 160 ms silence ends a turn at the fixture's pause after "a map.", so every turn counts.
func (s *AssemblyAIIntegrationSuite) TestEverythingThisPackageSendsIsAccepted() {
	provider, err := New(Options{
		Keyterms:         []string{"Mia", "hues of gold"},
		LanguageHints:    []string{"en", "es"},
		Mode:             ModeMinLatency,
		MinTurnSilenceMs: 160,
		MaxTurnSilenceMs: 1200,
	})
	s.Require().NoError(err)
	s.Start(provider)

	collected := make(chan []string, 1)
	go func() {
		var finals []string
		for event := range provider.Events() {
			transcript, ok := event.(stt.Transcript)
			if ok && transcript.Final() {
				finals = append(finals, transcript.Text)
			}
		}
		collected <- finals
	}()

	s.Speak(provider, 0)
	s.Quiet(provider)
	s.Hangup(provider)

	select {
	case finals := <-collected:
		s.RequireAccurate(strings.Join(finals, " "))
	case <-time.After(30 * time.Second):
		s.FailNow("the event channel was not closed")
	}
}
