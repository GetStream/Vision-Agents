//go:build integration

package fish

import (
	"strings"
	"testing"
	"time"

	"github.com/stretchr/testify/require"
	"github.com/stretchr/testify/suite"

	"github.com/GetStream/Vision-Agents/acceleration/internal/stt"
	"github.com/GetStream/Vision-Agents/acceleration/internal/stt/sttsuite"
)

// FishIntegrationSuite inherits what every provider owes a call from sttsuite. It is fed
// 20 ms at a time, which is how a call arrives. audio_done settles the turn in progress,
// so ending the audio stream keeps the tail.
type FishIntegrationSuite struct {
	sttsuite.Suite
}

func TestFishIntegrationSuite(t *testing.T) {
	suite.Run(t, &FishIntegrationSuite{Suite: sttsuite.Suite{
		New: func() stt.STT {
			provider, err := New(Options{})
			require.NoError(t, err)
			return provider
		},
		Requires:       []string{"FISH_API_KEY"},
		ChunkMs:        20,
		SettlesOnClose: true,
		ClockFixture:   "saturday_seven_thirty.wav",
	}})
}

// TestEverythingThisPackageSendsIsAccepted opens a session with each setting the config
// can carry, and retunes it once open. The server refuses an unknown or out-of-range
// field by closing the session, and a refused configure by saying so, so a session that
// reports no error and still hears the caller is one that read them all.
func (s *FishIntegrationSuite) TestEverythingThisPackageSendsIsAccepted() {
	provider, err := New(Options{
		Keyterms:         []string{"Mia", "hues of gold"},
		Mode:             ModeVAD,
		SilenceTimeoutMs: 900,
		Eager:            true,
	})
	s.Require().NoError(err)
	s.Start(provider)
	s.Require().NoError(provider.SetKeyterms([]string{"Mia", "treasures"}))

	collected := make(chan []string, 1)
	go func() {
		var finals []string
		for event := range provider.Events() {
			switch typed := event.(type) {
			case stt.Transcript:
				if typed.Final() {
					finals = append(finals, typed.Text)
				}
			case stt.Error:
				s.Failf("provider error", "%v", typed.Err)
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

// TestAQuietCallerKeepsTheirTranscriber is the 30-second idle limit: the server closes a
// session it has heard nothing from for that long, and a muted caller sends no audio.
func (s *FishIntegrationSuite) TestAQuietCallerKeepsTheirTranscriber() {
	provider := s.Started()
	defer s.Hangup(provider)

	time.Sleep(35 * time.Second)

	s.RequireAccurate(s.SettledText(provider))
}
