//go:build integration

package elevenlabs

import (
	"testing"

	"github.com/stretchr/testify/require"
	"github.com/stretchr/testify/suite"

	"github.com/GetStream/Vision-Agents/acceleration/internal/tts"
	"github.com/GetStream/Vision-Agents/acceleration/internal/tts/ttssuite"
)

type ElevenLabsIntegrationSuite struct {
	ttssuite.Suite
}

func TestElevenLabsIntegrationSuite(t *testing.T) {
	suite.Run(t, &ElevenLabsIntegrationSuite{Suite: ttssuite.Suite{
		New: func() ttssuite.Provider {
			provider, err := New(Options{})
			require.NoError(t, err)
			return provider
		},
		Requires:      []string{"ELEVENLABS_API_KEY"},
		Interruptible: true,
	}})
}

// DialogueIntegrationSuite holds the v3 socket to the same contract as the rest. It is a
// different endpoint and protocol, so the unit tests, which answer with a server of our
// own, agree with themselves about it; only this one finds out whether ElevenLabs agrees,
// and whether the voice the provider falls back to is one the account may actually
// generate with. A voice the account cannot use is not refused on connect: the socket
// opens, and the error only arrives once there is enough text to generate from.
type DialogueIntegrationSuite struct {
	ttssuite.Suite
}

func TestDialogueIntegrationSuite(t *testing.T) {
	suite.Run(t, &DialogueIntegrationSuite{Suite: ttssuite.Suite{
		New: func() ttssuite.Provider {
			provider, err := NewDialogue(Options{})
			require.NoError(t, err)
			return provider
		},
		Requires:      []string{"ELEVENLABS_API_KEY"},
		Interruptible: true,
	}})
}

// TestADirectionIsAccepted covers the point of the v3 model, and what the agent's
// instructions ask a model writing for it to produce. A bracketed direction is not
// rejected, and the words around it are still said.
func (s *DialogueIntegrationSuite) TestADirectionIsAccepted() {
	provider := s.Started()
	defer s.Hangup(provider)

	directed := "[laughs] " + ttssuite.Sentence
	complete, chunks := s.Say(provider, tts.Request{ID: "u1", Text: directed, Final: true})

	s.Require().NotEmpty(chunks, "a direction should be performed, not refused")
	s.Greater(complete.AudioDurationMs, 1_000.0)
	s.False(complete.Interrupted)
}
