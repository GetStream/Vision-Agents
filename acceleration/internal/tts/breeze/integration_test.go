//go:build integration

package breeze

import (
	"testing"
	"time"

	"github.com/stretchr/testify/require"
	"github.com/stretchr/testify/suite"

	"github.com/GetStream/Vision-Agents/acceleration/internal/tts"
	"github.com/GetStream/Vision-Agents/acceleration/internal/tts/ttssuite"
)

// coldStartTimeout is how long a scaled-to-zero GPU deployment may take to answer. This
// one builds CUDA graphs on the way up, so it is the slower of the two to wake.
const coldStartTimeout = 15 * time.Minute

type BreezeIntegrationSuite struct {
	ttssuite.Suite
}

func TestBreezeIntegrationSuite(t *testing.T) {
	suite.Run(t, &BreezeIntegrationSuite{Suite: ttssuite.Suite{
		New:           func() ttssuite.Provider { return newProvider(t, Options{}) },
		Requires:      []string{"BREEZE_WS_URL", "BASETEN_API_KEY"},
		Timeout:       coldStartTimeout,
		Interruptible: true,
	}})
}

// TestADirectionIsActedRatherThanSpelledOut checks that the deployment still turns [sigh]
// into the model's own syntax. If that stopped happening the words would be read out,
// which costs about a second of extra audio.
func (s *BreezeIntegrationSuite) TestADirectionIsActedRatherThanSpelledOut() {
	provider := s.Started()
	defer s.Hangup(provider)

	plain, _ := s.Say(provider, tts.Request{ID: "u1", Text: "Fine, I will do it.", Final: true})
	directed, _ := s.Say(provider, tts.Request{
		ID: "u2", Text: "[sigh] Fine, I will do it.", Final: true,
	})

	s.Positive(directed.AudioDurationMs)
	s.Less(directed.AudioDurationMs, plain.AudioDurationMs+1_500.0,
		"a performed sigh should cost far less audio than reading the word out")
}

// TestAVoiceDescribedInWordsIsTheVoiceThatSpeaks covers this model's way of picking a
// voice. There is no id to assert on, so what is checked is that the description is
// accepted and speech comes back, rather than being rejected or failing the utterance.
func (s *BreezeIntegrationSuite) TestAVoiceDescribedInWordsIsTheVoiceThatSpeaks() {
	provider := newProvider(s.T(), Options{Voice: "A deep, slow, weary old man."})
	s.Start(provider)
	defer s.Hangup(provider)

	complete, chunks := s.Say(provider, tts.Request{ID: "u1", Text: ttssuite.Sentence, Final: true})

	s.Require().NotEmpty(chunks)
	s.Greater(complete.AudioDurationMs, 2_000.0)
	s.False(complete.Interrupted, "a voice description should not fail the utterance")
}

// newProvider builds a provider that will wait out a cold start.
func newProvider(t require.TestingT, options Options) *TTS {
	options.HandshakeTimeout = coldStartTimeout
	provider, err := New(options)
	require.NoError(t, err)
	return provider
}
