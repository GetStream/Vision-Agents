//go:build integration

package deepgram

import (
	"context"
	"os"
	"strings"
	"testing"

	"github.com/stretchr/testify/suite"

	"github.com/GetStream/Vision-Agents/acceleration/internal/stt"
	"github.com/GetStream/Vision-Agents/acceleration/internal/testaudio"
	// The batch endpoint needs its credentials, which live in the repository's .env
	// rather than in the environment an editor happens to run a test with.
	_ "github.com/GetStream/Vision-Agents/acceleration/internal/testenv"
)

// PrerecordedIntegrationSuite is the batch half of this package against the real endpoint.
//
// It exists mainly to settle what nova-3 declares in router.yaml. Deepgram's own pages
// disagree about which of their models accept profanity_filter and filler_words - the
// feature pages, the Flux-versus-Nova comparison table and their changelog each say
// something different - and a term is only declared if the request really carries it.
type PrerecordedIntegrationSuite struct {
	suite.Suite

	ctx      context.Context
	audio    []byte
	expected string
}

func TestPrerecordedIntegrationSuite(t *testing.T) {
	suite.Run(t, new(PrerecordedIntegrationSuite))
}

func (s *PrerecordedIntegrationSuite) SetupSuite() {
	if os.Getenv("DEEPGRAM_API_KEY") == "" {
		s.T().Skip("DEEPGRAM_API_KEY not set")
	}

	path, err := testaudio.Asset("mia.mp3")
	s.Require().NoError(err)
	audio, err := os.ReadFile(path)
	s.Require().NoError(err)
	s.audio = audio

	expected, err := testaudio.Reference("mia.mp3")
	s.Require().NoError(err)
	s.expected = expected
	s.ctx = context.Background()
}

// transcribe runs one recording through nova-3 and returns what came back.
func (s *PrerecordedIntegrationSuite) transcribe(recording stt.Recording) stt.Transcription {
	transcriber, err := NewPrerecorded(PrerecordedOptions{})
	s.Require().NoError(err)

	recording.Audio = s.audio
	transcription, err := transcriber.Transcribe(s.ctx, recording)
	s.Require().NoError(err)
	return transcription
}

func (s *PrerecordedIntegrationSuite) TestNova3AcceptsAProfanityFilter() {
	// The one thing Deepgram's pages agree on for this model. Nothing in the fixture is
	// masked, so what is under test is that the parameter is accepted and the transcript
	// survives it - a model that refused it would fail the request outright.
	transcription := s.transcribe(stt.Recording{ProfanityFilter: true})

	s.GreaterOrEqual(testaudio.Accuracy(s.expected, transcription.Text), 0.9)
}

func (s *PrerecordedIntegrationSuite) TestNova3AcceptsFillerWords() {
	// Whether it honours them is a different question, and the reason nova-3 does not
	// declare verbatim: the fixture has no ums in it to check against, and Deepgram's
	// comparison table says this model strips them whatever the parameter says. This
	// asserts only what can be asserted, which is that asking does not break the request.
	transcription := s.transcribe(stt.Recording{FillerWords: true})

	s.GreaterOrEqual(testaudio.Accuracy(s.expected, transcription.Text), 0.9)
}

func (s *PrerecordedIntegrationSuite) TestNova3AcceptsAModelImprovementOptOut() {
	// This is the request that makes the no-training half of nova-3's data policy true,
	// so a build where it is silently rejected is a build whose config is lying.
	transcriber, err := NewPrerecorded(PrerecordedOptions{MipOptOut: true})
	s.Require().NoError(err)

	transcription, err := transcriber.Transcribe(s.ctx, stt.Recording{Audio: s.audio})
	s.Require().NoError(err)

	s.NotEmpty(strings.TrimSpace(transcription.Text))
}
