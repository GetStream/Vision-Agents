//go:build integration

package agent

import (
	"context"
	"os"
	"testing"
	"time"

	"github.com/stretchr/testify/suite"
)

type EOTLiveSuite struct{ suite.Suite }

func TestEOTLiveSuite(t *testing.T) { suite.Run(t, new(EOTLiveSuite)) }

func (s *EOTLiveSuite) TestAuthenticatedEndpointScoresAudio() {
	endpoint, tokenFile := os.Getenv("EOT_LIVE_URL"), os.Getenv("EOT_LIVE_TOKEN_FILE")
	if endpoint == "" || tokenFile == "" {
		s.T().Skip("set EOT_LIVE_URL and EOT_LIVE_TOKEN_FILE for the live EOT test")
	}
	// Silence verifies the service contract without assuming a calibrated score.
	// Supply a raw mono PCM16LE file to try recorded speech instead.
	pcm := make([]byte, eotSampleRate*2)
	if path := os.Getenv("EOT_LIVE_PCM_PATH"); path != "" {
		var err error
		pcm, err = os.ReadFile(path)
		s.Require().NoError(err)
	}
	client, err := NewEOTClient(endpoint, tokenFile)
	s.Require().NoError(err)
	ctx, cancel := context.WithTimeout(context.Background(), 5*time.Second)
	defer cancel()

	score, err := client.Score(ctx, "eot-live", pcm)

	s.Require().NoError(err)
	s.Equal(len(pcm)/2, score.Samples)
	s.True(validProbability(score.Probability))
}
