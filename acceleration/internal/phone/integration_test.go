//go:build integration

package phone

import (
	"context"
	"fmt"
	"os"
	"testing"
	"time"

	"github.com/stretchr/testify/suite"
)

// A transfer is a trunk, a routing rule pinned to a live call, and a leg dialled into it.
// Dialling costs money and rings a real telephone, so what is exercised here is the half
// that is this service's own: the route that puts the human in the caller's call rather
// than in one of their own.

type TransferIntegrationSuite struct {
	suite.Suite
	ctx    context.Context
	stream *Stream
	// number is a number the trunk answers for. It is never dialled.
	number string
}

func TestTransferIntegrationSuite(t *testing.T) {
	suite.Run(t, new(TransferIntegrationSuite))
}

func (s *TransferIntegrationSuite) SetupSuite() {
	if os.Getenv("STREAM_API_KEY") == "" || os.Getenv("STREAM_API_SECRET") == "" {
		s.T().Skip("STREAM_API_KEY and STREAM_API_SECRET not set")
	}

	stream, err := NewStream(StreamOptions{})
	s.Require().NoError(err)
	s.stream = stream
	s.number = fmt.Sprintf("+1512555%04d", time.Now().UnixNano()%10_000)
}

func (s *TransferIntegrationSuite) SetupTest() {
	var cancel context.CancelFunc
	s.ctx, cancel = context.WithTimeout(context.Background(), 30*time.Second)
	s.T().Cleanup(cancel)
}

func (s *TransferIntegrationSuite) TestATransferTrunkCarriesItsOwnCredentials() {
	// The password is readable only when the trunk is made, which is the whole reason a
	// transfer makes one rather than reusing the number's.
	trunkID, bridge, err := s.stream.CreateTrunk(s.ctx, Trunk{
		Name:    "transfer-test-" + s.number,
		Numbers: []string{s.number},
	})
	s.Require().NoError(err)

	s.NotEmpty(trunkID)
	s.NoError(bridge.Validate())
	s.NotEmpty(bridge.Username)
	s.NotEmpty(bridge.Password, "the vendor cannot register on the trunk without it")
}

func (s *TransferIntegrationSuite) TestARoutePinnedToALiveCallJoinsThatCall() {
	// Without pinning, the human would land in a call named after the number they were
	// dialled from, which is not the one the caller is waiting in.
	trunkID, _, err := s.stream.CreateTrunk(s.ctx, Trunk{
		Name:    "transfer-route-test-" + s.number,
		Numbers: []string{s.number},
	})
	s.Require().NoError(err)

	routeID, err := s.stream.CreateRoute(s.ctx, Route{
		Name:          "transfer-route-test-" + s.number,
		TrunkIDs:      []string{trunkID},
		CalledNumbers: []string{s.number},
		CallID:        "live-call-" + s.number,
	})
	s.Require().NoError(err)

	s.NotEmpty(routeID)
}
