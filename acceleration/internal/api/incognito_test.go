//go:build integration

package api

import (
	"context"
	"net/http"
	"testing"
	"time"
)

// IncognitoSuite holds an incognito session to its promise: no session row, no turns, no
// transcript, and so no call row or turn timings either, which name the session and what
// it was told to be.
type IncognitoSuite struct {
	RouterSuite
}

func TestIncognitoSuite(t *testing.T) {
	runSuite(t, new(IncognitoSuite))
}

func (s *IncognitoSuite) SetupTest() {
	s.useFixture("standard")
}

// TestARecordedSessionLeavesItsCallAndTurns is the control: without it, the test below
// would pass against a router that records nothing for anyone.
func (s *IncognitoSuite) TestARecordedSessionLeavesItsCallAndTurns() {
	started := time.Now().UTC().Add(-time.Minute)
	opened := s.converse(false)

	ctx := context.Background()
	s.Require().Eventually(func() bool {
		_, err := s.store.Call(ctx, s.customerID(), opened.Id)
		return err == nil
	}, settleFor, 20*time.Millisecond, "the session's call was never recorded")
	s.Require().Eventually(func() bool {
		turns, err := s.store.CallTurns(ctx, s.customerID(), opened.AgentId, started, nil)
		return err == nil && len(turns) > 0
	}, settleFor, 20*time.Millisecond, "the session's turn was never timed")
}

func (s *IncognitoSuite) TestAnIncognitoSessionLeavesNoCallAndNoTurns() {
	started := time.Now().UTC().Add(-time.Minute)
	hidden := s.converse(true)
	// Calls are written in the order they began, so once a later session's row is there,
	// the incognito one's would have been too.
	recorded := s.converse(false)

	ctx := context.Background()
	s.Require().Eventually(func() bool {
		_, err := s.store.Call(ctx, s.customerID(), recorded.Id)
		return err == nil
	}, settleFor, 20*time.Millisecond, "the recorded session's call was never written")

	_, err := s.store.Call(ctx, s.customerID(), hidden.Id)
	s.ErrorContains(err, "there is no call", "an incognito session has no call row")
	s.Never(func() bool {
		turns, err := s.store.CallTurns(ctx, s.customerID(), hidden.AgentId, started, nil)
		return err != nil || len(turns) > 0
	}, time.Second, 50*time.Millisecond, "an incognito session has no turn timings")
}

// converse opens a text session, has one exchange in it and stops it, so whatever the
// session writes when it closes has been written.
func (s *IncognitoSuite) converse(incognito bool) Session {
	request := textSession(nil)
	request.Incognito = pointerTo(incognito)
	opened := s.serverClient.createSession(request)

	watching := s.serverClient.opens("/v1/agents/sessions/" + opened.Id + "/events")
	s.Require().Equal(http.StatusAccepted, s.serverClient.do(http.MethodPost,
		"/v1/agents/sessions/"+opened.Id+"/responses",
		CreateResponseRequest{Text: "What is the capital of France?"}, nil))
	s.await(watching, "responded")

	s.serverClient.stopSession(opened.Id)
	return opened
}
