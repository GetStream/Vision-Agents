//go:build integration

package api

import (
	"bytes"
	"context"
	"encoding/json"
	"log/slog"
	"net/http"
	"net/http/httptest"
	"time"

	"google.golang.org/grpc/codes"
	"google.golang.org/grpc/status"

	"github.com/GetStream/Vision-Agents/acceleration/internal/auth"
	"github.com/GetStream/Vision-Agents/acceleration/internal/node"
)

// forwardDelivery hands a scheduled delivery for a session on other to it the way a peer
// does, presenting credential, and returns the text it carried and what other answered.
func (s *ScheduledTasksSuite) forwardDelivery(other *httptest.Server, credential []byte) (string, *node.ForwardResponse, error) {
	config := s.agentWithScheduling(nil)
	opened := s.data.createUser().on(other).createSession(CreateSessionRequest{ConfigId: &config})

	text := "peer-delivery-" + s.utils.uuid()
	body, err := json.Marshal(map[string]any{
		"customer_id": s.customerID(),
		"session_id":  opened.Id,
		"caller_kind": "server",
		"task_id":     "peer-task",
		"run_id":      "peer-run",
		"command_id":  "st_peer",
		"text":        text,
		"due":         time.Now(),
	})
	s.Require().NoError(err)
	request, err := http.NewRequest(http.MethodPost, scheduledPeerPath, bytes.NewReader(body))
	s.Require().NoError(err)

	ctx, cancel := context.WithTimeout(context.Background(), 30*time.Second)
	defer cancel()
	answer, err := node.NewForwarder(nil, credential).Ask(ctx, other.Listener.Addr().String(), request, body)
	return text, answer, err
}

// suiteCredential is what a node holding the suite's keyring presents to its peers.
func (s *ScheduledTasksSuite) suiteCredential() []byte {
	credential, _ := s.sealer.Derive(node.CredentialPurpose)
	return credential
}

// refused checks that a forwarded delivery was turned away and never reached the session.
func (s *ScheduledTasksSuite) refused(text string, err error) {
	s.Equal(codes.Unauthenticated, status.Code(err), "error: %v", err)
	s.Never(func() bool { return s.model().heard(text) }, time.Second, 50*time.Millisecond,
		"the refused delivery reached the session")
}

// delivers checks that a forwarded delivery was answered and reached the session.
func (s *ScheduledTasksSuite) delivers(text string, answer *node.ForwardResponse, err error) {
	s.Require().NoError(err)
	s.Equal(int32(http.StatusOK), answer.GetStatus(), "body: %s", answer.GetBody())
	s.Eventually(func() bool { return s.model().heard(text) }, 10*time.Second, 50*time.Millisecond,
		"the delivery never reached the session")
}

func (s *ScheduledTasksSuite) TestAForwardWithoutTheDeploymentsCredentialIsRefused() {
	other, _ := s.otherRouter(nil)

	text, _, err := s.forwardDelivery(other, nil)
	s.refused(text, err)
}

func (s *ScheduledTasksSuite) TestAForwardWithAnotherDeploymentsCredentialIsRefused() {
	other, _ := s.otherRouter(nil)
	another, err := auth.NewSealer("another deployment")
	s.Require().NoError(err)
	credential, _ := another.Derive(node.CredentialPurpose)

	text, _, err := s.forwardDelivery(other, credential)
	s.refused(text, err)
}

func (s *ScheduledTasksSuite) TestAPeerHoldingTheKeyringDelivers() {
	other, _ := s.otherRouter(nil)

	s.delivers(s.forwardDelivery(other, s.suiteCredential()))
}

func (s *ScheduledTasksSuite) TestAPeerStillOnTheOlderKeyDeliversToARotatedNode() {
	rotated, err := auth.NewSealerWithKeyring(2, map[int]string{1: suiteKEK, 2: "the suite's next key"})
	s.Require().NoError(err)
	other, _ := s.otherRouterSealedBy(nil, rotated, slog.New(slog.DiscardHandler))

	s.delivers(s.forwardDelivery(other, s.suiteCredential()))
}

func (s *ScheduledTasksSuite) TestANodeWithNoKeyringRefusesEveryForward() {
	var logged bytes.Buffer
	other, _ := s.otherRouterSealedBy(nil, nil, slog.New(slog.NewTextHandler(&logged, nil)))
	s.Contains(logged.String(), "setting=ROUTER_AUTH_KEK", "the node never said which setting it lacks")

	text, _, err := s.forwardDelivery(other, s.suiteCredential())
	s.refused(text, err)
	text, _, err = s.forwardDelivery(other, nil)
	s.refused(text, err)
}
