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
	"github.com/GetStream/Vision-Agents/acceleration/internal/store"
)

// forwardDelivery hands a scheduled delivery for a stored task's session on other to it the
// way a peer does, presenting credential, and returns the text it carried and what other
// answered.
func (s *ScheduledTasksSuite) forwardDelivery(other *httptest.Server, credential []byte) (string, *node.ForwardResponse, error) {
	return s.forwardChanged(other, credential, nil)
}

// forwardChanged is forwardDelivery with change applied to the payload before it is sent.
func (s *ScheduledTasksSuite) forwardChanged(other *httptest.Server, credential []byte, change func(map[string]any)) (string, *node.ForwardResponse, error) {
	config := s.agentWithScheduling(nil)
	user := s.data.createUser()
	opened := user.on(other).createSession(CreateSessionRequest{ConfigId: &config})
	task := store.ScheduledTask{
		CustomerID: s.customerID(), ConfigID: config, UserID: user.userID, APIKey: s.app.key,
		CallerKind: "authenticated", SessionID: opened.Id, Title: "Stretch",
		Instructions: "Remind them to stretch.",
		StartLocal:   time.Now().UTC().Add(48 * time.Hour).Truncate(time.Second), Timezone: "UTC",
	}
	s.Require().NoError(s.store.CreateScheduledTask(context.Background(), &task, 25))

	text := "peer-delivery-" + s.utils.uuid()
	payload := map[string]any{
		"customer_id": s.customerID(),
		"session_id":  opened.Id,
		"user_id":     task.UserID,
		"caller_kind": task.CallerKind,
		"task_id":     task.ID,
		"run_id":      "peer-run",
		"command_id":  "st_peer",
		"text":        text,
		"due":         time.Now(),
	}
	if change != nil {
		change(payload)
	}
	body, err := json.Marshal(payload)
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

// refusedAsNoTask checks that an authenticated peer's delivery was answered as no such
// task and never reached the session.
func (s *ScheduledTasksSuite) refusedAsNoTask(text string, answer *node.ForwardResponse, err error) {
	s.Require().NoError(err)
	s.Equal(int32(http.StatusNotFound), answer.GetStatus(), "body: %s", answer.GetBody())
	s.Contains(string(answer.GetBody()), `"code":"`+codeScheduledTaskNotFound+`"`)
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

func (s *ScheduledTasksSuite) TestAPeerNamingAnotherUserThanTheStoredTaskIsRefused() {
	other, _ := s.otherRouter(nil)

	s.refusedAsNoTask(s.forwardChanged(other, s.suiteCredential(), func(payload map[string]any) {
		payload["user_id"] = s.utils.uuid()
	}))
}

func (s *ScheduledTasksSuite) TestAPeerNamingNoUserIsRefused() {
	other, _ := s.otherRouter(nil)

	s.refusedAsNoTask(s.forwardChanged(other, s.suiteCredential(), func(payload map[string]any) {
		delete(payload, "user_id")
	}))
}

func (s *ScheduledTasksSuite) TestAPeerNamingTheServerForAUsersTaskIsRefused() {
	other, _ := s.otherRouter(nil)

	s.refusedAsNoTask(s.forwardChanged(other, s.suiteCredential(), func(payload map[string]any) {
		payload["caller_kind"] = string(auth.KindServer)
	}))
}

func (s *ScheduledTasksSuite) TestAPeerNamingNoCallerKindIsRefused() {
	other, _ := s.otherRouter(nil)

	s.refusedAsNoTask(s.forwardChanged(other, s.suiteCredential(), func(payload map[string]any) {
		delete(payload, "caller_kind")
	}))
}

func (s *ScheduledTasksSuite) TestAPeerNamingATaskThatIsNotStoredIsRefused() {
	other, _ := s.otherRouter(nil)

	s.refusedAsNoTask(s.forwardChanged(other, s.suiteCredential(), func(payload map[string]any) {
		payload["task_id"] = s.utils.uuid()
	}))
}

func (s *ScheduledTasksSuite) TestAPeerNamingNoTaskIsRefused() {
	other, _ := s.otherRouter(nil)

	s.refusedAsNoTask(s.forwardChanged(other, s.suiteCredential(), func(payload map[string]any) {
		delete(payload, "task_id")
	}))
}

func (s *ScheduledTasksSuite) TestAPeerNamingNoCustomerIsRefused() {
	other, _ := s.otherRouter(nil)

	s.refusedAsNoTask(s.forwardChanged(other, s.suiteCredential(), func(payload map[string]any) {
		delete(payload, "customer_id")
	}))
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

func (s *ScheduledTasksSuite) TestTheUnansweredPeerPathIsRefusedWithoutAPeer() {
	body, err := json.Marshal(map[string]string{"customer_id": s.customerID(), "session_id": s.utils.uuid()})
	s.Require().NoError(err)

	status, answer := s.deliver(unansweredPeerPath, string(body), "")

	s.Equal(http.StatusNotFound, status, "body: %s", answer)
}
