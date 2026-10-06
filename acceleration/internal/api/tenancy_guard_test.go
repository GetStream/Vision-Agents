//go:build integration

package api

import (
	"context"
	"fmt"
	"net/http"
	"strconv"
	"testing"
	"time"

	"github.com/GetStream/Vision-Agents/acceleration/internal/conversation/chattest"
	"github.com/GetStream/Vision-Agents/acceleration/internal/dispatch"
	"github.com/GetStream/Vision-Agents/acceleration/internal/store"
)

// TenancyGuardSuite runs a router in app mode with the fallback off and two customers who
// registered apps of their own, drives what a customer does in Stream through every path the
// router offers, and watches what is written with which key. Stream here is one server that
// answers as the deployment's app or as either customer's by the key it is asked with.
type TenancyGuardSuite struct {
	RouterSuite
}

func TestTenancyGuardSuite(t *testing.T) {
	runSuite(t, new(TenancyGuardSuite))
}

func (s *TenancyGuardSuite) SetupSuite() {
	s.appMode, s.appRefuses = true, true
	s.RouterSuite.SetupSuite()
}

// tenant is one customer with an app of its own.
type tenant struct {
	app            testApp
	id             int64
	apiKey, secret string
}

// tenantWithApp makes a customer and registers its own app, which mints guests.
func (s *TenancyGuardSuite) tenantWithApp() tenant {
	id := streamAppID()
	made := tenant{app: s.numberedApp(id), id: id,
		apiKey: "key-" + strconv.FormatInt(id, 10), secret: "secret-" + s.utils.uuid()}
	s.chat.SetAppFor(made.apiKey, chattest.App{
		ID:           id,
		ChannelTypes: map[string]map[string][]string{"agent": {"channel_member": {"read-channel", "create-message"}}},
		CallTypes:    []string{"agent"},
	})
	s.useApp(made.app)
	var settings AppSettings
	s.Require().Equal(http.StatusOK, s.serverClient.do(http.MethodPut, "/v1/settings/app/stream/credentials",
		map[string]any{"keys": []keyInput{key(made.apiKey, made.secret)}, "expected_revision": 0, "allow_guests": true}, &settings))
	return made
}

// drive does in Stream, as one customer, everything the router does in Stream for one.
func (s *TenancyGuardSuite) drive(customer tenant) {
	s.useApp(customer.app)
	ctx := context.Background()

	opened := s.serverClient.createSession(textSession(nil))
	s.Require().NotNil(opened.ConversationId, "the conversation is kept")
	s.Require().Equal(http.StatusCreated, s.serverClient.do(http.MethodPost,
		"/v1/agents/sessions/"+opened.Id+"/fork", ForkSessionRequest{Title: pointerTo("again")}, nil))
	s.Require().Equal(http.StatusOK, s.serverClient.do(http.MethodPatch,
		"/v1/agents/sessions/"+opened.Id, map[string]any{"title": "renamed"}, nil))

	s.Require().Equal(http.StatusOK, s.serverClient.do(http.MethodPost, "/v1/agents/chat-token",
		map[string]any{"agent_id": "agent-" + s.utils.uuid()}, nil))

	var guest GuestUser
	s.Require().Equal(http.StatusCreated, s.anonymousClient.do(http.MethodPost, "/v1/agents/guests", nil, &guest))
	s.Require().Equal(http.StatusOK, s.serverClient.do(http.MethodPost, "/v1/agents/guests/claim",
		ClaimGuestRequest{GuestId: guest.Id, UserId: "account-" + s.utils.uuid()}, nil))

	config := store.AgentConfig{CustomerID: s.customerID(), Name: "config-" + s.utils.uuid(), Mode: "text"}
	s.Require().NoError(s.store.CreateAgentConfig(ctx, &config))
	body := fmt.Sprintf(`{"type": "message.new", "channel_id": %q, "channel_type": "agent", "created_at": %q,
  "channel_custom": {%q: %q}, "message": {"id": %q, "text": "hello", "user": {"id": "sam"}}}`,
		"chat-"+s.utils.uuid(), time.Now().UTC().Format(time.RFC3339Nano), ConfigField, config.ID, "message-"+s.utils.uuid())
	request := s.request("/v1/chat/hooks/stream/"+strconv.FormatInt(customer.id, 10), body, sign(body, customer.secret))
	response, err := s.server.Client().Do(request)
	s.Require().NoError(err)
	s.Require().NoError(response.Body.Close())
	s.Require().Equal(http.StatusOK, response.StatusCode)
}

func (s *TenancyGuardSuite) TestPerAppRouterNeverWritesAnotherCustomerIntoTheDeploymentApp() {
	s.chat.SetAppFor(suiteStreamKey, chattest.App{ID: suiteStreamApp})
	first, second := s.tenantWithApp(), s.tenantWithApp()

	s.drive(first)
	s.drive(second)

	for _, request := range s.chat.Requests(suiteStreamKey) {
		s.False(request.Writes(), "the deployment's key wrote %s %s for a customer with an app of its own",
			request.Method, request.Path)
	}
	for _, customer := range []tenant{first, second} {
		written := 0
		for _, request := range s.chat.Requests(customer.apiKey) {
			if request.Writes() {
				written++
			}
		}
		s.Positive(written, "what the customer did was written with its own key")
	}
}

func (s *TenancyGuardSuite) TestACustomerWithNoAppIsWrittenNowhereWhenTheFallbackIsOff() {
	s.useApp(s.numberedApp(streamAppID()))
	before := len(s.chat.Requests(suiteStreamKey))

	status, failure := s.serverClient.failure(http.MethodPost, "/v1/agents/sessions", textSession(nil))

	s.Equal(http.StatusBadRequest, status)
	s.Contains(failure, "register this app's Stream keys")
	s.Len(s.chat.Requests(suiteStreamKey), before, "nothing was asked of the deployment's app on its behalf")
}

func (s *TenancyGuardSuite) TestTheDeploymentAppsHooksStartNothingForACustomerOnlyReadThere() {
	// A customer with no app of its own, under a refusing fallback, can still have rows in
	// the deployment's app from before. A hook from that app must not start its work.
	s.chat.SetAppFor(suiteStreamKey, chattest.App{ID: suiteStreamApp})
	s.useApp(s.numberedApp(streamAppID()))
	agent := "chat-" + s.utils.uuid()
	s.Require().NoError(s.store.StartCall(context.Background(), &store.Call{
		ID: s.utils.uuid(), CustomerID: s.customerID(), CallID: s.utils.callID(), AgentID: agent,
		StartedAt: time.Now().UTC(),
	}))
	worker, release := s.dispatch.Register(s.customerID(), dispatch.Registration{Capacity: 1})
	defer release()
	body := fmt.Sprintf(`{"type": "message.new", "channel_id": %q, "channel_type": "agent", "created_at": %q,
  "message": {"id": %q, "text": "hello", "user": {"id": "sam"}}}`,
		agent, time.Now().UTC().Format(time.RFC3339Nano), "message-"+s.utils.uuid())

	s.Equal(http.StatusOK, s.signedly("/v1/chat/hooks/stream", body))

	select {
	case <-worker.Messages():
		s.Fail("a deployment app hook started work for a customer only read there")
	case <-time.After(dropped):
	}
}

func (s *TenancyGuardSuite) TestASessionWithNoStreamAppIsPinnedToNone() {
	// A call needs no conversation kept, so a customer with no app can still hold one.
	s.useApp(s.numberedApp(streamAppID()))
	call := s.utils.callID()

	created := s.serverClient.createSession(CreateSessionRequest{CallId: &call})

	s.Require().Eventually(func() bool {
		stored, err := s.store.StoredSession(context.Background(), s.customerID(), created.Id)
		return err == nil && stored.StreamAppPK == store.ForeignStreamApp
	}, settleFor, 10*time.Millisecond, "the session is not left reading as the deployment app's")
}
