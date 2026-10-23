//go:build integration

package api

import (
	"context"
	"errors"
	"net/http"
	"strconv"
	"testing"
	"time"

	"github.com/GetStream/Vision-Agents/acceleration/internal/conversation/chattest"
	"github.com/GetStream/Vision-Agents/acceleration/internal/routing"
	"github.com/GetStream/Vision-Agents/acceleration/internal/session"
	"github.com/GetStream/Vision-Agents/acceleration/internal/store"
)

// SessionVoiceSuite covers starting and stopping voice on a session: the agent joins the
// call named after the session and leaves it, and the conversation is one either way.
type SessionVoiceSuite struct {
	RouterSuite
}

func TestSessionVoiceSuite(t *testing.T) {
	runSuite(t, new(SessionVoiceSuite))
}

func (s *SessionVoiceSuite) SetupTest() {
	s.useApp(s.data.createApp())
}

func (s *SessionVoiceSuite) TestStartingVoicePutsTheAgentOnTheCallNamedAfterTheSession() {
	opened := s.serverClient.createSession(textSession(nil))
	s.Empty(opened.CallId, "a session held in writing is on no call")

	started := s.startVoice(opened.Id)

	s.Equal(opened.Id, started.CallId)
	s.Equal("agent", started.CallType)
	s.Equal(value(opened.ConversationId), value(started.ConversationId), "the conversation stays in its channel")
	s.Equal(opened.Id, s.serverClient.getSession(opened.Id).CallId)
}

func (s *SessionVoiceSuite) TestASessionHeldByAStringIdIsOnTheCallNamedAfterIt() {
	id := "order_4471-" + s.utils.uuid()[24:]
	request := textSession(&id)
	request.StartVoice = pointerTo(true)

	s.Equal(id, s.serverClient.createSession(request).CallId)
}

func (s *SessionVoiceSuite) TestASessionCreatedToStartVoiceIsOnItsCall() {
	opened := s.serverClient.createSession(CreateSessionRequest{StartVoice: pointerTo(true)})

	s.Equal(opened.Id, opened.CallId)
	s.Equal("agent", opened.CallType)
}

func (s *SessionVoiceSuite) TestASessionCreatedOnANamedCallJoinsIt() {
	s.inOwnApp()
	call := s.namedCall()

	opened := s.serverClient.createSession(CreateSessionRequest{StartVoice: pointerTo(true), Call: &call})

	s.Equal("default", opened.CallType)
	s.Equal(call.Id, opened.CallId)
	s.Equal(call.Id, s.serverClient.getSession(opened.Id).CallId)
}

func (s *SessionVoiceSuite) TestStartingVoiceOnANamedCallJoinsIt() {
	s.inOwnApp()
	opened := s.serverClient.createSession(textSession(nil))
	call := s.namedCall()

	started := s.startVoiceOn(opened.Id, call)

	s.Equal("default", started.CallType)
	s.Equal(call.Id, started.CallId)
}

func (s *SessionVoiceSuite) TestACallNamedForASessionHeldInWritingIsRefused() {
	call := s.namedCall()

	status, failure := s.serverClient.failure(http.MethodPost, "/v1/agents/sessions",
		CreateSessionRequest{Call: &call})

	s.Equal(http.StatusBadRequest, status)
	s.Equal(errCallWithoutVoice.Message, failure)
}

func (s *SessionVoiceSuite) TestACallNamedForAPhoneSessionIsRefused() {
	call := s.namedCall()

	status, failure := s.serverClient.failure(http.MethodPost, "/v1/agents/sessions", CreateSessionRequest{
		StartVoice: pointerTo(true), Call: &call, Phone: &SessionPhone{Number: "+15551234567"},
	})

	s.Equal(http.StatusBadRequest, status)
	s.Equal(errCallWithPhone.Message, failure)
}

func (s *SessionVoiceSuite) TestVoiceOnAPhoneSessionTakesNoCall() {
	s.inOwnApp()
	// The suite has no telephony to place a call with, so the session is opened as the phone
	// service opens one, by the manager.
	opened, err := s.manager.Create(context.Background(), session.Spec{CustomerID: s.customerID(), Phone: &session.PhoneSpec{},
		Attribution: routing.Attribution{APIKey: "key-phone"}})
	s.Require().NoError(err)
	call := s.namedCall()

	status, failure := s.serverClient.failure(http.MethodPost, "/v1/agents/sessions/"+opened.ID()+"/voice",
		StartSessionVoiceRequest{Call: &call})

	s.Equal(http.StatusBadRequest, status)
	s.Equal(errCallWithPhone.Message, failure)
}

func (s *SessionVoiceSuite) TestADeviceMayNotNameACall() {
	call := s.namedCall()
	opened := s.client.createSession(textSession(nil))

	created, createFailure := s.client.failure(http.MethodPost, "/v1/agents/sessions",
		CreateSessionRequest{StartVoice: pointerTo(true), Call: &call})
	started, startFailure := s.client.failure(http.MethodPost, "/v1/agents/sessions/"+opened.Id+"/voice",
		StartSessionVoiceRequest{Call: &call})

	s.Equal(http.StatusForbidden, created)
	s.Equal(errDeviceCall.Message, createFailure)
	s.Equal(http.StatusForbidden, started)
	s.Equal(errDeviceCall.Message, startFailure)
	s.Empty(s.client.getSession(opened.Id).CallId, "the device's session stays in writing")
}

// The control: a device starts voice on its session's own call, as before.
func (s *SessionVoiceSuite) TestADeviceStartsVoiceOnItsSessionsOwnCall() {
	opened := s.client.createSession(textSession(nil))

	var started Session
	s.Require().Equal(http.StatusOK, s.client.do(http.MethodPost,
		"/v1/agents/sessions/"+opened.Id+"/voice", nil, &started))

	s.Equal(opened.Id, started.CallId)
}

func (s *SessionVoiceSuite) TestVoiceStartedAgainGoesBackToTheNamedCall() {
	s.inOwnApp()
	call := s.namedCall()
	opened := s.serverClient.createSession(CreateSessionRequest{StartVoice: pointerTo(true), Call: &call})
	s.stopVoice(opened.Id)

	again := s.startVoice(opened.Id)

	s.Equal("default", again.CallType)
	s.Equal(call.Id, again.CallId)
}

func (s *SessionVoiceSuite) TestVoiceStartedAgainOnAnotherCallMovesThere() {
	s.inOwnApp()
	first, second := s.namedCall(), s.namedCall()
	opened := s.serverClient.createSession(CreateSessionRequest{StartVoice: pointerTo(true), Call: &first})
	s.stopVoice(opened.Id)

	moved := s.startVoiceOn(opened.Id, second)

	s.Equal(second.Id, moved.CallId)
}

func (s *SessionVoiceSuite) TestNamingAnotherCallWhileVoiceIsOnIsRefused() {
	s.inOwnApp()
	first, second := s.namedCall(), s.namedCall()
	opened := s.serverClient.createSession(CreateSessionRequest{StartVoice: pointerTo(true), Call: &first})

	status, failure := s.serverClient.failure(http.MethodPost, "/v1/agents/sessions/"+opened.Id+"/voice",
		StartSessionVoiceRequest{Call: &second})

	s.Equal(http.StatusConflict, status)
	s.Equal(errVoiceOnAnotherCall.Message, failure)
	s.Equal(first.Id, s.serverClient.getSession(opened.Id).CallId, "the agent stays on its call")
}

// The control: naming the call voice is already on changes nothing.
func (s *SessionVoiceSuite) TestNamingTheCallVoiceIsOnChangesNothing() {
	s.inOwnApp()
	call := s.namedCall()
	opened := s.serverClient.createSession(CreateSessionRequest{StartVoice: pointerTo(true), Call: &call})

	again := s.startVoiceOn(opened.Id, call)

	s.Equal(call.Id, again.CallId)
}

func (s *SessionVoiceSuite) TestStartingVoiceOnACallInAnotherAppIsRefused() {
	opened := s.serverClient.createSession(textSession(nil))
	call := s.namedCall()
	s.pinCall(call, 4242)

	status, failure := s.serverClient.failure(http.MethodPost, "/v1/agents/sessions/"+opened.Id+"/voice",
		StartSessionVoiceRequest{Call: &call})

	s.Equal(http.StatusBadRequest, status)
	s.Equal(errCallInAnotherApp.Message, failure)
}

// Another session runs the call in the app it was created in, and the call's lines have
// since been pinned to another app, which a new session would act in.
func (s *SessionVoiceSuite) TestCreatingASessionOnACallInAnotherAppIsRefused() {
	s.inOwnApp()
	call := s.namedCall()
	s.serverClient.createSession(CreateSessionRequest{StartVoice: pointerTo(true), Call: &call})
	s.pinCall(call, 4242)

	status, failure := s.serverClient.failure(http.MethodPost, "/v1/agents/sessions",
		CreateSessionRequest{StartVoice: pointerTo(true), Call: &call})

	s.Equal(http.StatusBadRequest, status)
	s.Equal(errCallInAnotherApp.Message, failure)
}

// The session ended, so voice is started on it again from its row alone, as another process
// would.
func (s *SessionVoiceSuite) TestASessionCarriedOnFromItsRowGoesBackToTheNamedCall() {
	s.inOwnApp()
	call := s.namedCall()
	opened := s.serverClient.createSession(CreateSessionRequest{StartVoice: pointerTo(true), Call: &call})
	// The row is written off the request path, and the session is carried on from it.
	s.Require().Eventually(func() bool {
		_, err := s.store.StoredSession(context.Background(), s.customerID(), opened.Id)
		return err == nil
	}, settleFor, 10*time.Millisecond, "the session was never written down")
	s.serverClient.stopSession(opened.Id)
	s.assertEnds(opened.Id)

	again := s.startVoice(opened.Id)

	s.Equal("default", again.CallType)
	s.Equal(call.Id, again.CallId)
}

func (s *SessionVoiceSuite) TestAForkJoinsItsOwnCallRatherThanTheParents() {
	s.inOwnApp()
	call := s.namedCall()
	opened := s.serverClient.createSession(CreateSessionRequest{StartVoice: pointerTo(true), Call: &call})
	var forked Session
	s.Require().Equal(http.StatusCreated, s.serverClient.do(http.MethodPost,
		"/v1/agents/sessions/"+opened.Id+"/fork", ForkSessionRequest{Messages: pointerTo(false)}, &forked))

	started := s.startVoice(forked.Id)

	s.Equal("agent", started.CallType)
	s.Equal(forked.Id, started.CallId)
}

func (s *SessionVoiceSuite) TestASecondSessionOnANamedCallEndsTheFirst() {
	s.inOwnApp()
	call := s.namedCall()
	first := s.serverClient.createSession(CreateSessionRequest{StartVoice: pointerTo(true), Call: &call})

	s.serverClient.createSession(CreateSessionRequest{StartVoice: pointerTo(true), Call: &call})

	s.assertEnds(first.Id)
}

func (s *SessionVoiceSuite) TestStartingVoiceOnANamedCallEndsTheSessionAlreadyOnIt() {
	s.inOwnApp()
	call := s.namedCall()
	first := s.serverClient.createSession(CreateSessionRequest{StartVoice: pointerTo(true), Call: &call})
	second := s.serverClient.createSession(textSession(nil))

	s.startVoiceOn(second.Id, call)

	s.assertEnds(first.Id)
	s.Equal(Live, s.serverClient.getSession(second.Id).State)
}

func (s *SessionVoiceSuite) TestStoppingVoiceCarriesTheConversationOnInWriting() {
	request := textSession(nil)
	request.Llm = pointerTo("recites/recites-model")
	opened := s.serverClient.createSession(request)
	s.startVoice(opened.Id)
	s.answerTo(opened.Id, "Where is order 4471?")

	var stopped Session
	s.Require().Equal(http.StatusOK, s.serverClient.do(http.MethodDelete,
		"/v1/agents/sessions/"+opened.Id+"/voice", nil, &stopped))

	s.Empty(stopped.CallId)
	s.Equal(value(opened.ConversationId), value(stopped.ConversationId))
	s.Contains(s.answerTo(opened.Id, "And when?"), "Where is order 4471?",
		"what was typed while voice was on is part of the conversation")
}

func (s *SessionVoiceSuite) TestStartingVoiceTwiceKeepsTheAgentOnTheSameCall() {
	opened := s.serverClient.createSession(textSession(nil))
	s.startVoice(opened.Id)

	again := s.startVoice(opened.Id)

	s.Equal(opened.Id, again.CallId)
}

func (s *SessionVoiceSuite) TestStartingVoiceOnASessionNobodyHasIsNotFound() {
	status, _ := s.serverClient.call(http.MethodPost, "/v1/agents/sessions/"+s.utils.uuid()+"/voice", nil)

	s.Equal(http.StatusNotFound, status)
}

// A shared Stream app holds every customer's calls, so a customer acting in one names none:
// not another customer's live agent call, not a default call.
func (s *SessionVoiceSuite) TestACustomerInASharedAppMayNotNameACall() {
	otherCustomer := s.serverClient
	other := otherCustomer.createSession(CreateSessionRequest{StartVoice: pointerTo(true)})
	s.useApp(s.data.createApp())

	for _, call := range []SessionCall{{Type: "agent", Id: other.Id}, s.namedCall()} {
		status, failure := s.serverClient.failure(http.MethodPost, "/v1/agents/sessions",
			CreateSessionRequest{StartVoice: pointerTo(true), Call: &call})

		s.Equal(http.StatusForbidden, status, call.Type)
		s.Equal(errCallInSharedApp.Message, failure, call.Type)
	}
	s.Equal(Live, otherCustomer.getSession(other.Id).State, "the other customer's agent stays on its call")
}

func (s *SessionVoiceSuite) TestStartingVoiceOnANamedCallInASharedAppIsRefused() {
	opened := s.serverClient.createSession(textSession(nil))
	call := s.namedCall()

	status, failure := s.serverClient.failure(http.MethodPost, "/v1/agents/sessions/"+opened.Id+"/voice",
		StartSessionVoiceRequest{Call: &call})

	s.Equal(http.StatusForbidden, status)
	s.Equal(errCallInSharedApp.Message, failure)
	s.Empty(s.serverClient.getSession(opened.Id).CallId, "the session stays in writing")
}

// inOwnApp makes the suite's clients a customer acting in a Stream app of its own, which is
// the app a named call has to be in.
func (s *SessionVoiceSuite) inOwnApp() {
	id := streamAppID()
	s.useApp(s.numberedApp(id))
	s.giveApp(s.customerID(), id, "key-"+s.customerID())
}

// namedCall is a default call nobody has named before.
func (s *SessionVoiceSuite) namedCall() SessionCall {
	return SessionCall{Type: "default", Id: "practice-" + s.utils.uuid()}
}

// pinCall gives a call lines made in the app given, as a placed call's are.
func (s *SessionVoiceSuite) pinCall(call SessionCall, app int64) {
	s.Require().NoError(s.store.RecordCallResource(context.Background(), &store.CallResource{
		TrunkID: "trunk-" + s.utils.uuid(), RouteID: "route-" + s.utils.uuid(),
		CallType: call.Type, CallID: call.Id, CustomerID: s.customerID(), StreamAppPK: app,
	}))
}

// assertEnds waits for a session to end.
func (s *SessionVoiceSuite) assertEnds(id string) {
	s.Require().Eventually(func() bool {
		var read Session
		return s.serverClient.do(http.MethodGet, "/v1/agents/sessions/"+id, nil, &read) == http.StatusOK && read.State == Ended
	}, settleFor, 20*time.Millisecond, "the session left in the call never ended")
}

// startVoiceOn starts voice on a session on the call named and returns what it says of itself.
func (s *SessionVoiceSuite) startVoiceOn(id string, call SessionCall) Session {
	var started Session
	s.Require().Equal(http.StatusOK, s.serverClient.do(http.MethodPost,
		"/v1/agents/sessions/"+id+"/voice", StartSessionVoiceRequest{Call: &call}, &started))
	return started
}

// stopVoice stops voice on a session.
func (s *SessionVoiceSuite) stopVoice(id string) {
	s.Require().Equal(http.StatusOK, s.serverClient.do(http.MethodDelete,
		"/v1/agents/sessions/"+id+"/voice", nil, nil))
}

// startVoice starts voice on a session and returns what it says of itself.
func (s *SessionVoiceSuite) startVoice(id string) Session {
	var started Session
	s.Require().Equal(http.StatusOK, s.serverClient.do(http.MethodPost,
		"/v1/agents/sessions/"+id+"/voice", nil, &started))
	return started
}

// answerTo asks a running session a question in writing and returns what it answered.
func (s *SessionVoiceSuite) answerTo(id, question string) string {
	watching := s.serverClient.opens("/v1/agents/sessions/" + id + "/events")
	s.Require().Equal(http.StatusAccepted, s.serverClient.do(http.MethodPost,
		"/v1/agents/sessions/"+id+"/responses", CreateResponseRequest{Text: question}, nil))
	answered, _ := s.await(watching, "responded")["text"].(string)
	return answered
}

// DeploymentNamedCallSuite names calls in deployment tenancy with the deployment's own app
// known, where every customer acts in that one app and only its own customer owns it.
type DeploymentNamedCallSuite struct {
	RouterSuite
}

func TestDeploymentNamedCallSuite(t *testing.T) {
	runSuite(t, new(DeploymentNamedCallSuite))
}

func (s *DeploymentNamedCallSuite) SetupSuite() {
	s.deploymentKnown = true
	s.RouterSuite.SetupSuite()
}

func (s *DeploymentNamedCallSuite) TestTheDeploymentAppsOwnCustomerNamesACall() {
	s.asTheDeploymentsCustomer()
	call := SessionCall{Type: "default", Id: "practice-" + s.utils.uuid()}

	opened := s.serverClient.createSession(CreateSessionRequest{StartVoice: pointerTo(true), Call: &call})

	s.Equal(call.Id, opened.CallId)
}

func (s *DeploymentNamedCallSuite) TestAnotherCustomerOfTheDeploymentAppMayNotNameACall() {
	s.useApp(s.data.createApp())
	call := SessionCall{Type: "default", Id: "practice-" + s.utils.uuid()}

	status, failure := s.serverClient.failure(http.MethodPost, "/v1/agents/sessions",
		CreateSessionRequest{StartVoice: pointerTo(true), Call: &call})

	s.Equal(http.StatusForbidden, status)
	s.Equal(errCallInSharedApp.Message, failure)
}

// The control: another customer of the shared app still opens a session on its own call.
func (s *DeploymentNamedCallSuite) TestAnotherCustomerStillJoinsItsSessionsOwnCall() {
	s.useApp(s.data.createApp())

	opened := s.serverClient.createSession(CreateSessionRequest{StartVoice: pointerTo(true)})

	s.Equal("agent", opened.CallType)
	s.Equal(opened.Id, opened.CallId)
}

// asTheDeploymentsCustomer makes the suite's clients the customer whose app the deployment's
// is, which an earlier run may have made already.
func (s *DeploymentNamedCallSuite) asTheDeploymentsCustomer() {
	app, err := s.store.AppByID(context.Background(), strconv.Itoa(suiteStreamApp))
	if errors.Is(err, store.ErrNoApp) {
		s.useApp(s.numberedApp(suiteStreamApp))
		return
	}
	s.Require().NoError(err)
	s.useApp(s.data.keyed(store.Organization{ID: app.OrganizationID}, app))
}

// AppModeNamedCallSuite names calls in app tenancy, where a customer acts in its own app or,
// with none, falls back to the deployment's app that every such customer shares.
type AppModeNamedCallSuite struct {
	RouterSuite
}

func TestAppModeNamedCallSuite(t *testing.T) {
	runSuite(t, new(AppModeNamedCallSuite))
}

func (s *AppModeNamedCallSuite) SetupSuite() {
	s.appMode = true
	s.RouterSuite.SetupSuite()
}

func (s *AppModeNamedCallSuite) TestACustomerInItsOwnAppNamesACall() {
	id := streamAppID()
	s.useApp(s.numberedApp(id))
	key := "key-" + s.customerID()
	s.answerAs(key, id)
	s.forwarded(key)
	call := SessionCall{Type: "default", Id: "practice-" + s.utils.uuid()}

	opened := s.serverClient.createSession(CreateSessionRequest{StartVoice: pointerTo(true), Call: &call})

	s.Equal(call.Id, opened.CallId)
}

func (s *AppModeNamedCallSuite) TestACustomerFallingBackToTheDeploymentAppMayNotNameACall() {
	s.chat.SetAppFor(suiteStreamKey, chattest.App{ID: suiteStreamApp})
	s.useApp(s.numberedApp(streamAppID()))
	call := SessionCall{Type: "default", Id: "practice-" + s.utils.uuid()}

	status, failure := s.serverClient.failure(http.MethodPost, "/v1/agents/sessions",
		CreateSessionRequest{StartVoice: pointerTo(true), Call: &call})

	s.Equal(http.StatusForbidden, status)
	s.Equal(errCallInSharedApp.Message, failure)
}
