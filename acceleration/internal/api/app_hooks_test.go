//go:build integration

package api

import (
	"context"
	"fmt"
	"net/http"
	"strconv"
	"testing"
	"time"

	"github.com/GetStream/Vision-Agents/acceleration/internal/auth"
	"github.com/GetStream/Vision-Agents/acceleration/internal/dispatch"
	"github.com/GetStream/Vision-Agents/acceleration/internal/store"
)

// AppHooksSuite delivers Stream's hooks to a router in app mode, where each registered app
// signs its own and what a hook names is acted on only within the app it came from.
type AppHooksSuite struct {
	RouterSuite

	channelID string
	apiKey    string
	secret    string
}

func TestAppHooksSuite(t *testing.T) {
	runSuite(t, new(AppHooksSuite))
}

func (s *AppHooksSuite) SetupSuite() {
	s.appMode = true
	s.RouterSuite.SetupSuite()
}

func (s *AppHooksSuite) SetupTest() {
	s.useApp(s.numberedApp(streamAppID()))
	s.answerAs(s.appID())
	s.apiKey, s.secret = "own-key-"+s.utils.uuid(), "own-secret-"+s.utils.uuid()
	s.registered(0, key(s.apiKey, s.secret))
	s.channelID = "chat-" + s.utils.uuid()
}

func (s *AppHooksSuite) appPath(hook string) string {
	return hook + "/" + strconv.FormatInt(s.appID(), 10)
}

// hook delivers a body signed with a secret, naming the key given when there is one.
func (s *AppHooksSuite) hook(path, body, secret, apiKey string) int {
	request := s.request(path, body, sign(body, secret))
	if apiKey != "" {
		request.Header.Set(auth.APIKeyHeader, apiKey)
	}
	response, err := s.server.Client().Do(request)
	s.Require().NoError(err)
	s.Require().NoError(response.Body.Close())
	return response.StatusCode
}

func (s *AppHooksSuite) config(customer string) string {
	config := store.AgentConfig{CustomerID: customer, Name: "config-" + s.utils.uuid(), Mode: "text"}
	s.Require().NoError(s.store.CreateAgentConfig(context.Background(), &config))
	return config.ID
}

// message is a message.new on the suite's channel, which names the config given.
func (s *AppHooksSuite) message(configID string) string {
	return fmt.Sprintf(`{
  "type": "message.new",
  "channel_id": %q,
  "channel_type": "agent",
  "created_at": %q,
  "channel_custom": {%q: %q},
  "message": {"id": %q, "text": "is my order on its way?", "user": {"id": "sam", "name": "Sam"}}
}`, s.channelID, time.Now().UTC().Format(time.RFC3339Nano), ConfigField, configID, "message-"+s.utils.uuid())
}

func (s *AppHooksSuite) reaches(messages <-chan dispatch.Message) bool {
	select {
	case <-messages:
		return true
	case <-time.After(dropped):
		return false
	}
}

func (s *AppHooksSuite) TestAMessageHookIsVerifiedWithTheSendingAppsSecret() {
	worker, release := s.dispatch.Register(s.customerID(), 1)
	defer release()

	s.Equal(http.StatusOK, s.hook(s.appPath("/v1/chat/hooks/stream"), s.message(s.config(s.customerID())), s.secret, s.apiKey))

	s.True(s.reaches(worker.Messages()))
	app, err := s.store.StreamApp(context.Background(), s.customerID())
	s.Require().NoError(err)
	s.NotNil(app.Keys[0].LastWebhookAt, "the key is known to sign the app's hooks")
}

func (s *AppHooksSuite) TestAHookWithAnAppSegmentAndNoKeyTriesThatAppsKeys() {
	worker, release := s.dispatch.Register(s.customerID(), 1)
	defer release()

	s.Equal(http.StatusOK, s.hook(s.appPath("/v1/chat/hooks/stream"), s.message(s.config(s.customerID())), s.secret, ""))

	s.True(s.reaches(worker.Messages()))
}

func (s *AppHooksSuite) TestAHookIsVerifiedOnlyWithItsAppsKeys() {
	// The deployment's secret signs only the deployment app's hooks.
	s.Equal(http.StatusUnauthorized, s.hook(s.appPath("/v1/chat/hooks/stream"),
		s.message(s.config(s.customerID())), suiteStreamSecret, ""))
}

func (s *AppHooksSuite) TestAnUnknownKeyWithAnAppSegmentNeverUsesTheDeploymentSecret() {
	s.Equal(http.StatusUnauthorized, s.hook(s.appPath("/v1/chat/hooks/stream"),
		s.message(s.config(s.customerID())), suiteStreamSecret, "nobody's-key"))
}

func (s *AppHooksSuite) TestANamedKeyNeverWidensTheKeysTried() {
	second, secondSecret := "second-key-"+s.utils.uuid(), "second-secret-"+s.utils.uuid()
	s.registered(1, key(s.apiKey, s.secret), key(second, secondSecret))

	s.Equal(http.StatusUnauthorized, s.hook(s.appPath("/v1/chat/hooks/stream"),
		s.message(s.config(s.customerID())), s.secret, second), "signed with one key and naming the other")
}

func (s *AppHooksSuite) TestAHookSignedWithASecondaryKeyIsAccepted() {
	second, secondSecret := "second-key-"+s.utils.uuid(), "second-secret-"+s.utils.uuid()
	s.registered(1, key(s.apiKey, s.secret), key(second, secondSecret))
	worker, release := s.dispatch.Register(s.customerID(), 1)
	defer release()

	s.Equal(http.StatusOK, s.hook(s.appPath("/v1/chat/hooks/stream"), s.message(s.config(s.customerID())), secondSecret, second))

	s.True(s.reaches(worker.Messages()))
}

func (s *AppHooksSuite) TestAHookSignedByAnUnregisteredKeyUsesTheDeploymentSecret() {
	// A hook to the old path naming a key no app holds is the deployment app's.
	s.Equal(http.StatusOK, s.hook("/v1/chat/hooks/stream", s.message(s.config(s.customerID())), suiteStreamSecret, "nobody's-key"))
}

func (s *AppHooksSuite) TestAKnownKeyOnTheOldPathIsCheckedOnlyWithItsSecret() {
	s.Equal(http.StatusUnauthorized, s.hook("/v1/chat/hooks/stream", s.message(s.config(s.customerID())), suiteStreamSecret, s.apiKey))
}

func (s *AppHooksSuite) TestAChannelConfigIsHonouredOnlyForTheHooksApp() {
	// A channel in this app naming another customer's config starts nobody's agent.
	other := "someone-" + s.utils.uuid()
	somebodyElse, release := s.dispatch.Register(other, 1)
	defer release()

	s.Equal(http.StatusOK, s.hook(s.appPath("/v1/chat/hooks/stream"), s.message(s.config(other)), s.secret, s.apiKey))

	s.False(s.reaches(somebodyElse.Messages()))
}

func (s *AppHooksSuite) TestAMessageInTheDeploymentAppIsNotDispatchedToACustomerNowElsewhere() {
	// A channel in the router's own app names this customer's config, but the customer
	// now acts in its own app, so a deployment-signed hook starts nothing of theirs.
	worker, release := s.dispatch.Register(s.customerID(), 1)
	defer release()

	s.Equal(http.StatusOK, s.hook("/v1/chat/hooks/stream", s.message(s.config(s.customerID())), suiteStreamSecret, ""))

	s.False(s.reaches(worker.Messages()))
}

func (s *AppHooksSuite) TestAReplayedMessageIsAnsweredOnce() {
	worker, release := s.dispatch.Register(s.customerID(), 2)
	defer release()
	body := s.message(s.config(s.customerID()))

	s.Equal(http.StatusOK, s.hook(s.appPath("/v1/chat/hooks/stream"), body, s.secret, s.apiKey))
	s.Equal(http.StatusOK, s.hook(s.appPath("/v1/chat/hooks/stream"), body, s.secret, s.apiKey))

	s.True(s.reaches(worker.Messages()))
	s.False(s.reaches(worker.Messages()), "the second delivery is the same message")
}

func (s *AppHooksSuite) TestAStaleCallEventIsIgnored() {
	e164 := s.utils.number()
	call := "phone-" + e164
	s.attach(e164, call, s.appID())
	worker, release := s.dispatch.Register(s.customerID(), 1)
	defer release()

	s.Equal(http.StatusOK, s.hook(s.appPath("/v1/phone/hooks/stream"),
		s.ringing(call, time.Now().Add(-time.Hour)), s.secret, s.apiKey))

	select {
	case <-worker.Calls():
		s.Fail("an hour-old call was answered")
	case <-time.After(dropped):
	}
}

func (s *AppHooksSuite) TestAnArrivingCallIsDispatchedToTheNumbersOwnerInThatApp() {
	e164 := s.utils.number()
	call := "phone-" + e164
	s.attach(e164, call, s.appID())
	worker, release := s.dispatch.Register(s.customerID(), 2)
	defer release()

	s.Equal(http.StatusOK, s.hook("/v1/phone/hooks/stream", s.ringing(call, time.Now()), suiteStreamSecret, ""))
	select {
	case <-worker.Calls():
		s.Fail("the deployment app's hook rang a number attached in the customer's app")
	case <-time.After(dropped):
	}

	s.Equal(http.StatusOK, s.hook(s.appPath("/v1/phone/hooks/stream"), s.ringing(call, time.Now()), s.secret, s.apiKey))
	select {
	case arrived := <-worker.Calls():
		s.Equal(call, arrived.CallID)
	case <-time.After(settleFor):
		s.Fail("the call never reached the worker")
	}
}

// attach is a number the customer holds, attached in the app given.
func (s *AppHooksSuite) attach(e164, call string, app int64) {
	ctx := context.Background()
	s.Require().NoError(s.store.RecordNumber(ctx, &store.PhoneNumber{
		E164: e164, Vendor: "telnyx", Country: "US", CustomerID: s.customerID(), PurchasedAt: time.Now().UTC(),
	}))
	s.Require().NoError(s.store.AttachNumber(ctx, s.customerID(), e164, store.NumberAttachment{
		TrunkID: "trunk-" + s.utils.uuid(), StreamAppPK: app, CallType: "default", CallID: call,
	}))
}

// ringing is a call.session_started for a call, made at the time given.
func (s *AppHooksSuite) ringing(call string, at time.Time) string {
	return fmt.Sprintf(`{"type": "call.session_started", "call_cid": "default:%s", "session_id": %q,
  "created_at": %q, "call": {"id": %q, "type": "default", "custom": {}}}`,
		call, "session-"+s.utils.uuid(), at.UTC().Format(time.RFC3339Nano), call)
}

func (s *AppHooksSuite) TestACallOfTheSameNameInAnotherAppDoesNotHideThisOne() {
	// Call ids are only unique within an app: another customer attaching a line to a call of
	// the same name, in its own app, must not stop this customer's ringing.
	line := "line-" + s.utils.uuid()
	other := s.numberedApp(streamAppID())
	ctx := context.Background()
	otherNumber := s.utils.number()
	s.Require().NoError(s.store.RecordNumber(ctx, &store.PhoneNumber{
		E164: otherNumber, Vendor: "telnyx", Country: "US", CustomerID: other.app.ID, PurchasedAt: time.Now().UTC(),
	}))
	s.Require().NoError(s.store.AttachNumber(ctx, other.app.ID, otherNumber, store.NumberAttachment{
		TrunkID: "trunk-" + s.utils.uuid(), StreamAppPK: 7, CallType: "default", CallID: line,
	}))
	s.attach(s.utils.number(), line, s.appID())
	worker, release := s.dispatch.Register(s.customerID(), 1)
	defer release()

	s.Equal(http.StatusOK, s.hook(s.appPath("/v1/phone/hooks/stream"), s.ringing(line, time.Now()), s.secret, s.apiKey))

	select {
	case arrived := <-worker.Calls():
		s.Equal(line, arrived.CallID)
	case <-time.After(settleFor):
		s.Fail("the call never reached the worker")
	}
}
