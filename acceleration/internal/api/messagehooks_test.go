//go:build integration

package api

import (
	"context"
	"fmt"
	"net/http"
	"strings"
	"testing"
	"time"

	"github.com/GetStream/Vision-Agents/acceleration/internal/chatlog"
	"github.com/GetStream/Vision-Agents/acceleration/internal/conversation"
	"github.com/GetStream/Vision-Agents/acceleration/internal/dispatch"
	"github.com/GetStream/Vision-Agents/acceleration/internal/store"
	"github.com/GetStream/Vision-Agents/acceleration/internal/streamapp"
)

type MessageHooksSuite struct {
	RouterSuite

	// channelID is the channel written in, which is also the agent id a session opens on.
	// It is unique, because a channel is looked up by it alone, and it is not named
	// support-<uuid>, which is the namespace durable session commands own.
	channelID string
	// logged is what the router logged, at debug and up.
	logged *lockedLog
}

func TestMessageHooksSuite(t *testing.T) {
	runSuite(t, new(MessageHooksSuite))
}

func (s *MessageHooksSuite) SetupSuite() {
	s.logged = &lockedLog{}
	s.logs = s.logged
	s.RouterSuite.SetupSuite()
}

func (s *MessageHooksSuite) SetupTest() {
	s.useApp(s.data.createApp())
	s.channelID = "chat-" + s.utils.uuid()
}

func (s *MessageHooksSuite) TestAMessageOnAChannelAConversationAlreadyRanOnReachesItsOwner() {
	configID := s.holds()
	s.ran(configID)
	worker, release := s.dispatch.Register(s.customerID(), dispatch.Registration{Capacity: 1})
	defer release()

	s.Require().Equal(http.StatusOK, s.wrote(""))

	select {
	case message := <-worker.Messages():
		s.Equal(s.channelID, message.ChannelID)
		s.Equal(s.channelID, message.AgentID, "a session opened on anything else answers elsewhere")
		s.Equal(configID, message.ConfigID)
		s.Equal("sam", message.UserID)
		s.Equal("does the react sdk retry a failed upload?", message.Text)
	case <-time.After(settleFor):
		s.Fail("the message never reached the worker")
	}
}

// Deployment mode acts on every delivery of a message, as it always has, while Stream Chat
// delivers none itself: only then does the webhook share a record of the message with it.
func (s *MessageHooksSuite) TestAMessageDeliveredTwiceIsActedOnTwiceWithoutStreamChatDeliveries() {
	configID := s.holds()
	worker, release := s.dispatch.Register(s.customerID(), dispatch.Registration{Capacity: 2})
	defer release()
	written := s.writing(fmt.Sprintf(`{"%s": "%s"}`, ConfigField, configID))

	s.Require().Equal(http.StatusOK, s.signedly("/v1/chat/hooks/stream", written))
	s.Require().Equal(http.StatusOK, s.signedly("/v1/chat/hooks/stream", written))

	for range 2 {
		select {
		case <-worker.Messages():
		case <-time.After(settleFor):
			s.Fail("a delivery of the message was not acted on")
		}
	}
}

// A plain chat channel, which no line binds, keeps the config of the conversation it last held
// over the one it names.
func (s *MessageHooksSuite) TestAChannelThatHeldAConversationIsAnsweredByItsConfigOverTheOneItNames() {
	ran, named := s.holds(), s.holds()
	s.ran(ran)
	worker, release := s.dispatch.Register(s.customerID(), dispatch.Registration{Capacity: 1})
	defer release()

	s.Require().Equal(http.StatusOK, s.wrote(fmt.Sprintf(`{"%s": "%s"}`, ConfigField, named)))

	select {
	case message := <-worker.Messages():
		s.Equal(ran, message.ConfigID)
	case <-time.After(settleFor):
		s.Fail("the message never reached the worker")
	}
}

func (s *MessageHooksSuite) TestAChannelThatNamesItsConfigIsAnsweredWithoutEverHavingHeldAConversation() {
	// This is a conversation that starts in writing: somebody opened a support chat
	// rather than ringing a number, so there is no call row to say whose channel it is.
	// Without the channel saying, nobody could ever be written to first.
	configID := s.holds()
	worker, release := s.dispatch.Register(s.customerID(), dispatch.Registration{Capacity: 1})
	defer release()

	s.Require().Equal(http.StatusOK, s.wrote(fmt.Sprintf(`{"%s": "%s"}`, ConfigField, configID)))

	select {
	case message := <-worker.Messages():
		s.Equal(s.channelID, message.ChannelID)
		s.Equal(configID, message.ConfigID, "the worker has to know which agent was written to")
	case <-time.After(settleFor):
		s.Fail("the message never reached the worker")
	}
}

func (s *MessageHooksSuite) TestWhatTheChannelWasCreatedWithIsCarriedToTheWorker() {
	// This service has no opinion about an organization or a locale and should not need
	// one. It carries what the channel was created with, and the worker, which is where
	// the agent runs, decides what any of it means.
	configID := s.holds()
	worker, release := s.dispatch.Register(s.customerID(), dispatch.Registration{Capacity: 1})
	defer release()

	s.Require().Equal(http.StatusOK, s.wrote(fmt.Sprintf(
		`{"%s": "%s", "organization_id": "1234", "locale": "en-GB", "seats": 12}`,
		ConfigField, configID)))

	select {
	case message := <-worker.Messages():
		s.Equal("1234", message.Custom["organization_id"])
		s.Equal("en-GB", message.Custom["locale"])
		s.NotContains(message.Custom, "seats",
			"Stream takes arbitrary JSON here and a worker reads strings; a number rendered into one would be believed")
	case <-time.After(settleFor):
		s.Fail("the message never reached the worker")
	}
}

func (s *MessageHooksSuite) TestAChannelCannotSendItsMessagesToAnotherCustomersWorkers() {
	// Whoever creates a channel decides what is on it, so the customer is worked out from
	// the config rather than read off the channel. A channel naming somebody else's config
	// is answered by that somebody else, and a channel claiming a customer is not answered
	// on that claim at all.
	configID := s.holds()
	somebodyElse, release := s.dispatch.Register(s.utils.uuid(), dispatch.Registration{Capacity: 1})
	defer release()

	s.Require().Equal(http.StatusOK, s.wrote(fmt.Sprintf(
		`{"customer_id": "somebody-else", "%s": "%s"}`, ConfigField, configID)))

	s.nothingReaches(somebodyElse.Messages(), "a message was misrouted")
}

func (s *MessageHooksSuite) TestAChannelNamingAConfigNobodyHoldsIsAcceptedAndDropped() {
	worker, release := s.dispatch.Register(s.customerID(), dispatch.Registration{Capacity: 1})
	defer release()

	s.Require().Equal(http.StatusOK,
		s.wrote(fmt.Sprintf(`{"%s": "config-nobody-has"}`, ConfigField)),
		"Stream retries a non-2xx, and no retry finds an owner")

	s.nothingReaches(worker.Messages(), "a message naming a config nobody holds was answered")
	// An app whose hooks deliver to two routers sends each the other's channels: no failure
	// of this router, so not an ERROR a post-deploy check counts.
	s.Contains(s.logged.String(), `level=INFO msg="an arriving message's channel names a config nobody in its app holds" channel=`+
		s.channelID+` config=config-nobody-has`)
	s.NotContains(s.logged.String(), `level=ERROR msg="an arriving message's channel names a config nobody in its app holds"`)
}

// A store failure reading the config is this router's failure, not another router's channel:
// it stays an ERROR, under its own message.
func (s *MessageHooksSuite) TestAStoreFailureReadingTheDeclaredConfigIsAnError() {
	ctx, cancel := context.WithCancel(context.Background())
	cancel()
	before := len(s.logged.String())

	_, _, found, err := s.router.ownerOf(ctx, hookOrigin{customer: s.customerID(), app: 77}, messageEvent{
		ChannelID: s.channelID, ChannelCustom: map[string]any{ConfigField: "config-unread"},
	})

	s.False(found)
	s.Error(err)
	logged := s.logged.String()[before:]
	s.Contains(logged, `level=ERROR msg="could not read the config an arriving message's channel names"`, logged)
	s.NotContains(logged, `level=INFO msg="an arriving message's channel names a config nobody in its app holds"`)
}

// A hook is not tried again, so a deployment's app not known yet ends it as a customer no
// longer writing there does: INFO, as before the queue. Only a queued delivery, which the
// worker retries, logs it as an ERROR and is left unsettled.
func (s *MessageHooksSuite) TestAnAppNotKnownYetEndsAHookAtInfoAndAQueuedDeliveryAtError() {
	customer := streamapp.CustomerOf(suiteStreamApp)
	s.setApps(customer, func(apps *suiteApps) { apps.perApp = true; apps.waiting[customer] = true })
	s.T().Cleanup(func() {
		s.setApps(customer, func(apps *suiteApps) { apps.perApp = false; delete(apps.waiting, customer) })
	})
	event := messageEvent{ChannelID: s.channelID, ChannelCustom: map[string]any{ConfigField: deliveredConfig(&s.RouterSuite, customer)}}
	event.Message.Text = "is my order on its way?"
	const line = `msg="not starting work for a customer whose app could not be read"`
	const hookLine = `level=INFO msg="not starting work in the deployment's app for a customer no longer writing there"`
	before := len(s.logged.String())

	settled := s.router.routeArrivingMessage(context.Background(), hookOrigin{app: suiteStreamApp, deployment: true}, event)

	s.True(settled)
	logged := s.logged.String()[before:]
	s.Contains(logged, hookLine, logged)
	s.NotContains(logged, line)

	before = len(s.logged.String())
	settled = s.router.routeArrivingMessage(context.Background(), hookOrigin{app: suiteStreamApp, deployment: true, queued: true}, event)

	s.False(settled)
	logged = s.logged.String()[before:]
	s.Contains(logged, "level=ERROR "+line, logged)
}

func (s *MessageHooksSuite) TestAChannelWithNoHistoryAndNoConfigIsAcceptedAndDropped() {
	// Every message in the app arrives at this hook. One in a channel nothing claims is
	// not answerable on a retry either.
	worker, release := s.dispatch.Register(s.customerID(), dispatch.Registration{Capacity: 1})
	defer release()

	s.Require().Equal(http.StatusOK, s.wrote(""))

	s.nothingReaches(worker.Messages(), "a message nothing says the owner of was answered")
}

func (s *MessageHooksSuite) TestAConfigThatWasDeletedNoLongerClaimsAChannel() {
	configID := s.holds()
	s.Require().NoError(s.store.DeleteAgentConfig(context.Background(), s.customerID(), configID))
	worker, release := s.dispatch.Register(s.customerID(), dispatch.Registration{Capacity: 1})
	defer release()

	s.Require().Equal(http.StatusOK, s.wrote(fmt.Sprintf(`{"%s": "%s"}`, ConfigField, configID)))

	s.nothingReaches(worker.Messages(), "a message naming a deleted config was answered")
}

func (s *MessageHooksSuite) TestAMessageTheRouterWroteItselfReachesNoWorker() {
	// An episode card is written into an agent channel with a source, as everything the
	// router stores there is, and Stream delivers its custom fields at the top level of the
	// message. Handed to a worker, it is the router answering itself; with none waiting, it
	// is an error about nobody answering a message nobody wrote (AI-989).
	configID := s.holds()
	s.ran(configID)
	worker, release := s.dispatch.Register(s.customerID(), dispatch.Registration{Capacity: 1})
	defer release()

	// The card's fields as GET /messages/{id} returned episode-46b7e83c… on 2026-10-09.
	s.Require().Equal(http.StatusOK, s.signedly("/v1/chat/hooks/stream", fmt.Sprintf(`{
  "type": "message.new",
  "cid": "agent:%[1]s",
  "channel_id": "%[1]s",
  "channel_type": "agent",
  "message": {
    "id": "episode-%[2]s",
    "text": "Episode in progress",
    "type": "regular",
    "user": {"id": "%[1]s"},
    "%[3]s": "slack",
    "status": "in_progress",
    "episode_id": "%[2]s",
    "started_at": "2026-10-09T13:30:37Z",
    "thread_channel": "agent:thread-%[2]s"
  }
}`, s.channelID, s.utils.uuid(), chatlog.SourceField)))

	s.nothingReaches(worker.Messages(), "the router's own card was handed to a worker")
}

func (s *MessageHooksSuite) TestAnUnsignedMessageIsRefused() {
	status, _ := s.deliver("/v1/chat/hooks/stream", s.writing(""), "")

	s.Equal(http.StatusUnauthorized, status)
}

func (s *MessageHooksSuite) TestAMessageSignedWithTheWrongSecretIsRefused() {
	body := s.writing("")

	status, _ := s.deliver("/v1/chat/hooks/stream", body, sign(body, "somebody-elses-secret"))

	s.Equal(http.StatusUnauthorized, status)
}

func (s *MessageHooksSuite) TestATamperedMessageIsRefused() {
	body := s.writing("")
	signature := sign(body, suiteStreamSecret)
	tampered := strings.Replace(body, "sam", "mallory", 1)

	status, _ := s.deliver("/v1/chat/hooks/stream", tampered, signature)

	s.Equal(http.StatusUnauthorized, status)
}

func (s *MessageHooksSuite) TestSomethingThatIsNotAMessageEventIsAccepted() {
	// Every event in the app arrives here, and one this hook has no use for is not an
	// outage worth retrying.
	s.Equal(http.StatusOK, s.signedly("/v1/chat/hooks/stream",
		`{"type":"reaction.new","cid":"agent:somewhere"}`))
}

// Only the session of the customer a channel names answers a message there, even with
// another customer's session on the channel's id newer.
func (s *MessageHooksSuite) TestAChannelsMessageReachesOnlyTheSessionOfTheCustomerItNames() {
	owner := s.customerID()
	ownSession, own := s.dispatching(), s.workerOf(owner)
	s.useApp(s.data.createApp())
	s.dispatching()
	other := s.workerOf(s.customerID())

	s.Require().Equal(http.StatusOK, s.wrote(s.ownedBy(owner)))

	s.Equal(ownSession, s.handed(own).SessionID)
	s.nothingReaches(other, "another customer's session answered the channel")
}

// With the channel's customer running nothing on it, the message goes to that customer's
// worker as on any channel nothing runs on, and never to the other customer's session.
func (s *MessageHooksSuite) TestAChannelsMessageReachesNoOtherCustomersSessionWhileItsOwnIsNotRunning() {
	owner, configID := s.customerID(), s.holds()
	own := s.workerOf(owner)
	s.useApp(s.data.createApp())
	s.dispatching()
	other := s.workerOf(s.customerID())

	s.Require().Equal(http.StatusOK, s.wrote(fmt.Sprintf(`{%q: %q, %q: %q}`,
		conversation.CustomerField, owner, ConfigField, configID)))

	handed := s.handed(own)
	s.Empty(handed.SessionID, "handed to the owner's worker to start a conversation")
	s.Equal(configID, handed.ConfigID)
	s.nothingReaches(other, "another customer's session answered the channel")
}

// A channel names its customer only as a string the router wrote. Anything else there names
// nobody's session, the other customer's included.
func (s *MessageHooksSuite) TestAChannelNamingItsCustomerAsAnythingButTheirIdReachesNoOtherCustomersSession() {
	owner := s.customerID()
	s.useApp(s.data.createApp())
	other := s.workerOf(s.customerID())
	for _, stamp := range []string{`""`, `"` + owner + `"`, `null`, `0`, `{}`, `[]`, `"` + strings.ToUpper(s.customerID()) + `"`} {
		s.channelID = "chat-" + s.utils.uuid()
		s.dispatching()

		s.Require().Equal(http.StatusOK, s.wrote(fmt.Sprintf(`{%q: %s}`, conversation.CustomerField, stamp)))

		s.nothingReaches(other, "a channel stamped "+stamp+" was answered by another customer's session")
	}
}

// A channel that names no customer and holds no call is answered by nobody while two
// customers' sessions run on it: neither is known to be its customer's. Nor by the config it
// names, which would start a second conversation beside the session running there.
func (s *MessageHooksSuite) TestAChannelNamingNobodyThatTwoCustomersSessionsRunOnIsAnsweredByNone() {
	s.dispatching()
	own := s.workerOf(s.customerID())
	s.useApp(s.data.createApp())
	s.dispatching()
	other := s.workerOf(s.customerID())

	s.Require().Equal(http.StatusOK, s.wrote(""))

	s.nothingReaches(own, "a channel two customers' sessions run on was answered")
	s.nothingReaches(other, "a channel two customers' sessions run on was answered")
	s.Contains(s.logged.String(), `msg="not answering a message in a channel more than one customer's sessions run on" channel=`+s.channelID)
}

// A channel naming nobody whose calls are one customer's is answered only by that customer:
// with nothing of theirs running, by their worker, never by another customer's session.
func (s *MessageHooksSuite) TestAChannelNamingNobodyIsAnsweredOnlyByTheCustomerItsCallsAre() {
	owner, configID := s.customerID(), s.holds()
	s.ran(configID)
	own := s.workerOf(owner)
	s.useApp(s.data.createApp())
	s.dispatching()
	other := s.workerOf(s.customerID())

	s.Require().Equal(http.StatusOK, s.wrote(""))

	handed := s.handed(own)
	s.Empty(handed.SessionID, "handed to the owner's worker to start a conversation")
	s.Equal(configID, handed.ConfigID)
	s.nothingReaches(other, "another customer's session answered a channel naming nobody")
}

// The customer a channel naming nobody has its last calls of answers it with its session, even
// with another customer's session newer.
func (s *MessageHooksSuite) TestAChannelNamingNobodyIsAnsweredByTheSessionOfTheCustomerItsCallsAre() {
	owner := s.customerID()
	s.ran(s.holds())
	ownSession, own := s.dispatching(), s.workerOf(owner)
	s.useApp(s.data.createApp())
	s.dispatching()
	other := s.workerOf(s.customerID())

	s.Require().Equal(http.StatusOK, s.wrote(""))

	s.Equal(ownSession, s.handed(own).SessionID)
	s.nothingReaches(other, "another customer's session answered a channel naming nobody")
}

// A support chat names the config answering in it and holds no call. Its message reaches no
// other customer's session, but the worker of the customer holding the config.
func (s *MessageHooksSuite) TestAChannelNamingNobodyButAConfigIsAnsweredOnlyByTheCustomerHoldingItWhileItsOwnIsNotRunning() {
	owner, configID := s.customerID(), s.holds()
	own := s.workerOf(owner)
	s.useApp(s.data.createApp())
	s.dispatching()
	other := s.workerOf(s.customerID())

	s.Require().Equal(http.StatusOK, s.wrote(fmt.Sprintf(`{%q: %q}`, ConfigField, configID)))

	select {
	case m := <-other:
		s.Failf("another customer's session answered", "session %s of the other customer was handed the message", m.SessionID)
		return
	case <-time.After(3 * time.Second):
	}
	handed := s.handed(own)
	s.Empty(handed.SessionID)
	s.Equal(configID, handed.ConfigID)
	s.nothingReaches(other, "another customer's session answered a channel naming the owner's config")
}

// The session of the customer holding the config answers, with another customer's newer.
func (s *MessageHooksSuite) TestAChannelNamingNobodyButAConfigIsAnsweredByTheSessionOfTheCustomerHoldingIt() {
	owner, configID := s.customerID(), s.holds()
	ownSession, own := s.dispatching(), s.workerOf(owner)
	s.useApp(s.data.createApp())
	s.dispatching()
	other := s.workerOf(s.customerID())

	s.Require().Equal(http.StatusOK, s.wrote(fmt.Sprintf(`{%q: %q}`, ConfigField, configID)))

	select {
	case m := <-other:
		s.Failf("another customer's session answered", "session %s of the other customer was handed the message", m.SessionID)
		return
	case <-time.After(3 * time.Second):
	}
	s.Equal(ownSession, s.handed(own).SessionID)
}

// A channel naming nobody whose calls are two customers' is answered by none, the session
// running there included.
func (s *MessageHooksSuite) TestAChannelNamingNobodyWhoseCallsAreTwoCustomersIsAnsweredByNone() {
	s.ran(s.holds())
	own := s.workerOf(s.customerID())
	s.useApp(s.data.createApp())
	s.ran(s.holds())
	s.dispatching()
	other := s.workerOf(s.customerID())

	s.Require().Equal(http.StatusOK, s.wrote(""))

	s.nothingReaches(own, "a channel whose calls are two customers' was answered")
	s.nothingReaches(other, "another customer's session answered a channel naming nobody")
	s.Contains(s.logged.String(), `msg="not answering a message in a channel more than one customer's sessions run on" channel=`+s.channelID)
}

// One customer's two sessions on a channel are no ambiguity: the newest answers, which is
// the one a person writing there is watching, whether or not the channel names its customer.
func (s *MessageHooksSuite) TestAChannelsMessageReachesTheNewestOfItsCustomersSessions() {
	worker := s.workerOf(s.customerID())
	for _, custom := range []string{"", s.ownedBy(s.customerID())} {
		s.channelID = "chat-" + s.utils.uuid()
		s.dispatching()
		newest := s.dispatching()

		s.Require().Equal(http.StatusOK, s.wrote(custom))

		s.Equal(newest, s.handed(worker).SessionID, "channel custom %s", custom)
	}
}

// dispatching opens a session for the app's backend on the suite's channel, whose agent
// leaves what is written to the customer's dispatch worker, and returns its id. It is held
// off the record, so nothing but the session itself ties it to the channel.
func (s *MessageHooksSuite) dispatching() string {
	config := store.AgentConfig{CustomerID: s.customerID(), Name: "config-" + s.utils.uuid(),
		Mode: store.AgentModeText, LLM: "en-low-latency", DispatchText: true}
	s.Require().NoError(s.store.CreateAgentConfig(context.Background(), &config))
	return s.serverClient.createSession(CreateSessionRequest{
		AgentId: &s.channelID, Agent: &config.Name, Incognito: pointerTo(true),
	}).Id
}

// workerOf connects a dispatch worker for a customer and returns what it is handed.
func (s *MessageHooksSuite) workerOf(customer string) <-chan dispatch.Message {
	worker, release := s.dispatch.Register(customer, dispatch.Registration{Capacity: 4})
	s.T().Cleanup(release)
	return worker.Messages()
}

// ownedBy is the custom data of a channel the router created for a customer.
func (s *MessageHooksSuite) ownedBy(customer string) string {
	return fmt.Sprintf(`{%q: %q}`, conversation.CustomerField, customer)
}

// handed is the next message a worker was handed.
func (s *MessageHooksSuite) handed(messages <-chan dispatch.Message) dispatch.Message {
	select {
	case message := <-messages:
		return message
	case <-time.After(settleFor):
		s.FailNow("the worker was handed nothing")
		return dispatch.Message{}
	}
}

// holds stores an agent config for the suite's app and returns its id.
func (s *MessageHooksSuite) holds() string {
	config := store.AgentConfig{
		CustomerID: s.customerID(), Name: "config-" + s.utils.uuid(), Mode: "text",
	}
	s.Require().NoError(s.store.CreateAgentConfig(context.Background(), &config))
	return config.ID
}

// ran records a conversation having been held on the channel, which is what a call leaves
// behind.
func (s *MessageHooksSuite) ran(configID string) {
	s.Require().NoError(s.store.StartCall(context.Background(), &store.Call{
		ID:         s.utils.uuid(),
		CustomerID: s.customerID(),
		CallID:     s.utils.callID(),
		AgentID:    s.channelID,
		ConfigID:   configID,
		StartedAt:  time.Now().UTC(),
	}))
}

// wrote delivers a signed message.new for the channel, as Stream would. The custom data is
// whatever the channel was created with.
func (s *MessageHooksSuite) wrote(custom string) int {
	return s.signedly("/v1/chat/hooks/stream", s.writing(custom))
}

func (s *MessageHooksSuite) writing(custom string) string {
	if custom == "" {
		custom = "{}"
	}
	return fmt.Sprintf(`{
  "type": "message.new",
  "cid": "agent:%s",
  "channel_id": "%s",
  "channel_type": "agent",
  "channel_custom": %s,
  "message": {
    "id": "message-1",
    "text": "does the react sdk retry a failed upload?",
    "user": {"id": "sam", "name": "Sam"}
  }
}`, s.channelID, s.channelID, custom)
}

// nothingReaches fails when a message arrives on a channel that should stay empty.
func (s *MessageHooksSuite) nothingReaches(messages <-chan dispatch.Message, complaint string) {
	select {
	case message := <-messages:
		s.Failf(complaint, "%s reached a worker", message.ChannelID)
	case <-time.After(dropped):
	}
}
