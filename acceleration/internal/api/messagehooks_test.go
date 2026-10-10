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
	"github.com/GetStream/Vision-Agents/acceleration/internal/dispatch"
	"github.com/GetStream/Vision-Agents/acceleration/internal/store"
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
