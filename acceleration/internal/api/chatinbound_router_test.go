//go:build integration

package api

import (
	"context"
	"crypto/ed25519"
	"fmt"
	"net/http"
	"strconv"
	"strings"
	"testing"
	"time"

	"github.com/GetStream/Vision-Agents/acceleration/internal/chatlog"
	"github.com/GetStream/Vision-Agents/acceleration/internal/dispatch"
	"github.com/GetStream/Vision-Agents/acceleration/internal/node"
	"github.com/GetStream/Vision-Agents/acceleration/internal/store"
)

// ChatInboundSuite is Stream Chat delivering the messages written in agent channels to a
// router in deployment mode, which answers them as it answers the message webhook. What the
// endpoint refuses is ChatInboundRefusalSuite's.
type ChatInboundSuite struct {
	RouterSuite
	chatInbounds
}

func TestChatInboundSuite(t *testing.T) {
	runSuite(t, new(ChatInboundSuite))
}

func (s *ChatInboundSuite) SetupSuite() {
	s.chatInbounds.keys(&s.RouterSuite)
	s.deploymentKnown = true
	s.RouterSuite.SetupSuite()
}

func (s *ChatInboundSuite) SetupTest() {
	s.useApp(s.data.createApp())
	s.channelID = "chat-" + s.utils.uuid()
}

func (s *ChatInboundSuite) TestADeliveryIsAnsweredOnce() {
	worker, release := s.dispatch.Register(s.customerID(), dispatch.Registration{Capacity: 2})
	defer release()
	configID := s.config(s.customerID())

	s.Require().Equal(http.StatusOK, s.delivers(&s.RouterSuite, s.message(suiteStreamApp, configID, ""), s.current))

	select {
	case message := <-worker.Messages():
		s.Equal(s.channelID, message.ChannelID)
		s.Equal(configID, message.ConfigID)
		s.Equal("sam", message.UserID)
		s.Equal("is my order on its way?", message.Text)
	case <-time.After(settleFor):
		s.Fail("the message never reached the worker")
	}
	s.False(s.reachesWorker(worker.Messages()), "one delivery is one answer")
}

func (s *ChatInboundSuite) TestADeliverySignedWithTheOldKeyIsAnswered() {
	worker, release := s.dispatch.Register(s.customerID(), dispatch.Registration{Capacity: 1})
	defer release()

	s.Require().Equal(http.StatusOK, s.delivers(&s.RouterSuite, s.message(suiteStreamApp, s.config(s.customerID()), ""), s.previous))

	s.True(s.reachesWorker(worker.Messages()))
}

func (s *ChatInboundSuite) TestARefusedDeliveryReachesNoWorker() {
	worker, release := s.dispatch.Register(s.customerID(), dispatch.Registration{Capacity: 1})
	defer release()
	stranger := ed25519.NewKeyFromSeed([]byte("somebody-else-s-key-seed-32bytes"))

	s.Equal(http.StatusUnauthorized, s.delivers(&s.RouterSuite, s.message(suiteStreamApp, s.config(s.customerID()), ""), stranger))

	s.False(s.reachesWorker(worker.Messages()))
}

// Everything the agent stores carries a source, and answering it is the agent talking to
// itself, as the webhook knows (addressed).
func (s *ChatInboundSuite) TestAMessageTheRouterWroteItselfReachesNoWorker() {
	worker, release := s.dispatch.Register(s.customerID(), dispatch.Registration{Capacity: 1})
	defer release()

	s.Equal(http.StatusOK, s.delivers(&s.RouterSuite, s.message(suiteStreamApp, s.config(s.customerID()), `"agent"`), s.current))

	s.False(s.reachesWorker(worker.Messages()))
}

// Only a new message is answered: an edit of one carries the same message, already answered.
func (s *ChatInboundSuite) TestAnEventThatIsNotANewMessageReachesNoWorker() {
	worker, release := s.dispatch.Register(s.customerID(), dispatch.Registration{Capacity: 1})
	defer release()
	body := strings.Replace(s.message(suiteStreamApp, s.config(s.customerID()), ""), `"message.new"`, `"message.updated"`, 1)

	s.Equal(http.StatusOK, s.delivers(&s.RouterSuite, body, s.current))

	s.False(s.reachesWorker(worker.Messages()))
}

// The node a delivery lands on hands it over whole, headers and all, and the node with the
// worker checks Stream's signature for itself.
func (s *ChatInboundSuite) TestADeliveryCarriedToAnotherNodesWorkerIsAnsweredThere() {
	other := s.otherNodeDispatching(dispatch.NewPool())
	worker := s.serverClient.on(other).opens("/v1/dispatch")
	s.await(worker, "ready")

	s.Equal(http.StatusOK, s.delivers(&s.RouterSuite, s.message(suiteStreamApp, s.config(s.customerID()), ""), s.current))

	s.Equal(s.channelID, s.await(worker, "message")["channel_id"])
}

// The forwarded header is only a peer's to set: one that came with a delivery from outside
// keeps it on this node rather than hand it to the node running the conversation.
func (s *ChatInboundSuite) TestADeliveryCarryingThePeersHeaderFromOutsideIsNotHandedOn() {
	other := s.otherNode()
	s.serverClient.on(other).createSession(CreateSessionRequest{
		AgentId: &s.channelID, Llm: pointerTo("vision/vision-model"), Incognito: pointerTo(true),
	})
	worker, release := s.dispatch.Register(s.customerID(), dispatch.Registration{Capacity: 1})
	defer release()
	body := s.message(suiteStreamApp, s.config(s.customerID()), "")
	request, err := http.NewRequest(http.MethodPost, s.server.URL+chatInboundPath, strings.NewReader(body))
	s.Require().NoError(err)
	request.Header = signDelivery(s.current, "msg_"+s.utils.uuid(), time.Now(), body)
	request.Header.Set(node.ForwardedHeader, "1")
	response, err := s.server.Client().Do(request)
	s.Require().NoError(err)
	s.Require().NoError(response.Body.Close())

	s.Equal(http.StatusOK, response.StatusCode)
	s.True(s.reachesWorker(worker.Messages()))
	s.Empty(s.vision.requests(), "the node running the conversation was handed it")
}

// ChatInboundAppSuite is Stream Chat delivering to a router in app mode, where a delivery
// names a customer's own app and no gateway vouches for it: Stream's signature over the app
// id is what says whose it is.
type ChatInboundAppSuite struct {
	RouterSuite
	chatInbounds
	apiKey string
}

func TestChatInboundAppSuite(t *testing.T) {
	runSuite(t, new(ChatInboundAppSuite))
}

func (s *ChatInboundAppSuite) SetupSuite() {
	s.chatInbounds.keys(&s.RouterSuite)
	s.appMode = true
	s.RouterSuite.SetupSuite()
}

func (s *ChatInboundAppSuite) SetupTest() {
	s.useApp(s.numberedApp(streamAppID()))
	s.apiKey = "own-key-" + s.utils.uuid()
	s.answerAs(s.apiKey, s.appID())
	s.forwarded(s.apiKey)
	s.channelID = "chat-" + s.utils.uuid()
}

func (s *ChatInboundAppSuite) TestADeliveryInACustomersOwnAppIsAnsweredThere() {
	worker, release := s.dispatch.Register(s.customerID(), dispatch.Registration{Capacity: 1})
	defer release()

	s.Equal(http.StatusOK, s.delivers(&s.RouterSuite, s.message(s.appID(), s.config(s.customerID()), ""), s.current))

	s.True(s.reachesWorker(worker.Messages()))
}

// A channel in this app naming another customer's config starts nobody's agent: the app the
// delivery names is the only one acted in.
func (s *ChatInboundAppSuite) TestADeliveryIsActedOnOnlyInItsOwnApp() {
	other := "someone-" + s.utils.uuid()
	somebodyElse, release := s.dispatch.Register(other, dispatch.Registration{Capacity: 1})
	defer release()

	s.Equal(http.StatusOK, s.delivers(&s.RouterSuite, s.message(s.appID(), s.config(other), ""), s.current))

	s.False(s.reachesWorker(somebodyElse.Messages()))
}

// A call row from before rows named an app, another customer's, on a channel of the same
// name is not this app's: the delivery is answered by the config its channel names, as the
// app's own webhook is, and not by whoever left that row.
func (s *ChatInboundAppSuite) TestAnUnpinnedCallOnTheSameChannelIsNotTheDeliverysOwner() {
	own, release := s.dispatch.Register(s.customerID(), dispatch.Registration{Capacity: 1})
	defer release()
	other := "someone-" + s.utils.uuid()
	somebodyElse, releaseOther := s.dispatch.Register(other, dispatch.Registration{Capacity: 1})
	defer releaseOther()
	s.Require().NoError(s.store.StartCall(context.Background(), &store.Call{
		ID: s.utils.uuid(), CustomerID: other, CallID: s.utils.callID(), AgentID: s.channelID,
		ConfigID: s.config(other), StartedAt: time.Now().UTC(),
	}))

	s.Equal(http.StatusOK, s.delivers(&s.RouterSuite, s.message(s.appID(), s.config(s.customerID()), ""), s.current))

	s.True(s.reachesWorker(own.Messages()))
	s.False(s.reachesWorker(somebodyElse.Messages()))
}

// While both are switched on, the webhook and the delivery carry the same message, and the
// app's record of what it has acted on answers it once.
func (s *ChatInboundAppSuite) TestTheWebhookAndADeliveryOfOneMessageAreAnsweredOnce() {
	worker, release := s.dispatch.Register(s.customerID(), dispatch.Registration{Capacity: 2})
	defer release()
	body := s.message(s.appID(), s.config(s.customerID()), "")
	hook := s.request(chatMessageHookPathFor(s.appID()), body, sign(body, ownAppSecret))
	vouched(hook, s.appID(), s.apiKey)
	response, err := s.server.Client().Do(hook)
	s.Require().NoError(err)
	s.Require().NoError(response.Body.Close())
	s.Require().Equal(http.StatusOK, response.StatusCode)

	s.Equal(http.StatusOK, s.delivers(&s.RouterSuite, body, s.current))

	s.True(s.reachesWorker(worker.Messages()))
	s.False(s.reachesWorker(worker.Messages()), "the delivery is the message the webhook brought")
}

// chatInbounds is what the delivery suites share: Stream's keys, the channel written in, and
// the messages Stream signs.
type chatInbounds struct {
	current, previous ed25519.PrivateKey
	// channelID is the channel written in, unique, and not named support-<uuid>, the
	// namespace durable session commands own.
	channelID string
}

// keys has the suite's router take deliveries signed with either of Stream's two keys.
func (d *chatInbounds) keys(router *RouterSuite) {
	d.current = ed25519.NewKeyFromSeed([]byte("the-current-delivery-key-seed-32"))
	d.previous = ed25519.NewKeyFromSeed([]byte("the-previous-delivery-key-seed32"))
	router.chatInboundKeys = []ed25519.PublicKey{public(d.current), public(d.previous)}
}

// delivers posts a body as Stream Chat delivers it, signed with key, and answers the status.
func (d *chatInbounds) delivers(router *RouterSuite, body string, key ed25519.PrivateKey) int {
	request, err := http.NewRequest(http.MethodPost, router.server.URL+chatInboundPath, strings.NewReader(body))
	router.Require().NoError(err)
	request.Header = signDelivery(key, "msg_"+router.utils.uuid(), time.Now(), body)
	request.Header.Set("Content-Type", "application/json")
	response, err := router.server.Client().Do(request)
	router.Require().NoError(err)
	router.Require().NoError(response.Body.Close())
	return response.StatusCode
}

// message is a message.new written in the channel in app, which names configID. A source is
// what the agent marks everything it stores with, empty for a person's message.
func (d *chatInbounds) message(app int64, configID, source string) string {
	marked := ""
	if source != "" {
		marked = fmt.Sprintf(`, %q: %s`, chatlog.SourceField, source)
	}
	return fmt.Sprintf(`{
  "type": "message.new",
  "app_id": %d,
  "channel_id": %q,
  "channel_type": "agent",
  "created_at": %q,
  "channel_custom": {%q: %q},
  "message": {"id": %q, "text": "is my order on its way?", "user": {"id": "sam", "name": "Sam"}%s}
}`, app, d.channelID, time.Now().UTC().Format(time.RFC3339Nano), ConfigField, configID,
		"message-"+strconv.FormatInt(time.Now().UnixNano(), 36), marked)
}

func (s *ChatInboundSuite) config(customer string) string {
	return deliveredConfig(&s.RouterSuite, customer)
}
func (s *ChatInboundAppSuite) config(customer string) string {
	return deliveredConfig(&s.RouterSuite, customer)
}

func deliveredConfig(router *RouterSuite, customer string) string {
	config := store.AgentConfig{CustomerID: customer, Name: "config-" + router.utils.uuid(), Mode: "text"}
	router.Require().NoError(router.store.CreateAgentConfig(context.Background(), &config))
	return config.ID
}

// reachesWorker reports whether a message reached a worker before a delivery would have.
func (d *chatInbounds) reachesWorker(messages <-chan dispatch.Message) bool {
	select {
	case <-messages:
		return true
	case <-time.After(dropped):
		return false
	}
}

func chatMessageHookPathFor(app int64) string {
	return "/v1/chat/hooks/stream/" + strconv.FormatInt(app, 10)
}
