//go:build integration

package api

import (
	"context"
	"crypto/ed25519"
	"errors"
	"fmt"
	"log/slog"
	"net/http"
	"net/http/httptest"
	"os"
	"strconv"
	"strings"
	"sync/atomic"
	"testing"
	"time"

	"github.com/GetStream/Vision-Agents/acceleration/internal/auth"
	"github.com/GetStream/Vision-Agents/acceleration/internal/chatinbound"
	"github.com/GetStream/Vision-Agents/acceleration/internal/chatlog"
	"github.com/GetStream/Vision-Agents/acceleration/internal/dispatch"
	"github.com/GetStream/Vision-Agents/acceleration/internal/routing"
	"github.com/GetStream/Vision-Agents/acceleration/internal/store"
	"github.com/GetStream/Vision-Agents/acceleration/internal/streamapp"
)

// ChatInboundSuite is Stream Chat delivering the messages written in agent channels to a
// router in deployment mode, which queues them for its worker to answer as it answers the
// message webhook. What the endpoint refuses is ChatInboundRefusalSuite's. A database of its
// own, as the worker claims every row in it.
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
	s.database = "chatinbound"
	s.RouterSuite.SetupSuite()
	s.wake(&s.RouterSuite)
}

func (s *ChatInboundSuite) SetupTest() {
	s.useApp(s.data.createApp())
	s.channelID = "chat-" + s.utils.uuid()
	s.answering(&s.RouterSuite)
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
	s.Zero(s.queued(&s.RouterSuite))
}

// Everything the agent stores carries a source, and answering it is the agent talking to
// itself, as the webhook knows (addressed).
func (s *ChatInboundSuite) TestAMessageTheRouterWroteItselfReachesNoWorker() {
	worker, release := s.dispatch.Register(s.customerID(), dispatch.Registration{Capacity: 1})
	defer release()

	s.Equal(http.StatusOK, s.delivers(&s.RouterSuite, s.message(suiteStreamApp, s.config(s.customerID()), `"agent"`), s.current))

	s.False(s.reachesWorker(worker.Messages()))
	s.Zero(s.queued(&s.RouterSuite))
}

// Only a new message is answered: an edit of one carries the same message, already answered.
func (s *ChatInboundSuite) TestAnEventThatIsNotANewMessageReachesNoWorker() {
	worker, release := s.dispatch.Register(s.customerID(), dispatch.Registration{Capacity: 1})
	defer release()
	body := strings.Replace(s.message(suiteStreamApp, s.config(s.customerID()), ""), `"message.new"`, `"message.updated"`, 1)

	s.Equal(http.StatusOK, s.delivers(&s.RouterSuite, body, s.current))

	s.False(s.reachesWorker(worker.Messages()))
	s.Zero(s.queued(&s.RouterSuite))
}

// The 200 comes once the row is committed, so a router that stops right after it leaves the
// message to the next worker to start, which answers it although the delivery was recorded.
func (s *ChatInboundSuite) TestADeliveryTakenBeforeTheRouterStoppedIsAnsweredOnceAWorkerStarts() {
	worker, release := s.dispatch.Register(s.customerID(), dispatch.Registration{Capacity: 2})
	defer release()
	s.worker.Load().Close()

	s.Require().Equal(http.StatusOK, s.delivers(&s.RouterSuite, s.message(suiteStreamApp, s.config(s.customerID()), ""), s.current))
	s.Require().False(s.reachesWorker(worker.Messages()), "answered with no worker running")
	s.answering(&s.RouterSuite)

	s.True(s.reachesWorker(worker.Messages()), "the message taken before the stop was never answered")
	s.False(s.reachesWorker(worker.Messages()), "one delivery is one answer")
}

// Stream Chat makes a delivery again, the same signed request, when it did not hear back in
// time. Deployment mode answers it once, as app mode does.
func (s *ChatInboundSuite) TestADeliveryMadeAgainIsAnsweredOnce() {
	worker, release := s.dispatch.Register(s.customerID(), dispatch.Registration{Capacity: 2})
	defer release()
	body := s.message(suiteStreamApp, s.config(s.customerID()), "")
	signed := signDelivery(s.current, "msg_"+s.utils.uuid(), time.Now(), body)

	s.Equal(http.StatusOK, s.deliversTo(&s.RouterSuite, s.server.URL, body, signed))
	s.Equal(http.StatusOK, s.deliversTo(&s.RouterSuite, s.server.URL, body, signed))

	s.True(s.reachesWorker(worker.Messages()))
	s.False(s.reachesWorker(worker.Messages()), "the delivery made again was answered too")
	s.Equal(1, s.queued(&s.RouterSuite))
}

// An idle worker looks at the queue once a minute; the endpoint wakes it, so a message to a
// router with nothing queued is answered at once, not up to a minute on.
func (s *ChatInboundSuite) TestADeliveryToAnIdleRouterIsAnsweredAtOnce() {
	worker, release := s.dispatch.Register(s.customerID(), dispatch.Registration{Capacity: 2})
	defer release()
	s.Require().Equal(http.StatusOK, s.delivers(&s.RouterSuite, s.message(suiteStreamApp, s.config(s.customerID()), ""), s.current))
	s.Require().True(s.reachesWorker(worker.Messages()))
	s.Require().False(s.reachesWorker(worker.Messages()), "the worker is idle again")

	s.Require().Equal(http.StatusOK, s.delivers(&s.RouterSuite, s.message(suiteStreamApp, s.config(s.customerID()), ""), s.current))

	select {
	case <-worker.Messages():
	case <-time.After(settleFor):
		s.Fail("the message waited for the worker's next look")
	}
}

// A delivery the router could not queue, its database down, is a 503, so Stream Chat makes it
// again, and the delivery made again is answered.
func (s *ChatInboundSuite) TestADeliveryThatCouldNotBeQueuedIsMadeAgainAndAnswered() {
	worker, release := s.dispatch.Register(s.customerID(), dispatch.Registration{Capacity: 2})
	defer release()
	body := s.message(suiteStreamApp, s.config(s.customerID()), "")
	signed := signDelivery(s.current, "msg_"+s.utils.uuid(), time.Now(), body)

	s.Equal(http.StatusServiceUnavailable, s.deliversTo(&s.RouterSuite, s.routerOnAClosedStore(), body, signed))
	s.False(s.reachesWorker(worker.Messages()))

	s.Equal(http.StatusOK, s.deliversTo(&s.RouterSuite, s.server.URL, body, signed), "the delivery made again")
	s.True(s.reachesWorker(worker.Messages()))
}

// While both are switched on, the webhook and the delivery carry the same message, whichever
// comes first, and one record of it answers it once.
func (s *ChatInboundSuite) TestAMessageTheWebhookBroughtIsNotQueuedWhenDelivered() {
	worker, release := s.dispatch.Register(s.customerID(), dispatch.Registration{Capacity: 2})
	defer release()
	body := s.message(suiteStreamApp, s.config(s.customerID()), "")

	s.Require().Equal(http.StatusOK, s.hooked(&s.RouterSuite, "/v1/chat/hooks/stream", body, suiteStreamSecret))
	s.Equal(http.StatusOK, s.delivers(&s.RouterSuite, body, s.current))

	s.True(s.reachesWorker(worker.Messages()))
	s.False(s.reachesWorker(worker.Messages()), "the delivery is the message the webhook brought")
	s.Zero(s.queued(&s.RouterSuite))
}

func (s *ChatInboundSuite) TestTheWebhookOfAMessageDeliveredFirstIsNotActedOn() {
	worker, release := s.dispatch.Register(s.customerID(), dispatch.Registration{Capacity: 2})
	defer release()
	body := s.message(suiteStreamApp, s.config(s.customerID()), "")

	s.Require().Equal(http.StatusOK, s.delivers(&s.RouterSuite, body, s.current))
	s.Equal(http.StatusOK, s.hooked(&s.RouterSuite, "/v1/chat/hooks/stream", body, suiteStreamSecret))

	s.True(s.reachesWorker(worker.Messages()))
	s.False(s.reachesWorker(worker.Messages()), "the webhook is the message the delivery brought")
}

// The node a delivery lands on queues it, and its worker hands it to the node with the
// dispatch worker.
func (s *ChatInboundSuite) TestADeliveryCarriedToAnotherNodesWorkerIsAnsweredThere() {
	other := s.otherNodeDispatching(dispatch.NewPool())
	worker := s.serverClient.on(other).opens("/v1/dispatch")
	s.await(worker, "ready")

	s.Equal(http.StatusOK, s.delivers(&s.RouterSuite, s.message(suiteStreamApp, s.config(s.customerID()), ""), s.current))

	s.Equal(s.channelID, s.await(worker, "message")["channel_id"])
}

// A router that has not learned its own app yet makes Stream deliver the webhook again, as the
// endpoint does: recorded under no app, the webhook would not be the delivery the endpoint
// records under the app once learned, and both would answer the message.
func (s *ChatInboundSuite) TestAWebhookBeforeTheRouterKnowsItsAppIsMadeAgainAndAnsweredOnce() {
	router, deployment := s.routerLearningItsApp(s.chatInboundKeys)
	body := s.message(suiteStreamApp, s.config(s.customerID()), "")

	s.Equal(http.StatusServiceUnavailable, s.hookedAt(router, body))
	deployment.SetApp(suiteStreamApp)
	s.Require().Equal(http.StatusOK, s.hookedAt(router, body))
	s.Equal(http.StatusOK, s.deliversTo(&s.RouterSuite, router, body, signDelivery(s.current, "msg_"+s.utils.uuid(), time.Now(), body)))

	s.Zero(s.queued(&s.RouterSuite), "the delivery is the message the webhook brought")
}

// Only a message written to an agent waits for the app: Stream delivers every message in the app
// to the webhook, and the rest are not delivered again.
func (s *ChatInboundSuite) TestAMessageTheRouterWroteItselfIsTakenBeforeTheRouterKnowsItsApp() {
	router, _ := s.routerLearningItsApp(s.chatInboundKeys)

	s.Equal(http.StatusOK, s.hookedAt(router, s.message(suiteStreamApp, s.config(s.customerID()), `"agent"`)))
}

// Without Stream Chat's deliveries nothing records a message in deployment mode, so the webhook
// is acted on before the router knows its app, as it always was.
func (s *ChatInboundSuite) TestWithoutStreamChatDeliveriesAWebhookBeforeTheRouterKnowsItsAppIsTaken() {
	router, _ := s.routerLearningItsApp(nil)

	s.Equal(http.StatusOK, s.hookedAt(router, s.message(suiteStreamApp, s.config(s.customerID()), "")))
}

// A deployment that can never learn its app has nothing for the endpoint to wait for, so the
// webhook is taken at once and answered 200: a 503 would have Stream deliver it again for ever. Two
// deployments cannot learn it: one with no Stream credentials, and one whose configured app id
// Stream says is another.
func (s *ChatInboundSuite) TestAWebhookToADeploymentThatCannotLearnItsAppIsTaken() {
	unconfigured := streamapp.NewDeployment(streamapp.DeploymentOptions{BaseURL: s.chat.URL})
	wrong := streamapp.NewDeployment(streamapp.DeploymentOptions{APIKey: suiteStreamKey, Secret: suiteStreamSecret, BaseURL: s.chat.URL, App: 111})
	wrong.SetApp(222)
	s.Require().Zero(wrong.App(), "Stream told the deployment an app other than the configured one")

	for name, deployment := range map[string]*streamapp.Deployment{"no credentials": unconfigured, "wrong app id": wrong} {
		s.Run(name, func() {
			router := s.routerIn(deployment, s.chatInboundKeys)

			s.Equal(http.StatusOK, s.hookedAt(router, s.message(suiteStreamApp, s.config(s.customerID()), "")))
		})
	}
}

// A peer recorded the webhook it hands over, so a node that has not learned its app yet takes
// it rather than have Stream deliver it again.
func (s *ChatInboundSuite) TestAWebhookHandedToANodeThatDoesNotKnowItsAppIsAnsweredThere() {
	warm := s.stream
	s.stream, _ = learningItsApp(s.chat.URL)
	other := s.otherNodeDispatching(dispatch.NewPool())
	s.stream = warm
	worker := s.serverClient.on(other).opens("/v1/dispatch")
	s.await(worker, "ready")

	s.Equal(http.StatusOK, s.hooked(&s.RouterSuite, "/v1/chat/hooks/stream", s.message(suiteStreamApp, s.config(s.customerID()), ""), suiteStreamSecret))

	s.Equal(s.channelID, s.await(worker, "message")["channel_id"])
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
	s.database = "chatinboundapp"
	s.RouterSuite.SetupSuite()
	s.wake(&s.RouterSuite)
}

func (s *ChatInboundAppSuite) SetupTest() {
	s.useApp(s.numberedApp(streamAppID()))
	s.apiKey = "own-key-" + s.utils.uuid()
	s.answerAs(s.apiKey, s.appID())
	s.forwarded(s.apiKey)
	s.channelID = "chat-" + s.utils.uuid()
	s.answering(&s.RouterSuite)
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

func (s *ChatInboundAppSuite) TestTheWebhookOfAMessageDeliveredFirstIsNotActedOn() {
	worker, release := s.dispatch.Register(s.customerID(), dispatch.Registration{Capacity: 2})
	defer release()
	body := s.message(s.appID(), s.config(s.customerID()), "")

	s.Require().Equal(http.StatusOK, s.delivers(&s.RouterSuite, body, s.current))
	hook := s.request(chatMessageHookPathFor(s.appID()), body, sign(body, ownAppSecret))
	vouched(hook, s.appID(), s.apiKey)
	response, err := s.server.Client().Do(hook)
	s.Require().NoError(err)
	s.Require().NoError(response.Body.Close())
	s.Equal(http.StatusOK, response.StatusCode)

	s.True(s.reachesWorker(worker.Messages()))
	s.False(s.reachesWorker(worker.Messages()), "the webhook is the message the delivery brought")
}

// A message made longer ago than a hook is acted on is no message to answer now, as the
// app's webhook knows (acting), however fresh Stream's signature on it is.
func (s *ChatInboundAppSuite) TestAStaleMessageIsNotQueued() {
	worker, release := s.dispatch.Register(s.customerID(), dispatch.Registration{Capacity: 1})
	defer release()
	body := s.messageAt(s.appID(), s.config(s.customerID()), "", time.Now().Add(-hookStaleAfter-time.Minute))

	s.Equal(http.StatusOK, s.delivers(&s.RouterSuite, body, s.current))

	s.False(s.reachesWorker(worker.Messages()))
	s.Zero(s.queued(&s.RouterSuite))
}

// chatInbounds is what the delivery suites share: Stream's keys, the channel written in, and
// the messages Stream signs.
type chatInbounds struct {
	current, previous ed25519.PrivateKey
	// channelID is the channel written in, unique, and not named support-<uuid>, the
	// namespace durable session commands own.
	channelID string
	// worker is the queue's worker answering what the suite's router queues (answering).
	worker atomic.Pointer[chatinbound.Worker]
}

// keys has the suite's router take deliveries signed with either of Stream's two keys.
func (d *chatInbounds) keys(router *RouterSuite) {
	d.current = ed25519.NewKeyFromSeed([]byte("the-current-delivery-key-seed-32"))
	d.previous = ed25519.NewKeyFromSeed([]byte("the-previous-delivery-key-seed32"))
	router.chatInboundKeys = []ed25519.PublicKey{public(d.current), public(d.previous)}
}

// wake has the suite's router wake the worker answering now, as cmd/router has it wake its
// own.
func (d *chatInbounds) wake(router *RouterSuite) {
	router.router.SetChatInboundKick(func() {
		if worker := d.worker.Load(); worker != nil {
			worker.Kick()
		}
	})
}

// answering runs a queue worker on the suite's router until the test ends, trying a delivery
// again an hour on, after the test. Once it has looked, it looks again only when woken, or a
// minute on.
func (d *chatInbounds) answering(router *RouterSuite) {
	worker, err := chatinbound.NewWorker(chatinbound.Options{
		Store: router.store, Answer: router.router.AnswerChatInbound, Poll: 20 * time.Millisecond,
		Retries: []time.Duration{time.Hour},
	})
	router.Require().NoError(err)
	worker.Start()
	d.worker.Store(worker)
	router.T().Cleanup(worker.Close)
}

// delivers posts a body as Stream Chat delivers it, signed with key, and answers the status.
func (d *chatInbounds) delivers(router *RouterSuite, body string, key ed25519.PrivateKey) int {
	return d.deliversTo(router, router.server.URL, body, signDelivery(key, "msg_"+router.utils.uuid(), time.Now(), body))
}

// deliversTo posts a body with the headers Stream Chat signed it with to the router at url, and
// answers the status.
func (d *chatInbounds) deliversTo(router *RouterSuite, url, body string, signed http.Header) int {
	request, err := http.NewRequest(http.MethodPost, url+chatInboundPath, strings.NewReader(body))
	router.Require().NoError(err)
	request.Header = signed.Clone()
	request.Header.Set("Content-Type", "application/json")
	response, err := router.server.Client().Do(request)
	router.Require().NoError(err)
	router.Require().NoError(response.Body.Close())
	return response.StatusCode
}

// hooked posts body to the message webhook at path, signed with secret, and answers the status.
func (d *chatInbounds) hooked(router *RouterSuite, path, body, secret string) int {
	response, err := router.server.Client().Do(router.request(path, body, sign(body, secret)))
	router.Require().NoError(err)
	router.Require().NoError(response.Body.Close())
	return response.StatusCode
}

// queued is how many rows the endpoint queued for the channel.
func (d *chatInbounds) queued(router *RouterSuite) int {
	count, err := router.store.DB().NewSelect().Table("chat_inbound").
		Where("position(? in body) > 0", []byte(fmt.Sprintf("%q", d.channelID))).Count(context.Background())
	router.Require().NoError(err)
	return count
}

// routerOnAClosedStore is a router in the suite's deployment, taking Stream Chat's deliveries,
// whose database is down, and returns its URL.
func (s *ChatInboundSuite) routerOnAClosedStore() string {
	closed, err := store.Open(testDatabase(s.T(), os.Getenv("ROUTER_POSTGRES_DSN")))
	s.Require().NoError(err)
	s.Require().NoError(closed.Close())
	server, err := NewServer(Options{
		Routers:  map[routing.Modality]routing.Inspector{routing.LLM: idleInspector{}},
		AuthMode: auth.Custom,
		Auth: auth.Func(func(context.Context, *http.Request) (auth.Principal, error) {
			return auth.Principal{}, errors.New("nobody signs in here")
		}),
		Store:           closed,
		Stream:          s.stream,
		HookSecret:      suiteStreamSecret,
		ChatInboundKeys: s.chatInboundKeys,
		Logger:          slog.New(slog.DiscardHandler),
	})
	s.Require().NoError(err)
	router := httptest.NewServer(server.Handler())
	s.T().Cleanup(router.Close)
	return router.URL
}

// routerLearningItsApp is a router in the suite's deployment, taking Stream Chat's deliveries
// signed with keys, that has not learned its own app yet, and returns its URL and the deployment
// to tell it.
func (s *ChatInboundSuite) routerLearningItsApp(keys []ed25519.PublicKey) (string, *streamapp.Deployment) {
	stream, deployment := learningItsApp(s.chat.URL)
	return s.routerOver(stream, keys), deployment
}

// routerIn is a router in deployment, taking Stream Chat's deliveries signed with keys, and
// returns its URL.
func (s *ChatInboundSuite) routerIn(deployment *streamapp.Deployment, keys []ed25519.PublicKey) string {
	return s.routerOver(streamapp.NewClients(deployment, streamapp.ClientsOptions{}), keys)
}

func (s *ChatInboundSuite) routerOver(stream *streamapp.Clients, keys []ed25519.PublicKey) string {
	server, err := NewServer(Options{
		Routers:  map[routing.Modality]routing.Inspector{routing.LLM: idleInspector{}},
		AuthMode: auth.Custom,
		Auth: auth.Func(func(context.Context, *http.Request) (auth.Principal, error) {
			return auth.Principal{}, errors.New("nobody signs in here")
		}),
		Store:           s.store,
		Stream:          stream,
		HookSecret:      suiteStreamSecret,
		ChatInboundKeys: keys,
		Logger:          slog.New(slog.DiscardHandler),
	})
	s.Require().NoError(err)
	router := httptest.NewServer(server.Handler())
	s.T().Cleanup(router.Close)
	return router.URL
}

// learningItsApp is the suite's deployment, configured with no app id and not told it yet.
func learningItsApp(chatURL string) (*streamapp.Clients, *streamapp.Deployment) {
	deployment := streamapp.NewDeployment(streamapp.DeploymentOptions{APIKey: suiteStreamKey, Secret: suiteStreamSecret, BaseURL: chatURL})
	return streamapp.NewClients(deployment, streamapp.ClientsOptions{}), deployment
}

// hookedAt posts body to the message webhook of the router at url, signed with the suite's
// secret, and answers the status.
func (s *ChatInboundSuite) hookedAt(url, body string) int {
	request, err := http.NewRequest(http.MethodPost, url+"/v1/chat/hooks/stream", strings.NewReader(body))
	s.Require().NoError(err)
	request.Header.Set("Content-Type", "application/json")
	request.Header.Set(signatureHeader, sign(body, suiteStreamSecret))
	response, err := s.server.Client().Do(request)
	s.Require().NoError(err)
	s.Require().NoError(response.Body.Close())
	return response.StatusCode
}

// message is a message.new written in the channel in app, which names configID. A source is
// what the agent marks everything it stores with, empty for a person's message.
func (d *chatInbounds) message(app int64, configID, source string) string {
	return d.messageAt(app, configID, source, time.Now())
}

// messageAt is message, made at at.
func (d *chatInbounds) messageAt(app int64, configID, source string, at time.Time) string {
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
}`, app, d.channelID, at.UTC().Format(time.RFC3339Nano), ConfigField, configID,
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
