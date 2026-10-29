//go:build integration

package api

import (
	"context"
	"encoding/json"
	"log/slog"
	"net/http"
	"net/http/httptest"
	"strings"
	"sync"
	"testing"
	"time"

	"github.com/gorilla/websocket"

	"github.com/GetStream/Vision-Agents/acceleration/internal/chatdelivery"
	"github.com/GetStream/Vision-Agents/acceleration/internal/conversation"
	"github.com/GetStream/Vision-Agents/acceleration/internal/dispatch"
	"github.com/GetStream/Vision-Agents/acceleration/internal/node"
	"github.com/GetStream/Vision-Agents/acceleration/internal/store"
	"github.com/GetStream/Vision-Agents/acceleration/internal/streamapp"
)

// ChatDeliveryQueueSuite is the queue's worker answering the Stream Chat deliveries queued in
// its database, as the endpoint answers them on the request: here, or on the node answering in
// the channel. A database of its own, as the worker claims every row in it.
type ChatDeliveryQueueSuite struct {
	RouterSuite
	chatDeliveries
}

func TestChatDeliveryQueueSuite(t *testing.T) {
	runSuite(t, new(ChatDeliveryQueueSuite))
}

func (s *ChatDeliveryQueueSuite) SetupSuite() {
	s.chatDeliveries.keys(&s.RouterSuite)
	s.deploymentKnown = true
	s.database = "chatqueue"
	s.RouterSuite.SetupSuite()
}

func (s *ChatDeliveryQueueSuite) SetupTest() {
	s.useApp(s.data.createApp())
	s.channelID = "chat-" + s.utils.uuid()
}

func (s *ChatDeliveryQueueSuite) TestAQueuedDeliveryIsAnsweredOnce() {
	worker, release := s.dispatch.Register(s.customerID(), dispatch.Registration{Capacity: 2})
	defer release()
	queued := s.queue(s.message(suiteStreamApp, deliveredConfig(&s.RouterSuite, s.customerID()), ""))

	s.answering()

	select {
	case message := <-worker.Messages():
		s.Equal(s.channelID, message.ChannelID)
		s.Equal("sam", message.UserID)
	case <-time.After(settleFor):
		s.Fail("the queued message never reached the worker")
	}
	s.False(s.reachesWorker(worker.Messages()), "one queued delivery is one answer")
	s.Eventually(func() bool { return s.row(queued).State == store.ChatDeliveryDone }, settleFor, 20*time.Millisecond)
}

// A queued delivery's row is settled as its answer starts, so no other try answers it while the
// model writes, and the answer runs in line: the worker's slot, and Close, wait for it rather
// than a goroutine nobody waits for.
func (s *ChatDeliveryQueueSuite) TestAQueuedDeliveryIsSettledAsItsAnswerStartsAndReturnsOnceItHasEnded() {
	s.serverClient.createSession(CreateSessionRequest{
		AgentId: &s.channelID, Llm: pointerTo("vision/vision-model"), Incognito: pointerTo(true),
	})
	s.vision.holds()
	defer s.vision.answers()
	text := "where is order " + s.utils.uuid()
	queued := s.row(s.queue(strings.Replace(s.message(suiteStreamApp, "", ""), "is my order on its way?", text, 1)))

	ended := make(chan error, 1)
	go func() { ended <- s.router.AnswerChatDelivery(context.Background(), queued) }()

	s.Eventually(func() bool { return s.asked(text) == 1 }, settleFor, 20*time.Millisecond)
	s.Equal(store.ChatDeliveryDone, s.row(queued.MessageID).State, "settled as its answer started")
	s.Never(func() bool { return len(ended) > 0 }, dropped, 20*time.Millisecond, "returned before the answer ended")
	s.vision.answers()
	select {
	case err := <-ended:
		s.NoError(err)
	case <-time.After(settleFor):
		s.Fail("the answer never ended")
	}
}

// The signed request is gone by the time a worker claims the row, so the row goes over the
// peer path, and the node running the conversation answers it there, once.
func (s *ChatDeliveryQueueSuite) TestAQueuedDeliveryForASessionOnAnotherNodeIsAnsweredThereOnce() {
	other := s.otherNode()
	s.serverClient.on(other).createSession(CreateSessionRequest{
		AgentId: &s.channelID, Llm: pointerTo("vision/vision-model"), Incognito: pointerTo(true),
	})
	worker, release := s.dispatch.Register(s.customerID(), dispatch.Registration{Capacity: 1})
	defer release()
	text := "where is order " + s.utils.uuid()
	queued := s.queue(strings.Replace(s.message(suiteStreamApp, "", ""), "is my order on its way?", text, 1))

	s.answering()

	s.Eventually(func() bool { return s.asked(text) == 1 }, settleFor, 20*time.Millisecond,
		"the message never reached the conversation on the node running it")
	s.Eventually(func() bool { return s.row(queued).State == store.ChatDeliveryDone }, settleFor, 20*time.Millisecond)
	s.Never(func() bool { return s.asked(text) > 1 }, dropped, 20*time.Millisecond, "one queued delivery is one answer")
	s.False(s.reachesWorker(worker.Messages()), "a second agent was started")
}

// With no worker waiting here, the row goes to the node where one waits, and is placed there.
func (s *ChatDeliveryQueueSuite) TestAQueuedDeliveryReachesAWorkerWaitingOnAnotherNode() {
	other := s.otherNodeDispatching(dispatch.NewPool())
	worker := s.serverClient.on(other).opens("/v1/dispatch")
	s.await(worker, "ready")
	queued := s.queue(s.message(suiteStreamApp, deliveredConfig(&s.RouterSuite, s.customerID()), ""))

	s.answering()

	s.Equal(s.channelID, s.await(worker, "message")["channel_id"])
	s.Eventually(func() bool { return s.row(queued).State == store.ChatDeliveryDone }, settleFor, 20*time.Millisecond)
}

// The peer path takes a router's word for an app and a customer, so only a peer the
// forwarder vouches for is heard, and a header anybody can send vouches for nothing.
func (s *ChatDeliveryQueueSuite) TestThePeerPathRefusesWhatNoPeerHandedOver() {
	worker, release := s.dispatch.Register(s.customerID(), dispatch.Registration{Capacity: 1})
	defer release()
	body, err := json.Marshal(queuedChatDelivery{StreamApp: suiteStreamApp, Deployment: true,
		Body: []byte(s.message(suiteStreamApp, deliveredConfig(&s.RouterSuite, s.customerID()), ""))})
	s.Require().NoError(err)

	request, err := http.NewRequest(http.MethodPost, s.server.URL+chatDeliveryPeerPath, strings.NewReader(string(body)))
	s.Require().NoError(err)
	request.Header.Set(node.ForwardedHeader, "1")
	response, err := s.server.Client().Do(request)
	s.Require().NoError(err)
	s.Require().NoError(response.Body.Close())

	s.Equal(http.StatusNotFound, response.StatusCode)
	s.False(s.reachesWorker(worker.Messages()))
}

// A peer that could not place the message says so, so the router that handed it over tries
// again rather than record it answered.
func (s *ChatDeliveryQueueSuite) TestAPeerThatCouldNotPlaceADeliveryAnswersUnavailable() {
	other := s.otherNodeDispatching(dispatch.NewPool())
	address := other.Listener.Addr().String()

	answer, err := s.router.handChatDeliveryToPeer(context.Background(), address,
		queuedChatDelivery{StreamApp: suiteStreamApp, Deployment: true, Body: []byte("not a delivery")})
	s.Require().NoError(err)
	s.Equal(int32(http.StatusBadRequest), answer.GetStatus(), "body: %s", answer.GetBody())

	answer, err = s.router.handChatDeliveryToPeer(context.Background(), address, queuedChatDelivery{StreamApp: suiteStreamApp,
		Deployment: true, Body: []byte(s.message(suiteStreamApp, deliveredConfig(&s.RouterSuite, s.customerID()), ""))})
	s.Require().NoError(err)
	s.Equal(int32(http.StatusServiceUnavailable), answer.GetStatus(), "body: %s", answer.GetBody())
}

// Nobody waiting to answer is no answer: the delivery stays queued for a worker to connect.
func (s *ChatDeliveryQueueSuite) TestADeliveryNobodyTookIsTriedAgain() {
	queued := s.queue(s.message(suiteStreamApp, deliveredConfig(&s.RouterSuite, s.customerID()), ""))

	s.answering()

	s.Eventually(func() bool { return s.row(queued).Attempts == 1 }, settleFor, 20*time.Millisecond)
	s.Equal(store.ChatDeliveryQueued, s.row(queued).State)
}

// A node the directory says answers in the channel, gone, is still the only one that may: the
// delivery stays queued for it, or for its entry to lapse.
func (s *ChatDeliveryQueueSuite) TestADeliveryForANodeThatCannotBeReachedIsTriedAgain() {
	other := s.otherNode()
	s.serverClient.on(other).createSession(CreateSessionRequest{
		AgentId: &s.channelID, Llm: pointerTo("vision/vision-model"), Incognito: pointerTo(true),
	})
	other.Close()
	worker, release := s.dispatch.Register(s.customerID(), dispatch.Registration{Capacity: 1})
	defer release()
	queued := s.queue(s.message(suiteStreamApp, deliveredConfig(&s.RouterSuite, s.customerID()), ""))

	s.answering()

	s.Eventually(func() bool { return s.row(queued).Attempts == 1 }, settleFor+forwardTimeout, 20*time.Millisecond)
	s.Equal(store.ChatDeliveryQueued, s.row(queued).State)
	s.False(s.reachesWorker(worker.Messages()), "a worker here answered in the conversation the other node holds")
}

// A peer that took a delivery answers it itself: handing it on to a third node's worker could
// pass it back and forth between nodes that each believe the other answers.
func (s *ChatDeliveryQueueSuite) TestADeliveryAPeerHandedOverIsNotHandedOnAgain() {
	waiting := s.serverClient.opens("/v1/dispatch")
	s.await(waiting, "ready")
	other := s.otherNodeDispatching(dispatch.NewPool())

	answer, err := s.router.handChatDeliveryToPeer(context.Background(), other.Listener.Addr().String(), queuedChatDelivery{StreamApp: suiteStreamApp,
		Deployment: true, Body: []byte(s.message(suiteStreamApp, deliveredConfig(&s.RouterSuite, s.customerID()), ""))})

	s.Require().NoError(err)
	s.Equal(int32(http.StatusServiceUnavailable), answer.GetStatus(), "body: %s", answer.GetBody())
}

// A row that is not a delivery is no answer, and says why, rather than read as nothing to do.
func (s *ChatDeliveryQueueSuite) TestARowThatIsNotADeliveryIsAnError() {
	err := s.router.AnswerChatDelivery(context.Background(), store.ChatDelivery{StreamApp: suiteStreamApp, Deployment: true, Body: []byte("not a delivery")})

	s.Error(err)
	s.NotErrorIs(err, errChatDeliveryUnsettled)
}

// A channel no agent in the app answers in is settled, not tried again: no later try finds one.
func (s *ChatDeliveryQueueSuite) TestADeliveryNoAgentAnswersIsSettled() {
	s.NoError(s.router.AnswerChatDelivery(context.Background(), store.ChatDelivery{StreamApp: suiteStreamApp, Deployment: true,
		Body: []byte(s.message(suiteStreamApp, "no-such-config", ""))}))
}

// A read that failed is no answer: the delivery is tried again rather than recorded settled,
// here the channel's last call, which Close cancelling the worker's context fails.
func (s *ChatDeliveryQueueSuite) TestADeliveryWhoseChannelsLastCallCouldNotBeReadIsTriedAgain() {
	cancelled, cancel := context.WithCancel(context.Background())
	cancel()

	err := s.router.AnswerChatDelivery(cancelled, store.ChatDelivery{StreamApp: suiteStreamApp, Deployment: true,
		Body: []byte(s.message(suiteStreamApp, "", ""))})

	s.ErrorIs(err, errChatDeliveryUnsettled)
}

// Calls of two customers on the channel say nobody owns it, as a hook reads them: the config
// the channel names answers, and nothing is tried again.
func (s *ChatDeliveryQueueSuite) TestADeliveryOnAChannelTwoCustomersCalledOnIsAnsweredByItsConfig() {
	worker, release := s.dispatch.Register(s.customerID(), dispatch.Registration{Capacity: 1})
	defer release()
	for _, customer := range []string{"someone-" + s.utils.uuid(), "someone-" + s.utils.uuid()} {
		s.Require().NoError(s.store.StartCall(context.Background(), &store.Call{
			ID: s.utils.uuid(), CustomerID: customer, CallID: s.utils.callID(), AgentID: s.channelID,
			ConfigID: deliveredConfig(&s.RouterSuite, customer), StartedAt: time.Now().UTC(),
		}))
	}

	err := s.router.AnswerChatDelivery(context.Background(), store.ChatDelivery{StreamApp: suiteStreamApp, Deployment: true,
		Body: []byte(s.message(suiteStreamApp, deliveredConfig(&s.RouterSuite, s.customerID()), ""))})

	s.NoError(err)
	s.True(s.reachesWorker(worker.Messages()))
}

// In app mode, whether the customer may still write in the deployment's app is not known
// before the deployment's app is: the delivery waits for it.
func (s *ChatDeliveryQueueSuite) TestADeliveryForACustomerWhoseAppIsNotKnownYetIsTriedAgain() {
	customer := s.inAppMode()
	s.setApps(customer, func(apps *suiteApps) { apps.waiting[customer] = true })

	err := s.router.AnswerChatDelivery(context.Background(), store.ChatDelivery{StreamApp: suiteStreamApp, Deployment: true,
		Body: []byte(s.message(suiteStreamApp, deliveredConfig(&s.RouterSuite, customer), ""))})

	s.ErrorIs(err, errChatDeliveryUnsettled)
}

// A customer no longer writing in the deployment's app is an answer that holds: settled.
func (s *ChatDeliveryQueueSuite) TestADeliveryForACustomerNoLongerWritingThereIsSettled() {
	customer := s.inAppMode()
	s.setApps(customer, func(apps *suiteApps) { apps.readOnly[customer] = true })

	s.NoError(s.router.AnswerChatDelivery(context.Background(), store.ChatDelivery{StreamApp: suiteStreamApp, Deployment: true,
		Body: []byte(s.message(suiteStreamApp, deliveredConfig(&s.RouterSuite, customer), ""))}))
}

// A thread channel whose thread could not be read may hold one: tried again.
func (s *ChatDeliveryQueueSuite) TestADeliveryOnAThreadChannelThatCouldNotBeReadIsTriedAgain() {
	s.channelID = conversation.ThreadChannelPrefix + s.utils.uuid()
	cancelled, cancel := context.WithCancel(context.Background())
	cancel()

	err := s.router.AnswerChatDelivery(cancelled, store.ChatDelivery{StreamApp: suiteStreamApp, Deployment: true,
		Body: []byte(s.message(suiteStreamApp, "", ""))})

	s.ErrorIs(err, errChatDeliveryUnsettled)
}

// A session that leaves text to the customer's dispatch worker, with none connected, did not
// take the message: tried again for a worker to connect.
func (s *ChatDeliveryQueueSuite) TestADeliveryForADispatchingSessionNoWorkerTookIsTriedAgain() {
	config := store.AgentConfig{CustomerID: s.customerID(), Name: "config-" + s.utils.uuid(),
		Mode: store.AgentModeText, LLM: "en-low-latency", DispatchText: true}
	s.Require().NoError(s.store.CreateAgentConfig(context.Background(), &config))
	s.serverClient.createSession(CreateSessionRequest{AgentId: &s.channelID, Agent: &config.Name, Incognito: pointerTo(true)})

	err := s.router.AnswerChatDelivery(context.Background(), store.ChatDelivery{StreamApp: suiteStreamApp, Deployment: true,
		Body: []byte(s.message(suiteStreamApp, "", ""))})

	s.ErrorIs(err, errChatDeliveryUnsettled)
}

// A peer answered the delivery and its reply never came back, so the router that handed it
// over tries again: the peer finds the answer started and answers nothing, and no worker claims
// the row again.
func (s *ChatDeliveryQueueSuite) TestAQueuedDeliveryAPeerAnsweredIsNotAnsweredAgainWhenItsReplyWasLost() {
	other := s.otherNode()
	s.serverClient.on(other).createSession(CreateSessionRequest{
		AgentId: &s.channelID, Llm: pointerTo("vision/vision-model"), Incognito: pointerTo(true),
	})
	text := "where is order " + s.utils.uuid()
	queued := s.row(s.queue(strings.Replace(s.message(suiteStreamApp, "", ""), "is my order on its way?", text, 1)))
	s.Require().NoError(s.router.AnswerChatDelivery(context.Background(), queued))
	s.Eventually(func() bool { return s.asked(text) == 1 }, settleFor, 20*time.Millisecond)

	s.NoError(s.router.AnswerChatDelivery(context.Background(), queued), "a try after the answer started is settled")
	s.answering()

	s.Never(func() bool { return s.asked(text) > 1 }, dropped, 20*time.Millisecond, "one queued delivery is one answer")
	s.Equal(store.ChatDeliveryDone, s.row(queued.MessageID).State)
}

// Close cut the router's try short after the peer had placed the message with a worker there:
// the try is recorded nowhere, and still no router places it again.
func (s *ChatDeliveryQueueSuite) TestAQueuedDeliveryAPeerTookIsNotAnsweredAgainAfterCloseCutTheTryShort() {
	other := s.otherNodeDispatching(dispatch.NewPool())
	worker := s.serverClient.on(other).opens("/v1/dispatch")
	s.await(worker, "ready")
	queued := s.row(s.queue(s.message(suiteStreamApp, deliveredConfig(&s.RouterSuite, s.customerID()), "")))
	s.Require().NoError(s.router.AnswerChatDelivery(context.Background(), queued))
	s.Equal(s.channelID, s.await(worker, "message")["channel_id"])

	s.NoError(s.router.AnswerChatDelivery(context.Background(), queued), "a try after the answer started is settled")
	s.answering()

	s.False(s.receives(worker, "message"), "one queued delivery is one answer")
	s.Equal(store.ChatDeliveryDone, s.row(queued.MessageID).State)
}

// The router placed the message with a worker and stopped before it recorded that: the next
// router's worker finds it answered.
func (s *ChatDeliveryQueueSuite) TestAQueuedDeliveryPlacedBeforeTheRouterStoppedIsNotPlacedAgain() {
	worker, release := s.dispatch.Register(s.customerID(), dispatch.Registration{Capacity: 2})
	defer release()
	queued := s.row(s.queue(s.message(suiteStreamApp, deliveredConfig(&s.RouterSuite, s.customerID()), "")))
	s.Require().NoError(s.router.AnswerChatDelivery(context.Background(), queued))
	s.True(s.reachesWorker(worker.Messages()))

	s.NoError(s.router.AnswerChatDelivery(context.Background(), queued), "a try after the answer started is settled")
	s.answering()

	s.False(s.reachesWorker(worker.Messages()), "one queued delivery is one answer")
	s.Equal(store.ChatDeliveryDone, s.row(queued.MessageID).State)
}

// The session running here answered and the router stopped before it recorded that: the next
// router's worker finds it answered.
func (s *ChatDeliveryQueueSuite) TestAQueuedDeliveryASessionHereAnsweredBeforeTheRouterStoppedIsNotAnsweredAgain() {
	s.serverClient.createSession(CreateSessionRequest{
		AgentId: &s.channelID, Llm: pointerTo("vision/vision-model"), Incognito: pointerTo(true),
	})
	text := "where is order " + s.utils.uuid()
	queued := s.row(s.queue(strings.Replace(s.message(suiteStreamApp, "", ""), "is my order on its way?", text, 1)))
	s.Require().NoError(s.router.AnswerChatDelivery(context.Background(), queued))
	s.Equal(1, s.asked(text))

	s.NoError(s.router.AnswerChatDelivery(context.Background(), queued), "a try after the answer started is settled")
	s.answering()

	s.Never(func() bool { return s.asked(text) > 1 }, dropped, 20*time.Millisecond, "one queued delivery is one answer")
}

// Tries of one delivery at once, here and through the peer they are handed to, answer it once.
func (s *ChatDeliveryQueueSuite) TestTriesOfOneQueuedDeliveryAtOnceAnswerItOnce() {
	other := s.otherNode()
	s.serverClient.on(other).createSession(CreateSessionRequest{
		AgentId: &s.channelID, Llm: pointerTo("vision/vision-model"), Incognito: pointerTo(true),
	})
	text := "where is order " + s.utils.uuid()
	queued := s.row(s.queue(strings.Replace(s.message(suiteStreamApp, "", ""), "is my order on its way?", text, 1)))

	var wg sync.WaitGroup
	errs := make(chan error, 8)
	for range 8 {
		wg.Go(func() { errs <- s.router.AnswerChatDelivery(context.Background(), queued) })
	}
	wg.Wait()
	close(errs)

	for err := range errs {
		s.NoError(err)
	}
	s.Eventually(func() bool { return s.asked(text) == 1 }, settleFor, 20*time.Millisecond)
	s.Never(func() bool { return s.asked(text) > 1 }, dropped, 20*time.Millisecond, "one queued delivery is one answer")
}

// A dispatch worker the session leaves text to took nothing: the row is queued again, for the
// worker's next try to answer.
func (s *ChatDeliveryQueueSuite) TestAQueuedDeliveryForADispatchingSessionNoWorkerTookIsQueuedAgain() {
	config := store.AgentConfig{CustomerID: s.customerID(), Name: "config-" + s.utils.uuid(),
		Mode: store.AgentModeText, LLM: "en-low-latency", DispatchText: true}
	s.Require().NoError(s.store.CreateAgentConfig(context.Background(), &config))
	s.serverClient.createSession(CreateSessionRequest{AgentId: &s.channelID, Agent: &config.Name, Incognito: pointerTo(true)})
	queued := s.row(s.queue(s.message(suiteStreamApp, "", "")))

	err := s.router.AnswerChatDelivery(context.Background(), queued)

	s.ErrorIs(err, errChatDeliveryUnsettled)
	s.Equal(store.ChatDeliveryQueued, s.row(queued.MessageID).State)
}

// A dispatching session's message went to a worker on another node, whose reply never came
// back within its deadline: that node placed it, so the row stays settled and the next try
// places nothing.
func (s *ChatDeliveryQueueSuite) TestAQueuedDeliveryAnotherNodePlacedIsNotPlacedAgainWhenItsReplyWasLate() {
	worker, router, queued := s.dispatchingBesideAWorkerElsewhere()

	s.ErrorIs(router.AnswerChatDelivery(context.Background(), queued), errChatDeliveryUnsettled)
	s.Equal(s.channelID, s.await(worker, "message")["channel_id"])

	s.Equal(store.ChatDeliveryDone, s.row(queued.MessageID).State, "left settled")
	s.NoError(router.AnswerChatDelivery(context.Background(), queued), "a try after the answer started is settled")
	s.False(s.receives(worker, "message"), "one queued delivery is one answer")
}

// Close cut a try short while it waited on the reply of the node it gave the message to, which
// had placed it: the row stays settled and the next try places nothing.
func (s *ChatDeliveryQueueSuite) TestAQueuedDeliveryAnotherNodePlacedIsNotPlacedAgainAfterCloseCutTheWaitShort() {
	worker, router, queued := s.dispatchingBesideAWorkerElsewhere()
	ctx, cancel := context.WithCancel(context.Background())
	ended := make(chan error, 1)
	go func() { ended <- router.AnswerChatDelivery(ctx, queued) }()
	s.Equal(s.channelID, s.await(worker, "message")["channel_id"])

	cancel()

	select {
	case err := <-ended:
		s.ErrorIs(err, errChatDeliveryUnsettled)
	case <-time.After(settleFor):
		s.Fail("the try never ended")
	}
	s.Equal(store.ChatDeliveryDone, s.row(queued.MessageID).State, "left settled")
	s.NoError(router.AnswerChatDelivery(context.Background(), queued), "a try after the answer started is settled")
	s.False(s.receives(worker, "message"), "one queued delivery is one answer")
}

// A row that could not be settled before the answer is no answer: tried again, here with the
// context Close cancelled.
func (s *ChatDeliveryQueueSuite) TestAQueuedDeliveryWhoseRowCouldNotBeSettledIsTriedAgain() {
	s.serverClient.createSession(CreateSessionRequest{
		AgentId: &s.channelID, Llm: pointerTo("vision/vision-model"), Incognito: pointerTo(true),
	})
	text := "where is order " + s.utils.uuid()
	queued := s.row(s.queue(strings.Replace(s.message(suiteStreamApp, "", ""), "is my order on its way?", text, 1)))
	cancelled, cancel := context.WithCancel(context.Background())
	cancel()

	err := s.router.AnswerChatDelivery(cancelled, queued)

	s.ErrorIs(err, errChatDeliveryUnsettled)
	s.Zero(s.asked(text))
	s.Equal(store.ChatDeliveryQueued, s.row(queued.MessageID).State)
}

// A thread channel's answer is claimed by its turn (answerThread), not by settling the row: the
// row stays the worker's to record.
func (s *ChatDeliveryQueueSuite) TestAQueuedDeliveryOnAThreadChannelLeavesItsRowToTheWorker() {
	s.channelID = conversation.ThreadChannelPrefix + s.utils.uuid()
	_, err := s.store.LinkChannelThread(context.Background(), &store.ChannelThread{
		ChannelID: s.channelID, CustomerID: s.customerID(), ConnectorID: "slack", ProviderUnitID: "T1",
		ThreadKey: "C1:" + s.utils.uuid(), ConnectionID: "connection-" + s.utils.uuid(), ThreadParts: map[string]string{},
	})
	s.Require().NoError(err)
	queued := s.row(s.queue(s.message(suiteStreamApp, "", "")))

	s.NoError(s.router.AnswerChatDelivery(context.Background(), queued))

	s.Equal(store.ChatDeliveryQueued, s.row(queued.MessageID).State)
}

// dispatchingBesideAWorkerElsewhere is a router running a session that leaves text to a
// dispatch worker, with the only worker waiting on another node, and a delivery queued for the
// session. The router no longer hears that node's replies, as one whose subscription to them
// broke.
func (s *ChatDeliveryQueueSuite) dispatchingBesideAWorkerElsewhere() (*websocket.Conn, *Server, store.ChatDelivery) {
	elsewhere := s.otherNodeDispatching(dispatch.NewPool())
	worker := s.serverClient.on(elsewhere).opens("/v1/dispatch")
	s.await(worker, "ready")
	logger := slog.New(slog.DiscardHandler)
	here, router, _ := s.otherNodeHolding(dispatch.NewPool(), s.sealer, logger)
	router.workers = s.deafWorkers(logger)
	config := store.AgentConfig{CustomerID: s.customerID(), Name: "config-" + s.utils.uuid(),
		Mode: store.AgentModeText, LLM: "en-low-latency", DispatchText: true}
	s.Require().NoError(s.store.CreateAgentConfig(context.Background(), &config))
	s.serverClient.on(here).createSession(CreateSessionRequest{AgentId: &s.channelID, Agent: &config.Name, Incognito: pointerTo(true)})
	return worker, router, s.row(s.queue(s.message(suiteStreamApp, "", "")))
}

// deafWorkers is a node's view of the deployment's workers that carries work to its peers and
// hears none of their replies.
func (s *ChatDeliveryQueueSuite) deafWorkers(logger *slog.Logger) *node.Workers {
	listener := httptest.NewUnstartedServer(nil)
	s.T().Cleanup(listener.Close)
	ctx, cancel := context.WithCancel(context.Background())
	workers, err := node.NewWorkers(ctx, dispatch.NewPool(), s.nodeDirectory(listener, s.otherShard, logger), s.workBus(logger), logger)
	s.Require().NoError(err)
	redis := s.live.Redis()
	pattern := "*replies:" + listener.Listener.Addr().String()
	subscribed, err := redis.Do(context.Background(), redis.B().PubsubChannels().Pattern(pattern).Build()).AsStrSlice()
	s.Require().NoError(err)
	s.Require().Len(subscribed, 1, "the node hears its peers' replies on no channel named so")
	cancel()
	s.Require().Eventually(func() bool {
		subscribed, err := redis.Do(context.Background(), redis.B().PubsubChannels().Pattern(pattern).Build()).AsStrSlice()
		return err == nil && len(subscribed) == 0
	}, settleFor, 20*time.Millisecond, "the node still hears its peers' replies")
	return workers
}

// inAppMode has the suite's Stream answer as in app mode until the test ends, and returns the
// deployment app's own customer, the only one whose configs a channel there may name then.
func (s *ChatDeliveryQueueSuite) inAppMode() string {
	customer := streamapp.CustomerOf(suiteStreamApp)
	s.setApps(customer, func(apps *suiteApps) { apps.perApp = true })
	s.T().Cleanup(func() {
		s.setApps(customer, func(apps *suiteApps) {
			apps.perApp = false
			delete(apps.waiting, customer)
			delete(apps.readOnly, customer)
		})
	})
	return customer
}

// queue queues body as the endpoint would, from the deployment's own app, due now.
func (s *ChatDeliveryQueueSuite) queue(body string) string {
	id := "queued-" + s.utils.uuid()
	queued, err := s.store.QueueChatDelivery(context.Background(), store.ChatDelivery{
		StreamApp: suiteStreamApp, MessageID: id, Deployment: true, Body: []byte(body), NextAttemptAt: time.Now(),
	})
	s.Require().NoError(err)
	s.Require().True(queued)
	return id
}

// answering runs the queue's worker on the suite's router until the test ends, trying a
// delivery again an hour on, after the test.
func (s *ChatDeliveryQueueSuite) answering() {
	worker, err := chatdelivery.NewWorker(chatdelivery.Options{
		Store: s.store, Answer: s.router.AnswerChatDelivery, Poll: 20 * time.Millisecond,
		Retries: []time.Duration{time.Hour},
	})
	s.Require().NoError(err)
	worker.Start()
	s.T().Cleanup(worker.Close)
}

// receives reports whether a message of type wanted arrives on connection before a delivery
// would have.
func (s *ChatDeliveryQueueSuite) receives(connection *websocket.Conn, wanted string) bool {
	s.Require().NoError(connection.SetReadDeadline(time.Now().Add(dropped)))
	for {
		var received map[string]any
		if connection.ReadJSON(&received) != nil {
			return false
		}
		if received["type"] == wanted {
			return true
		}
	}
}

func (s *ChatDeliveryQueueSuite) row(messageID string) store.ChatDelivery {
	var row store.ChatDelivery
	s.Require().NoError(s.store.DB().NewSelect().Model(&row).Where("message_id = ?", messageID).Scan(context.Background()))
	return row
}

// asked is how many times the model was given text to answer.
func (s *ChatDeliveryQueueSuite) asked(text string) int {
	count := 0
	for _, request := range s.vision.requests() {
		for _, message := range request.Input {
			if message.Content == text {
				count++
			}
		}
	}
	return count
}
