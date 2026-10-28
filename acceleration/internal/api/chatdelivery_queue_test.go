//go:build integration

package api

import (
	"context"
	"encoding/json"
	"net/http"
	"strings"
	"testing"
	"time"

	"github.com/GetStream/Vision-Agents/acceleration/internal/chatdelivery"
	"github.com/GetStream/Vision-Agents/acceleration/internal/dispatch"
	"github.com/GetStream/Vision-Agents/acceleration/internal/node"
	"github.com/GetStream/Vision-Agents/acceleration/internal/store"
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

// The worker records a delivery once its answer has ended, so its lease, not a goroutine
// nobody waits for, holds the message while the model writes.
func (s *ChatDeliveryQueueSuite) TestAQueuedDeliveryIsRecordedOnceItsAnswerHasEnded() {
	s.serverClient.createSession(CreateSessionRequest{
		AgentId: &s.channelID, Llm: pointerTo("vision/vision-model"), Incognito: pointerTo(true),
	})
	s.vision.holds()
	defer s.vision.answers()
	text := "where is order " + s.utils.uuid()
	queued := s.queue(strings.Replace(s.message(suiteStreamApp, "", ""), "is my order on its way?", text, 1))

	s.answering()

	s.Eventually(func() bool { return s.asked(text) == 1 }, settleFor, 20*time.Millisecond)
	s.Never(func() bool { return s.row(queued).State != store.ChatDeliveryQueued }, dropped, 20*time.Millisecond,
		"recorded before the answer ended")
	s.vision.answers()
	s.Eventually(func() bool { return s.row(queued).State == store.ChatDeliveryDone }, settleFor, 20*time.Millisecond)
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
