//go:build integration

package api

import (
	"context"
	"net/http"
	"strings"
	"time"

	"github.com/GetStream/Vision-Agents/acceleration/internal/conversation"
)

// A message whose follow-up the thread's session refuses for any reason but a reply still
// running is given up at once (AI-1556): each attempt whose turn did not start leaves one more
// cancelled reply in the channel, so trying again until the turn's budget ran out wrote one
// per poll and held the thread's turn for askTimeout. Here the backend's session on the thread
// channel was closed under it, so its conversation refuses every follow-up.
func (s *SlackChannelSuite) TestAMessageTheThreadsSessionRefusesLetsGoOfTheTurnAtOnce() {
	channel := s.messaged("U0000ALICE", "is the build green?", "1759740000.000100", "")
	s.serverClient.createSession(CreateSessionRequest{Agent: &s.config.Name, AgentId: &channel})
	held, found := s.manager.ByConversationWhere("agent:"+channel, nil)
	s.Require().True(found)
	s.Require().NoError(held.Close())

	s.Require().Equal(http.StatusOK, s.streamDelivers(channel, 0))

	s.Require().Eventually(func() bool {
		return strings.Contains(s.logged.String(), "could not answer a message in a thread channel")
	}, settleFor, 10*time.Millisecond, "the hook kept trying a follow-up its session refuses")
	s.Require().Eventually(func() bool {
		thread, err := s.store.ChannelThread(context.Background(), channel)
		return err == nil && thread.TurnHolder == ""
	}, settleFor, 10*time.Millisecond, "the thread's turn is let go of once the message is given up")
	s.Empty(s.slack.Posts(), "nothing was answered")

	// A scheduled task's follow-up gives such a refusal up as fast (followUpWithin).
	delivered := make(chan error, 1)
	go func() {
		delivered <- followUpWithin(context.Background(), held, scheduledDelivery{CommandID: conversation.ReadBackPrefix + s.utils.uuid(), Text: "the task is due"})
	}()
	select {
	case err := <-delivered:
		s.Error(err)
	case <-time.After(settleFor):
		s.Fail("the scheduled follow-up kept trying a refusal that is not a running reply")
	}
}

// A message that arrives while the thread's session is still writing another reply waits for
// it and is then answered: "a response is already running" is the one refusal tried again.
func (s *SlackChannelSuite) TestAMessageWhileTheThreadsSessionRepliesIsAnsweredAfterIt() {
	s.config.LLM = "holding/holding-model"
	s.Require().NoError(s.store.UpdateAgentConfig(context.Background(), &s.config))
	channel := s.messaged("U0000ALICE", "is the build green?", "1759740000.000100", "")
	held := s.serverClient.createSession(CreateSessionRequest{Agent: &s.config.Name, AgentId: &channel})
	s.holding.holds()
	s.T().Cleanup(s.holding.answers)
	// The model is the suite's, so it counts the earlier tests' requests too.
	asked := len(s.holding.requests())
	respondIn(&s.RouterSuite, s.serverClient, held.Id, "what changed since yesterday?")
	s.Require().Eventually(func() bool { return len(s.holding.requests()) == asked+1 }, settleFor, 10*time.Millisecond,
		"the backend's reply never reached the model")

	s.Require().Equal(http.StatusOK, s.streamDelivers(channel, 0))
	s.Require().Eventually(func() bool {
		thread, err := s.store.ChannelThread(context.Background(), channel)
		return err == nil && thread.TurnHolder != ""
	}, settleFor, 10*time.Millisecond, "the hook never took the thread's turn")
	// Some polls of the hook's follow-up land on the reply still running.
	time.Sleep(5 * threadTurnPoll)
	s.Len(s.holding.requests(), asked+1, "the message waits for the running reply")

	s.holding.answers()

	s.Require().Eventually(func() bool { return len(s.holding.requests()) == asked+2 }, settleFor, 10*time.Millisecond,
		"the message was never answered once the reply finished")
	s.Equal("Held.", s.posted(2)[1].Text)
}

// A scheduled task's follow-up delivered while the session is still writing another reply
// waits for it the same way (followUpWithin), on the same refusal and no other.
func (s *SlackChannelSuite) TestAScheduledFollowUpWhileTheSessionRepliesWaitsForIt() {
	s.config.LLM = "holding/holding-model"
	s.Require().NoError(s.store.UpdateAgentConfig(context.Background(), &s.config))
	channel := s.messaged("U0000ALICE", "is the build green?", "1759740000.000100", "")
	held := s.serverClient.createSession(CreateSessionRequest{Agent: &s.config.Name, AgentId: &channel})
	found, ok := s.manager.ByConversationWhere("agent:"+channel, nil)
	s.Require().True(ok)
	s.holding.holds()
	s.T().Cleanup(s.holding.answers)
	// The model is the suite's, so it counts the earlier tests' requests too.
	asked := len(s.holding.requests())
	respondIn(&s.RouterSuite, s.serverClient, held.Id, "what changed since yesterday?")
	s.Require().Eventually(func() bool { return len(s.holding.requests()) == asked+1 }, settleFor, 10*time.Millisecond,
		"the backend's reply never reached the model")

	delivered := make(chan error, 1)
	go func() {
		delivered <- followUpWithin(context.Background(), found, scheduledDelivery{CommandID: conversation.ReadBackPrefix + s.utils.uuid(), Text: "the task is due"})
	}()
	select {
	case err := <-delivered:
		s.FailNow("the follow-up did not wait for the running reply", "%v", err)
	case <-time.After(scheduledPoll / 2):
	}

	s.holding.answers()

	select {
	case err := <-delivered:
		s.NoError(err)
	case <-time.After(settleFor):
		s.FailNow("the follow-up never started once the reply finished")
	}
	s.Require().Eventually(func() bool { return len(s.holding.requests()) == asked+2 }, settleFor, 10*time.Millisecond)
}
