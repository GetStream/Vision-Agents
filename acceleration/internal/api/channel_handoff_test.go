//go:build integration

package api

import (
	"context"
	"net/http"
	"time"

	getstream "github.com/GetStream/getstream-go/v5"

	"github.com/GetStream/Vision-Agents/acceleration/internal/channelbridge"
	"github.com/GetStream/Vision-Agents/acceleration/internal/chatlog"
	"github.com/GetStream/Vision-Agents/acceleration/internal/store"
)

// The inbound worker hands the message it wrote to the agent: a customer whose app sends this
// router no message.new still gets the reply.
func (s *SlackChannelSuite) TestAMessageIsAnsweredWithNoMessageNewDelivered() {
	channel := s.messaged("U0000ALICE", "<@U0000BOT> is the build green?", "1759740000.000100", "")

	s.Equal("Noted.", s.posted(1)[0].Text)
	s.True(s.told("is the build green?"), "the model is told the person's message")
	s.Never(func() bool { return len(s.slack.Posts()) > 1 }, dropped, 20*time.Millisecond)
	s.Equal(1, s.turns(channel))
}

// The message.new about a message the bridge wrote is left to the worker, even for one whose
// turn nothing claimed.
func (s *SlackChannelSuite) TestAMessageNewForAMessageTheBridgeWroteStartsNothing() {
	channel := s.messaged("U0000ALICE", "is the build green?", "1759740000.000100", "")
	s.posted(1)
	index := s.writtenStraightIntoStream(channel, map[string]any{channelbridge.BridgedField: true})

	s.Require().Equal(http.StatusOK, s.streamDelivers(channel, index))

	s.Never(func() bool { return len(s.slack.Posts()) > 1 }, dropped, 20*time.Millisecond)
	s.Equal(1, s.turns(channel), "the hook claimed no turn")
	s.Contains(s.logged.String(), `msg="leaving a message the channel bridge wrote to its worker" channel=`+channel)
}

// A message somebody writes in the thread channel through Stream Chat itself, not through the
// bridge, is still answered by the message hook.
func (s *SlackChannelSuite) TestAMessageAnAppUserWritesInAThreadChannelIsAnsweredByTheHook() {
	channel := s.messaged("U0000ALICE", "is the build green?", "1759740000.000100", "")
	s.posted(1)
	index := s.writtenStraightIntoStream(channel, nil)

	s.Require().Equal(http.StatusOK, s.streamDelivers(channel, index))

	s.Len(s.posted(2), 2)
	s.Never(func() bool { return len(s.slack.Posts()) > 2 }, dropped, 20*time.Millisecond)
}

// Slack retries a delivery it got no 2xx for in time; the person gets one reply.
func (s *SlackChannelSuite) TestARetriedDeliveryIsAnsweredOnce() {
	event := s.message("U0000ALICE", "is the build green?", "1759740000.000100", "")
	s.deliver(event, 0)

	s.deliver(event, 1)

	s.posted(1)
	s.Never(func() bool { return len(s.slack.Posts()) > 1 }, dropped, 20*time.Millisecond)
}

// A router that wrote the message and stopped before the agent claimed its turn leaves it
// queued: the next attempt finds it written (code 4), rebuilds it from its row and claims the
// turn, so the person gets one reply.
func (s *SlackChannelSuite) TestAMessageWrittenBeforeItsTurnWasClaimedIsAnsweredOnce() {
	drop := s.trigger("failing_turn_claim", "channel_thread_messages", "INSERT",
		"NEW.kind = '"+store.ClaimTurn+"' AND NEW.channel_id IN (SELECT channel_id FROM channel_threads WHERE customer_id = '"+s.customerID()+"')",
		"RAISE EXCEPTION 'the router stopped';")
	s.deliver(s.message("U0000ALICE", "is the build green?", "1759740000.000100", ""), 0)
	s.failedOnce()
	drop()

	s.Equal("Noted.", s.posted(1)[0].Text)
	s.settled()
	s.Never(func() bool { return len(s.slack.Posts()) > 1 }, dropped, 20*time.Millisecond)
	channel := s.threadChannel("C0000CHAN:1759740000.000100")
	s.Equal(1, s.turns(channel))
	s.Len(s.chat.Messages(channel), 2, "the person's message, written once, and the reply")
}

// A message whose turn was claimed before the worker hands it over, as by an attempt that
// claimed it and stopped before its row was done, is not answered again.
func (s *SlackChannelSuite) TestAMessageWhoseTurnWasClaimedBeforeIsNotAnsweredAgain() {
	handOver := s.holdHandOffs()
	channel := s.messaged("U0000ALICE", "is the build green?", "1759740000.000100", "")
	fresh, err := s.store.ClaimChannelThreadMessage(context.Background(), channel, store.ClaimTurn, s.bridged(channel)[0]["id"].(string))
	s.Require().NoError(err)
	s.Require().True(fresh)

	handOver()

	s.settled()
	s.Never(func() bool { return len(s.slack.Posts()) > 0 }, dropped, 20*time.Millisecond)
	s.Equal(1, s.turns(channel))
}

// A router with no LLM router has no sessions (cmd/router buildSessions): the message the
// worker wrote is handed to nothing, and its turn is left unclaimed, as the message hook
// answers no thread channel there (linkedThread).
func (s *SlackChannelSuite) TestARouterWithNoSessionsHandsTheAgentNothing() {
	s.holdHandOffs()
	channel := s.messaged("U0000ALICE", "is the build green?", "1759740000.000100", "")
	thread, err := s.store.ChannelThread(context.Background(), channel)
	s.Require().NoError(err)
	written := channelbridge.Written{ID: s.bridged(channel)[0]["id"].(string), Text: "is the build green?", Author: "U0000ALICE"}

	turn, err := (&Server{store: s.store}).AnswerQueued(context.Background(), thread, written)

	s.NoError(err)
	s.Nil(turn)
	s.Zero(s.turns(channel))
}

// writtenStraightIntoStream writes a message by U0000BOB into a thread channel through Stream
// Chat, as an app user would, with custom fields, and returns its index in the channel.
func (s *SlackChannelSuite) writtenStraightIntoStream(channel string, custom map[string]any) int {
	author, text := "U0000BOB", "and the deploy?"
	_, err := s.chat.Client.UpdateUsers(context.Background(), &getstream.UpdateUsersRequest{Users: map[string]getstream.UserRequest{author: {ID: author}}})
	s.Require().NoError(err)
	_, err = s.chat.Client.Chat().SendMessage(context.Background(), chatlog.ChannelType, channel, &getstream.SendMessageRequest{
		Message: getstream.MessageRequest{Text: &text, UserID: &author, Custom: custom},
	})
	s.Require().NoError(err)
	return len(s.chat.Stored(channel)) - 1
}

// turns is how many messages of a thread channel had their turn claimed.
func (s *SlackChannelSuite) turns(channel string) int {
	var count int
	s.Require().NoError(s.store.DB().QueryRowContext(context.Background(),
		"SELECT count(*) FROM channel_thread_messages WHERE channel_id = ? AND kind = ?", channel, store.ClaimTurn).Scan(&count))
	return count
}
