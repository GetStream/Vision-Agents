package api

import (
	"context"
	"errors"
	"strings"
	"time"

	"github.com/google/uuid"

	"github.com/GetStream/Vision-Agents/acceleration/internal/chatlog"
	"github.com/GetStream/Vision-Agents/acceleration/internal/conversation"
	"github.com/GetStream/Vision-Agents/acceleration/internal/session"
	"github.com/GetStream/Vision-Agents/acceleration/internal/store"
)

// threadTurnPoll is how often a turn in a thread channel is checked for being over, and a
// follow-up refused because a reply is still running is tried again. A choice, not a
// measurement: short against a model reply of seconds, long against a Postgres-free check.
const threadTurnPoll = 50 * time.Millisecond

// threadTurnWait is how often a router waiting for a thread's turn asks Postgres again. A
// choice: a reply takes seconds, and a waiter asking every quarter second costs one indexed
// update per ask.
const threadTurnWait = 250 * time.Millisecond

// threadTurnLease is how long a turn's lease holds: the turn's own budget, askTimeout, and a
// margin for closing its session, so a lease outlives the turn that holds it and a router
// that stopped mid-turn holds the thread no longer than that.
const threadTurnLease = askTimeout + 30*time.Second

// A thread channel holds one external thread, such as a Slack thread, which the channel bridge
// (internal/channelbridge) writes into as the people in it write. The Router answers it
// itself, with a persistent text session held on the thread channel, so the reply is written
// there too and leaves for the external thread once its final text is stored
// (conversation.Service.OnFinishedReply).
//
//	message.new, no source, in agent:thread-<uuid>     receiveMessageEvent
//	  linkedThread: channel_threads row, this app's
//	  answerThread, off the request
//	    claim (thread channel, turn, Stream message id)    a repeat delivery: nothing
//	    lease the thread's turn in Postgres                another router's turn: wait
//	    the session on the channel (ByAgentWhere), else
//	      FromConfig(the channel's agent config), Text, PersistConversation,
//	      ConversationID agent:thread-<uuid>               reply lands in the thread channel
//	    Session.FollowUp(text)                             the person's message is already
//	                                                       in the channel: none is written
//	    wait until the turn is over; close the session it opened; let go of the lease

// linkedThread is the external thread a channel holds, when it is a thread channel of the
// app the hook came from.
func (s *Server) linkedThread(ctx context.Context, origin hookOrigin, channelID string) (store.ChannelThread, bool) {
	if s.store == nil || s.sessions == nil || !strings.HasPrefix(channelID, conversation.ThreadChannelPrefix) {
		return store.ChannelThread{}, false
	}
	thread, err := s.store.ChannelThread(ctx, channelID)
	if err != nil {
		if !errors.Is(err, store.ErrNoChannelThread) {
			s.logger.Error("could not tell whether a channel holds an external thread", "channel", channelID, "error", err)
		}
		return store.ChannelThread{}, false
	}
	if !origin.owns(thread.CustomerID, thread.StreamAppPK) {
		s.logger.Info("ignoring a message in a thread channel of another app", "channel", channelID, "stream_app", origin.app)
		return store.ChannelThread{}, false
	}
	return thread, true
}

// answerThread has a session on a thread channel answer a person's message there. One turn
// of a thread at a time across every router: the turn's lease is a row in Postgres
// (store.TakeChannelThreadTurn), and a message that arrives while another router answers
// waits for it. A session this router opens for the turn is closed after it, so the next
// turn, on whichever router, opens the conversation again from the channel, with every reply
// in it.
func (s *Server) answerThread(origin hookOrigin, thread store.ChannelThread, event messageEvent) {
	ctx, cancel := context.WithTimeout(context.Background(), askTimeout)
	defer cancel()

	// Stream may deliver one message.new more than once, and each would be another reply.
	fresh, err := s.store.ClaimChannelThreadMessage(ctx, thread.ChannelID, store.ClaimTurn, event.Message.ID)
	if err != nil || !fresh {
		if err != nil {
			s.logger.Error("could not claim a message of a thread channel", "channel", thread.ChannelID, "error", err)
		}
		return
	}
	holder := uuid.NewString()
	if err := s.takeThreadTurn(ctx, thread.ChannelID, holder); err != nil {
		s.logger.Error("a message in a thread channel waited too long for the thread's turn", "channel", thread.ChannelID, "error", err)
		return
	}
	// A context of its own: the lease is not let go of because the wait used the budget.
	defer func() {
		releasing, done := context.WithTimeout(context.WithoutCancel(ctx), threadTurnPoll*20)
		defer done()
		if err := s.store.ReleaseChannelThreadTurn(releasing, thread.ChannelID, holder); err != nil {
			s.logger.Error("could not let go of a thread's turn; it runs out instead", "channel", thread.ChannelID, "error", err)
		}
	}()
	turn, cancelTurn := context.WithTimeout(context.Background(), askTimeout)
	defer cancelTurn()

	// A session a caller opened on the channel through the API answers there; otherwise this
	// router opens one for the turn.
	found, running := s.sessions.ByAgentWhere(thread.ChannelID, origin.owns)
	opened := false
	if !running || !found.Spec().PersistConversation {
		if found, err = s.threadSession(turn, origin, thread, event); err != nil {
			s.logger.Error("could not open the conversation of a thread channel", "channel", thread.ChannelID, "error", err)
			return
		}
		opened = true
	}
	if opened {
		defer func() {
			if _, err := s.sessions.Close(found.ID(), session.OwnerOf(found.Spec())); err != nil {
				s.logger.Error("could not close a thread channel's session", "channel", thread.ChannelID, "session", found.ID(), "error", err)
			}
		}()
	}

	for {
		err = found.FollowUp(turn, event.Message.Text)
		if err == nil || turn.Err() != nil {
			break
		}
		// A reply another way into the session is writing ("a response is already running").
		time.Sleep(threadTurnPoll)
	}
	if err != nil {
		s.logger.Error("could not answer a message in a thread channel", "channel", thread.ChannelID, "session", found.ID(), "error", err)
		return
	}
	for found.Busy() && turn.Err() == nil {
		time.Sleep(threadTurnPoll)
	}
}

// takeThreadTurn waits for a thread channel's turn and leases it to holder for threadTurnLease.
func (s *Server) takeThreadTurn(ctx context.Context, channelID, holder string) error {
	for {
		taken, err := s.store.TakeChannelThreadTurn(ctx, channelID, holder, time.Now().Add(threadTurnLease))
		if err != nil || taken {
			return err
		}
		select {
		case <-ctx.Done():
			return ctx.Err()
		case <-time.After(threadTurnWait):
		}
	}
}

// threadSession opens the persistent conversation held on a thread channel, with the agent
// config the channel names, as the Router's own session: no end user owns a thread several
// people write in.
func (s *Server) threadSession(ctx context.Context, origin hookOrigin, thread store.ChannelThread, event messageEvent) (*session.Session, error) {
	customerID, configID, found := s.ownerOf(ctx, origin, event)
	if !found || customerID != thread.CustomerID {
		return nil, errors.New("the thread channel names no agent config of its customer")
	}
	if !s.mayWrite(ctx, origin, customerID) {
		return nil, errors.New("the customer no longer writes in the app the hook came from")
	}
	config, err := s.store.AgentConfig(ctx, customerID, configID)
	if err != nil {
		return nil, err
	}
	spec := session.FromConfig(config)
	spec.Text = true
	spec.CallID, spec.STSTarget, spec.Greeting = "", "", ""
	spec.PersistConversation = true
	spec.ConversationID = chatlog.ChannelType + ":" + thread.ChannelID
	return s.sessions.Create(ctx, spec)
}

// threadConversation holds a text session asked for on a thread channel, by its agent id, on
// that channel: an agent id that names one of the customer's thread channels keeps the
// session's conversation there, so its replies land in the thread, not in a support channel
// of its own. Anything else is left as it is.
func (s *Server) threadConversation(ctx context.Context, customerID string, spec *session.Spec) {
	if s.store == nil || !spec.Text || !spec.PersistConversation || spec.ConversationID != "" ||
		!strings.HasPrefix(spec.AgentID, conversation.ThreadChannelPrefix) {
		return
	}
	thread, err := s.store.ChannelThread(ctx, spec.AgentID)
	if err != nil || thread.CustomerID != customerID {
		return
	}
	spec.ConversationID = chatlog.ChannelType + ":" + thread.ChannelID
	// A resumed conversation names its own agent (Spec.Normalize).
	spec.AgentID = ""
}
