package api

import (
	"context"
	"errors"
	"strings"
	"sync"
	"time"

	"github.com/GetStream/Vision-Agents/acceleration/internal/chatlog"
	"github.com/GetStream/Vision-Agents/acceleration/internal/conversation"
	"github.com/GetStream/Vision-Agents/acceleration/internal/session"
	"github.com/GetStream/Vision-Agents/acceleration/internal/store"
)

// threadTurnPoll is how often a turn in a thread channel is checked for being over, and a
// follow-up refused because a reply is still running is tried again. A choice, not a
// measurement: short against a model reply of seconds, long against a Postgres-free check.
const threadTurnPoll = 50 * time.Millisecond

// A thread channel holds one external thread, such as a Slack thread, which the channel bridge
// (internal/channelbridge) writes into as the people in it write. The Router answers it
// itself, with a persistent text session held on the thread channel, so the reply is written
// there too and leaves for the external thread once its final text is stored
// (conversation.Service.OnFinishedReply).
//
//	message.new, no source, in agent:thread-<uuid>     receiveMessageEvent
//	  linkedThread: channel_threads row, this app's
//	  answerThread, off the request, one turn per thread at a time
//	    claim (thread channel, turn, Stream message id)    a repeat delivery: nothing
//	    the session on the channel (ByAgentWhere), else
//	      FromConfig(the channel's agent config), Text, PersistConversation,
//	      ConversationID agent:thread-<uuid>               reply lands in the thread channel
//	    Session.FollowUp(text)                             the person's message is already
//	                                                       in the channel: none is written
//	    wait until the turn is over; detach, which leaves the session for DetachedGrace

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

// answerThread has the session on a thread channel answer a person's message there, starting
// one when none runs. One turn of a thread at a time: a message that arrives while a reply is
// written waits for it, as a session takes one turn at a time.
func (s *Server) answerThread(origin hookOrigin, thread store.ChannelThread, event messageEvent) {
	ctx, cancel := context.WithTimeout(context.Background(), askTimeout)
	defer cancel()
	release := s.holdThread(thread.ChannelID)
	defer release()

	// Stream may deliver one message.new more than once, and each would be another reply.
	fresh, err := s.store.ClaimChannelThreadMessage(ctx, thread.ChannelID, store.ClaimTurn, event.Message.ID)
	if err != nil || !fresh {
		if err != nil {
			s.logger.Error("could not claim a message of a thread channel", "channel", thread.ChannelID, "error", err)
		}
		return
	}
	found, running := s.sessions.ByAgentWhere(thread.ChannelID, origin.owns)
	if !running || !found.Spec().PersistConversation {
		if found, err = s.threadSession(ctx, origin, thread, event); err != nil {
			s.logger.Error("could not open the conversation of a thread channel", "channel", thread.ChannelID, "error", err)
			return
		}
	}
	// Watching and then letting go leaves a session nobody else watches running for
	// DetachedGrace, so the next message in the thread finds it, and ends it after that.
	_, detach := found.Watch()
	defer detach()

	for {
		err = found.FollowUp(ctx, event.Message.Text)
		if err == nil || ctx.Err() != nil {
			break
		}
		// A reply another way into the session is writing ("a response is already running").
		time.Sleep(threadTurnPoll)
	}
	if err != nil {
		s.logger.Error("could not answer a message in a thread channel", "channel", thread.ChannelID, "session", found.ID(), "error", err)
		return
	}
	for found.Busy() && ctx.Err() == nil {
		time.Sleep(threadTurnPoll)
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

// holdThread takes one thread channel's turn, so its messages are answered one at a time.
func (s *Server) holdThread(channelID string) func() {
	s.threadsMu.Lock()
	if s.threadTurns == nil {
		s.threadTurns = map[string]*threadTurn{}
	}
	held, waiting := s.threadTurns[channelID]
	if !waiting {
		held = &threadTurn{}
		s.threadTurns[channelID] = held
	}
	held.waiting++
	s.threadsMu.Unlock()

	held.lock.Lock()
	return func() {
		held.lock.Unlock()
		s.threadsMu.Lock()
		held.waiting--
		if held.waiting == 0 {
			delete(s.threadTurns, channelID)
		}
		s.threadsMu.Unlock()
	}
}

// threadTurn is one thread channel's turn, counted so the last one out forgets it.
type threadTurn struct {
	lock    sync.Mutex
	waiting int
}
