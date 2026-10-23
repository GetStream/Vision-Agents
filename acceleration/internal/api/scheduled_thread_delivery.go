package api

import (
	"context"
	"errors"

	"github.com/google/uuid"

	"github.com/GetStream/Vision-Agents/acceleration/internal/channelbridge"
	"github.com/GetStream/Vision-Agents/acceleration/internal/chatlog"
	"github.com/GetStream/Vision-Agents/acceleration/internal/conversation"
	"github.com/GetStream/Vision-Agents/acceleration/internal/scheduling"
	"github.com/GetStream/Vision-Agents/acceleration/internal/session"
	"github.com/GetStream/Vision-Agents/acceleration/internal/store"
)

// A task made in an SMS or WhatsApp thread (store.ScheduledTask.ChannelThread) is carried out
// in that thread: the Router opens the thread's conversation as a person's message there
// opens it (answerThread), and the reply leaves for the thread through the channel bridge, as
// every reply in a thread channel does (conversation.Service.OnFinishedReply).
//
//	the thread's row, the task's customer's and its user's     else thread_gone, final
//	ChannelBridge.Reachable                                     before the turn is paid for
//	  connection removed                                        connection_removed, final
//	  opted out                                                 channel_unreachable, final
//	  sandbox gate                                              channel_unreachable
//	  past WhatsApp's 24 h, no template                         outside_window, final
//	lease the thread's turn (store.TakeChannelThreadTurn)       another turn: wait
//	the session on the channel, else threadSpec on the task's config
//	Session.FollowUp("st_<run>"), wait for its reply            no_reply, reply_failed
//	Bridge.Send the reply, held from Bridge.Reply                opted out: channel_unreachable, final
//	                                                            refused: delivery_failed
//	close the session it opened; let go of the lease
//	delivered_to sms | whatsapp

// deliverOnThread carries out an occurrence of a task made in a thread in that thread, and
// answers where it landed once the bridge sent the agent's reply: the delivery holds the reply
// back from Bridge.Reply and sends it through the same path (Bridge.Send).
func (s *Server) deliverOnThread(ctx context.Context, delivery scheduledDelivery, channelID string) (string, string, error) {
	if s.store == nil {
		return "", "", errors.New("this router has no store to find a task's thread in")
	}
	thread, err := s.store.ChannelThread(ctx, channelID)
	if errors.Is(err, store.ErrNoChannelThread) {
		return "", "", undeliveredFinally(scheduling.ReasonThreadGone, err)
	}
	if err != nil {
		return "", "", err
	}
	// The thread is reached only under the task's customer, and only while it is still its
	// user's: the row names whose thread it is, not the task. A thread that is no one person's
	// names nobody (threadPerson), so no task's user.
	person, _ := threadPerson(thread)
	if thread.CustomerID != delivery.CustomerID || person != delivery.UserID {
		return "", "", undeliveredFinally(scheduling.ReasonThreadGone, errors.New("the task's thread is not its user's thread"))
	}
	if err := s.channelBridge.Reachable(ctx, thread); err != nil {
		return "", "", threadUnreachable(err)
	}
	holder := uuid.NewString()
	if err := s.takeThreadTurn(ctx, thread.ChannelID, holder); err != nil {
		return "", "", err
	}
	defer func() {
		releasing, done := context.WithTimeout(context.WithoutCancel(ctx), threadTurnPoll*20)
		defer done()
		if err := s.store.ReleaseChannelThreadTurn(releasing, thread.ChannelID, holder); err != nil {
			s.logger.Error("could not let go of a thread's turn; it runs out instead", "channel", thread.ChannelID, "error", err)
		}
	}()
	// A thread's turn has a turn's budget, which its lease outlasts (threadTurnLease).
	turn, cancelTurn := context.WithTimeout(ctx, askTimeout)
	defer cancelTurn()
	found, opened, err := s.taskThreadSession(turn, delivery, thread)
	if err != nil {
		return "", "", err
	}
	if opened {
		defer func() {
			if _, err := s.sessions.Close(found.ID(), session.OwnerOf(found.Spec())); err != nil {
				s.logger.Error("could not close a thread channel's session", "channel", thread.ChannelID, "session", found.ID(), "error", err)
			}
		}()
	}
	// The delivery sends the reply itself, to settle the run on whether it went out.
	replies, release := s.channelBridge.Hold(delivery.CommandID)
	defer release()
	// A retried occurrence whose turn the thread already holds is a duplicate follow-up, which
	// runs nothing (Session.FollowUp); its reply is read back, and sent unless it went out.
	_, _, retried := found.FollowUpReply(delivery.CommandID)
	if err := followUpWithin(turn, found, delivery); err != nil {
		return "", "", err
	}
	text, err := readBackReply(turn, found, delivery.CommandID)
	if err != nil {
		return "", "", err
	}
	reply := conversation.FinishedReply{Customer: delivery.CustomerID, CID: chatlog.ChannelType + ":" + thread.ChannelID,
		MessageID: found.FollowUpMessage(delivery.CommandID), CommandID: delivery.CommandID, Text: text}
	if !retried {
		// The reply as the conversation stored it, with its files.
		select {
		case reply = <-replies:
		case <-turn.Done():
			return "", "", turn.Err()
		}
	}
	if err := s.channelBridge.Send(reply); err != nil {
		return "", "", threadUnreachable(err)
	}
	return personalThreads[thread.ConnectorID], found.ID(), nil
}

// taskThreadSession is the session a task's occurrence is carried out in on its thread: one a
// caller opened on the channel and keeps, or one opened for the turn on the task's config,
// answering the task's user as the thread's person, so the turn may schedule for them again.
// opened says the delivery opened it, and so closes it.
func (s *Server) taskThreadSession(ctx context.Context, delivery scheduledDelivery, thread store.ChannelThread) (*session.Session, bool, error) {
	cid := chatlog.ChannelType + ":" + thread.ChannelID
	// Only the thread's customer opens a session on its channel today (threadConversation);
	// the filter keeps another customer's session off the task should that ever change.
	found, running := s.sessions.ByConversationWhere(cid, func(customer string, _ int64) bool {
		return customer == delivery.CustomerID
	})
	if running && found.Spec().PersistConversation {
		return found, false, nil
	}
	config, err := s.store.AgentConfig(ctx, delivery.CustomerID, delivery.ConfigID)
	if err != nil {
		return nil, false, err
	}
	spec := threadSpec(config, thread)
	spec.ThreadPerson, spec.ThreadChannel, spec.ThreadDelivery = delivery.UserID, thread.ChannelID, personalThreads[thread.ConnectorID]
	created, err := s.sessions.Create(conversation.RouterOpensThread(ctx, cid), spec)
	if err != nil {
		return nil, false, err
	}
	return created, true, nil
}

// threadUnreachable is the failure a run records for a thread the bridge would not reply on.
// A removed line, an opted-out person and a person past the window stay so until somebody
// acts (the window opens again only when the person writes), so another attempt would only
// repeat them.
func threadUnreachable(err error) error {
	switch {
	case errors.Is(err, store.ErrNoConnectorConnection):
		return undeliveredFinally(scheduling.ReasonConnectionRemoved, err)
	case errors.Is(err, channelbridge.ErrOptedOut):
		return undeliveredFinally(scheduling.ReasonUnreachable, err)
	case errors.Is(err, channelbridge.ErrHeldBack):
		return undelivered(scheduling.ReasonUnreachable, err)
	case errors.Is(err, channelbridge.ErrOutsideWindow):
		return undeliveredFinally(scheduling.ReasonOutsideWindow, err)
	}
	return err
}
