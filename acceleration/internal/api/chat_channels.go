package api

import (
	"context"
	"encoding/json"
	"strings"
	"time"
	"unicode"
	"unicode/utf8"

	getstream "github.com/GetStream/getstream-go/v5"
	"github.com/google/uuid"

	"github.com/GetStream/Vision-Agents/acceleration/internal/chatlog"
	"github.com/GetStream/Vision-Agents/acceleration/internal/conversation"
	"github.com/GetStream/Vision-Agents/acceleration/internal/session"
	"github.com/GetStream/Vision-Agents/acceleration/internal/stack"
	"github.com/GetStream/Vision-Agents/acceleration/internal/store"
)

// A channel of a type an agent config names (store.AgentConfig.ChatChannels), such as a team's
// group channel, is answered by the Router itself, as a thread channel is:
//
//	message.new, no source, in callouts:team-1          receiveMessageEvent, receiveChatInbound
//	  addressedIn: the hook's app's own customer has a    a shared app, a type no config names,
//	    config naming callouts (chatChannelConfig)          a failed read: nothing
//	  answerChatChannel, off the request
//	    lease the channel's turn in Postgres                another router's turn: wait
//	    the session on the channel (ByConversationWhere), else
//	      FromConfig(that config), Text, Dispatched,         opened on the Router's word
//	      ConversationID callouts:team-1                     (RouterOpensDispatched), whatever
//	                                                         the channel's custom data says
//	    Session.FollowUp(text, photos)                     the person's message is already
//	    wait until the turn is over; close the session      in the channel: only the reply is
//	    it opened; let go of the lease                       written, as the config's id

// addressedIn is addressed for a channel of any type: one of a type other than agent is
// written to an agent only when a config of the hook's app's own customer answers in channels
// of that type. A read that fails answers nothing, as a hook that cannot tell has always been
// left: Stream would only send it again.
func (s *Server) addressedIn(ctx context.Context, origin hookOrigin, event messageEvent) bool {
	if event.ChannelType == chatlog.ChannelType || !answerable(event) {
		return addressed(event)
	}
	config, found, err := s.chatChannelConfig(ctx, origin, event)
	if err != nil {
		s.logger.Error("could not read whether an agent config answers in a channel's type",
			"channel_type", event.ChannelType, "channel", event.ChannelID, "stream_app", origin.app, "error", err)
	}
	return found && answersTo(config, event)
}

// answersTo is whether the config answers a message in a channel of a type it names: any
// message, or, when it answers only its mentions there (store.ChatChannelAnswerMentioned), one
// that mentions the agent, whose Stream user is the config's id named after it
// (chatChannelSession). A message it does not answer stays in the channel unanswered, and the
// next one it answers reads it as what was said before (conversation.Answering).
func answersTo(config store.AgentConfig, event messageEvent) bool {
	return !config.AnswersOnlyMentionsIn(event.ChannelType) || event.mentions(config.ID, config.Name)
}

// mentions is whether the message mentions the Stream user id, named name: its mentioned_users
// holds the user, as Stream's clients send a mention picked from their list, or its text has @
// and the id or the name, as those clients write one (`@${name || id}`, stream-chat-js
// src/messageComposer/middleware/textComposer/mentions.ts) and a person may type one.
func (e messageEvent) mentions(id, name string) bool {
	var users []struct {
		ID string `json:"id"`
	}
	if json.Unmarshal(e.Message.MentionedUsers, &users) == nil {
		for _, user := range users {
			if user.ID == id {
				return true
			}
		}
	}
	return mentionedIn(e.Message.Text, id) || mentionedIn(e.Message.Text, name)
}

// mentionedIn is whether text has @handle, in any case, as a mention of its own: the @ starts
// the text or follows a character no handle holds, so an address such as anya@rota is none, and
// the handle is not followed by one, so @rota-2, another user, is no mention of rota.
func mentionedIn(text, handle string) bool {
	if handle == "" {
		return false
	}
	for rest := text; ; {
		at := strings.IndexByte(rest, '@')
		if at < 0 {
			return false
		}
		before, after := rest[:at], rest[at+1:]
		rest = after
		if previous, _ := utf8.DecodeLastRuneInString(before); before != "" && inHandle(previous) {
			continue
		}
		if len(after) < len(handle) || !strings.EqualFold(after[:len(handle)], handle) {
			continue
		}
		if next, _ := utf8.DecodeRuneInString(after[len(handle):]); len(after) > len(handle) && inHandle(next) {
			continue
		}
		return true
	}
}

// inHandle is whether r may be part of a Stream user's id or a word of their name: a letter, a
// digit, or one of _ - @, which Stream allows in a user id besides them.
func inHandle(r rune) bool {
	return unicode.IsLetter(r) || unicode.IsDigit(r) || r == '_' || r == '-' || r == '@'
}

// chatChannelConfig is the agent config that answers in the channel of a message of a type
// other than agent: the one of the hook's app's own customer naming the type. The channel's
// custom data is not read: whoever makes a channel decides what is on it. An app every customer
// shares, the deployment's in deployment mode, has no customer of its own (configOwnerOf), so
// none of them answers there.
func (s *Server) chatChannelConfig(ctx context.Context, origin hookOrigin, event messageEvent) (store.AgentConfig, bool, error) {
	owner, scoped := s.configOwnerOf(origin)
	if s.store == nil || !scoped {
		return store.AgentConfig{}, false, nil
	}
	return s.store.AgentConfigAnsweringChatChannel(ctx, owner, event.ChannelType)
}

// answerChatChannel has the Router's own session answer a message in a channel of a type an
// agent config answers in. Like a thread's, it answers what was written and the photos: a
// voice recording alone has nothing in it to answer there. settled is false when the config
// could not be read, which a later try of a queued delivery may get through.
func (s *Server) answerChatChannel(ctx context.Context, origin hookOrigin, event messageEvent) (settled bool) {
	if s.sessions == nil {
		return true
	}
	config, found, err := s.chatChannelConfig(ctx, origin, event)
	if err != nil {
		s.logger.Error("could not read the agent config answering in a channel's type",
			"channel_type", event.ChannelType, "channel", event.ChannelID, "stream_app", origin.app, "error", err)
		return false
	}
	// The config may have come to answer only its mentions since the message was queued.
	if !found || !answersTo(config, event) || strings.TrimSpace(event.Message.Text) == "" && len(event.photos()) == 0 {
		return true
	}
	if answer, settled := s.answerOnce(ctx, origin, event); !answer {
		return settled
	}
	cid := event.ChannelType + ":" + event.ChannelID
	origin.answer(func() { s.answerChatChannelTurn(origin, config, cid, event) })
	return true
}

// answerChatChannelTurn is answerChatChannel off the request. One turn of a channel at a time
// across every router, as a thread channel's: the turn's lease is a row in Postgres
// (store.TakeChatChannelTurn), and a message that arrives while another router answers waits
// for it, so the session opened for it reads that router's reply from the channel.
func (s *Server) answerChatChannelTurn(origin hookOrigin, config store.AgentConfig, cid string, event messageEvent) {
	ctx, cancel := context.WithTimeout(context.Background(), askTimeout)
	defer cancel()
	holder := uuid.NewString()
	if err := takeTurn(ctx, func(ctx context.Context, until time.Time) (bool, error) {
		return s.store.TakeChatChannelTurn(ctx, origin.app, cid, holder, until)
	}); err != nil {
		s.logger.Error("a message in a channel the Router answers in waited too long for the channel's turn", "channel", cid, "error", err)
		return
	}
	// A context of its own: the lease is not let go of because the wait used the budget.
	defer func() {
		releasing, done := context.WithTimeout(context.WithoutCancel(ctx), threadTurnPoll*20)
		defer done()
		if err := s.store.ReleaseChatChannelTurn(releasing, origin.app, cid, holder); err != nil {
			s.logger.Error("could not let go of a channel's turn; it runs out instead", "channel", cid, "error", err)
		}
	}()
	s.answerHosted(origin, cid, config.CustomerID, event,
		func(turn context.Context) (*session.Session, error) {
			return s.chatChannelSession(turn, origin, config, cid, event)
		})
}

// chatChannelSession opens the Router's own session on the channel cid, under the config that
// answers in its type. It writes as the Stream user whose id is the config's, named after it,
// made when the app has none: one user per agent, whichever channel of the type it is in.
func (s *Server) chatChannelSession(ctx context.Context, origin hookOrigin, config store.AgentConfig, cid string, event messageEvent) (*session.Session, error) {
	// Refused for a customer whose work in the app is only to be read any more (ErrReadOnly).
	bound, err := s.streamForApp(ctx, config.CustomerID, origin.app, false)
	if err != nil {
		return nil, err
	}
	spec := session.FromConfig(config)
	spec.Text = true
	spec.CallID, spec.STSTarget, spec.Greeting = "", "", ""
	spec.PersistConversation = true
	spec.ConversationID, spec.Dispatched = cid, true
	spec.AgentID = config.ID
	spec.Attribution = chatHookAttribution(config.CustomerID, event)
	opening := conversation.RouterOpensDispatched(ctx, cid)
	if config.AnswersOnlyMentionsIn(event.ChannelType) {
		// What was written since the last reply without mentioning the agent was never answered:
		// it is read as what was said before this message.
		opening = conversation.Answering(opening, event.Message.ID)
	}
	opened, err := s.sessions.Create(opening, spec)
	if err != nil {
		return nil, err
	}
	// The user is made once the session is: one the type's grants refuse (Manager.keepable), or
	// a type the app lacks, makes none and writes nothing to Stream.
	if err := conversation.CreateMissingUsers(ctx, bound.Client, map[string]getstream.UserRequest{
		config.ID: {ID: config.ID, Name: &config.Name},
	}); err != nil {
		if _, closeErr := s.sessions.Close(opened.ID(), session.OwnerOf(opened.Spec())); closeErr != nil {
			s.logger.Error("could not close the session of a channel the Router answers in", "channel", cid, "session", opened.ID(), "error", closeErr)
		}
		return nil, stack.Wrap(err)
	}
	return opened, nil
}
