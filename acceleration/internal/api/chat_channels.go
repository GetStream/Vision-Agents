package api

import (
	"context"
	"strings"

	getstream "github.com/GetStream/getstream-go/v5"

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
//	    the session on the channel (ByConversationWhere), else
//	      FromConfig(that config), Text, Dispatched,         opened on the Router's word
//	      ConversationID callouts:team-1                     (RouterOpensDispatched), whatever
//	                                                         the channel's custom data says
//	    Session.FollowUp(text, photos)                     the person's message is already
//	    wait until the turn is over; close the session      in the channel: only the reply is
//	                                                         written, as the config's id

// addressedIn is addressed for a channel of any type: one of a type other than agent is
// written to an agent only when a config of the hook's app's own customer answers in channels
// of that type. A read that fails answers nothing, as a hook that cannot tell has always been
// left: Stream would only send it again.
func (s *Server) addressedIn(ctx context.Context, origin hookOrigin, event messageEvent) bool {
	if event.ChannelType == chatlog.ChannelType || !answerable(event) {
		return addressed(event)
	}
	_, found, err := s.chatChannelConfig(ctx, origin, event)
	if err != nil {
		s.logger.Error("could not read whether an agent config answers in a channel's type",
			"channel_type", event.ChannelType, "channel", event.ChannelID, "stream_app", origin.app, "error", err)
	}
	return found
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
	if !found || strings.TrimSpace(event.Message.Text) == "" && len(event.photos()) == 0 {
		return true
	}
	if answer, settled := s.answerOnce(ctx, origin, event); !answer {
		return settled
	}
	cid := event.ChannelType + ":" + event.ChannelID
	origin.answer(func() {
		s.answerHosted(origin, cid, config.CustomerID, event,
			func(turn context.Context) (*session.Session, error) {
				return s.chatChannelSession(turn, origin, config, cid, event)
			})
	})
	return true
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
	if err := conversation.CreateMissingUsers(ctx, bound.Client, map[string]getstream.UserRequest{
		config.ID: {ID: config.ID, Name: &config.Name},
	}); err != nil {
		return nil, stack.Wrap(err)
	}
	spec := session.FromConfig(config)
	spec.Text = true
	spec.CallID, spec.STSTarget, spec.Greeting = "", "", ""
	spec.PersistConversation = true
	spec.ConversationID, spec.Dispatched = cid, true
	spec.AgentID = config.ID
	spec.Attribution = chatHookAttribution(config.CustomerID, event)
	return s.sessions.Create(conversation.RouterOpensDispatched(ctx, cid), spec)
}
