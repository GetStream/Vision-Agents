package api

import (
	"context"
	"strings"
	"time"

	"github.com/GetStream/Vision-Agents/acceleration/internal/chatlog"
	"github.com/GetStream/Vision-Agents/acceleration/internal/conversation"
	"github.com/GetStream/Vision-Agents/acceleration/internal/omnichannel"
	"github.com/GetStream/Vision-Agents/acceleration/internal/session"
	"github.com/GetStream/Vision-Agents/acceleration/internal/store"
)

// chatCardTimeout bounds writing one chat episode's card: three Stream Chat writes, each of
// which chatlog gives 10 s (writeTimeout), as the call path's callCardTimeout does.
const chatCardTimeout = 30 * time.Second

// chatEpisode opens the episode a person's message in an agent channel of the app chat is in,
// in the person's omni-channel, and writes its card when the message opened it. answering is
// the spec of the agent that answers the message: the running session's, or the config the
// channel's owner holds (chatEpisodeUnopened). Only a config with episode_cards on has any:
// off, nothing here reads or writes. A thread channel's episode is the channel bridge's, an
// omni-channel is where cards are written and never where an episode starts, and a session on
// a call has its call's card, whose lines a chat episode in the same channel would cut short
// (store.EpisodeCards, until).
//
// Every message calls Open, which keeps the episode open (episode_activity.last_message_at).
// A message delivered again, by Stream or by the queue's next try, finds the episode its first
// delivery opened (store.OpenEpisode, episodes_open_thread), so it opens no second one and
// writes no second card. An episode or a card that fails costs the card, not the answer: it
// is logged and the message is answered all the same.
func (s *Server) chatEpisode(ctx context.Context, origin hookOrigin, event messageEvent, answering session.Spec) {
	if s.chatCards == nil || !answering.EpisodeCards || answering.Incognito || answering.CallID != "" ||
		strings.HasPrefix(event.ChannelID, conversation.ThreadChannelPrefix) || omnichannel.IsOmniChannel(event.ChannelID) {
		return
	}
	person, err := omnichannel.StreamUser(origin.app, event.Message.User.ID)
	if err != nil {
		s.logger.Info("no episode card for a message with no Stream user or app", "channel", event.ChannelID,
			"stream_app", origin.app)
		return
	}
	opened, err := s.chatCards.Open(ctx, omnichannel.Episode{
		CustomerID:    answering.CustomerID,
		AgentConfigID: answering.ConfigID,
		AgentName:     answering.AgentName,
		Person:        person,
		Source:        store.EpisodeChat,
		ThreadChannel: chatlog.ChannelType + ":" + event.ChannelID,
		StreamAppPK:   origin.app,
	})
	if err != nil {
		s.logger.Error("could not open the episode of a chat message", "channel", event.ChannelID,
			"customer", answering.CustomerID, "config", answering.ConfigID, "error", err)
		return
	}
	// Only an episode this message opened gets its card (Cards.Write). Off a hook's request,
	// which Stream waits on, as its answer is (hookOrigin.answer), and counted until it ends, so
	// a draining router waits for it (ChatCardsRunning): a card lost then is lost for good, as
	// the next message finds its episode open.
	s.chatCardsWriting.Add(1)
	origin.answer(func() {
		defer s.chatCardsWriting.Add(-1)
		ctx, cancel := context.WithTimeout(context.WithoutCancel(ctx), chatCardTimeout)
		defer cancel()
		if err := s.chatCards.Write(ctx, opened); err != nil {
			s.logger.Error("could not write the episode card of a chat message", "channel", event.ChannelID,
				"customer", answering.CustomerID, "episode", opened.Episode.ID, "error", err)
		}
	})
}

// chatEpisodeUnopened is chatEpisode for a message nothing runs for, which a worker is handed:
// the agent that answers is the config the channel's owner holds (ownerOf), from a read made
// already, so a config with episode_cards off costs no read of its own. heardFor is that
// config's spec when a voice recording read it; declared is the config ownerOf read: the one
// the channel names, also when its last call (lastCallOn) ran under it (heldConfigOf).
func (s *Server) chatEpisodeUnopened(ctx context.Context, origin hookOrigin, event messageEvent, declared *store.AgentConfig,
	heardFor session.Spec, recorded bool) {
	switch {
	case recorded:
		s.chatEpisode(ctx, origin, event, heardFor)
	case declared != nil:
		s.chatEpisode(ctx, origin, event, session.FromConfig(*declared))
	}
}

// ChatCardsRunning is how many episode cards of the app chat are being written now, for a
// draining router to wait for, as channelbridge.Bridge.Running.
func (s *Server) ChatCardsRunning() int {
	return int(s.chatCardsWriting.Load())
}
