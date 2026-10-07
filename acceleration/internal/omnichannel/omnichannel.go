// Package omnichannel writes a person's episode cards (T41 and T43, AI-883; channels.md on
// connectors/planning, «The episode card» and «How the agent knows it is the same person»).
//
// The contact map turns how a person is known, a phone number or a Slack user, into their
// omni-channel: one agent channel for each person and agent. Each episode, one call or one
// run of messages on one external thread, gets one card there: a message with a source, the
// episode's status, when it started and the channel that holds its raw text. The raw text
// stays in the thread channel or the call channel. The channel bridge writes a card on a
// thread's first message, the session manager when a call's session starts.
package omnichannel

import (
	"context"
	"errors"
	"fmt"
	"regexp"
	"strings"
	"time"

	getstream "github.com/GetStream/getstream-go/v5"
	"github.com/google/uuid"

	"github.com/GetStream/Vision-Agents/acceleration/internal/chatlog"
	"github.com/GetStream/Vision-Agents/acceleration/internal/conversation"
	"github.com/GetStream/Vision-Agents/acceleration/internal/stack"
	"github.com/GetStream/Vision-Agents/acceleration/internal/store"
	"github.com/GetStream/Vision-Agents/acceleration/internal/streamapp"
)

// omniChannelPrefix starts every omni-channel's id. A UUID follows, never the number: a
// channel id is seen by every member of the channel and every client of the app (channels.md,
// «Risks»). With the UUID the id is 41 characters, within Stream's «max length 64 characters»
// for a channel id (https://getstream.io/chat/docs/go-golang/creating_channels/).
const omniChannelPrefix = "omni-"

// configField is the custom field on an agent channel naming the agent config it belongs to:
// api.ConfigField, which this package cannot import. On an omni-channel it is what makes a
// message without a source one the message hook hands to a worker, so the card's source is
// what keeps a card from starting anything.
const configField = "agent_config_id"

// agentField is the channel custom field naming the channel's agent user
// (conversation.ownedBy, support_agent_id). The omni-channel's agent writes as a user of the
// channel's own id, as a thread channel's does (channelbridge).
const agentField = "support_agent_id"

// The card's custom fields beside chatlog.SourceField (channels.md, «Fields of the card»).
const (
	statusField        = "status"
	startedAtField     = "started_at"
	threadChannelField = "thread_channel"
	callIDField        = "call_id"
	episodeField       = "episode_id"
)

// cardText is the card's text until T55 writes the summary into it. The design leaves the
// text empty while the episode goes on; whether Stream Chat stores a message with no text and
// no attachments is unverified, so the card says what it is instead.
const cardText = "Episode in progress"

// e164Digits is a number in E.164 without its plus: a country code, which starts with 1 to 9,
// and at most 15 digits in all (ITU-T Recommendation E.164, «the maximum number length»: 15).
var e164Digits = regexp.MustCompile(`^[1-9][0-9]{1,14}$`)

// Options configures Cards.
type Options struct {
	Store *store.Store
	// Stream is how the omni-channels are written, in the Stream app each contact is pinned to.
	Stream *streamapp.Clients
}

// Cards opens episodes and writes their cards.
type Cards struct {
	store  *store.Store
	stream *streamapp.Clients
}

// New validates the options and returns Cards.
func New(options Options) (*Cards, error) {
	if options.Store == nil || options.Stream == nil {
		return nil, stack.Wrap(errors.New("omnichannel: a store and Stream clients are required"))
	}
	return &Cards{store: options.Store, stream: options.Stream}, nil
}

// Person is who an episode is with, as the contact map keys them.
type Person struct {
	// Kind is store.ContactPhone or store.ContactSlack.
	Kind    string
	Address string
}

// Phone is the person a phone number is: the number in E.164, +<country code><number>. Spaces,
// hyphens, dots and brackets are dropped. A number without its plus is read as international,
// as Vonage quotes them (phone/vonage.e164); a national number, which starts with a 0 trunk
// prefix, is refused rather than guessed at.
func Phone(number string) (Person, error) {
	digits := strings.Map(func(r rune) rune {
		switch r {
		case ' ', '-', '.', '(', ')':
			return -1
		}
		return r
	}, strings.TrimSpace(number))
	digits = strings.TrimPrefix(digits, "+")
	if !e164Digits.MatchString(digits) {
		return Person{}, stack.Wrap(errors.New("omnichannel: not a phone number in E.164"))
	}
	return Person{Kind: store.ContactPhone, Address: "+" + digits}, nil
}

// SlackUser is the person a Slack user is in one workspace. Slack gives no phone number, so
// until account linking joins them, a Slack user's episodes go to an omni-channel of their
// own: one person in one workspace, for one agent.
func SlackUser(teamID, userID string) (Person, error) {
	if teamID == "" || userID == "" {
		return Person{}, stack.Wrap(errors.New("omnichannel: a Slack team and a Slack user are required"))
	}
	return Person{Kind: store.ContactSlack, Address: teamID + ":" + userID}, nil
}

// Episode is the start of an episode, as the channel bridge or the call path sees it.
type Episode struct {
	CustomerID    string
	AgentConfigID string
	// AgentName is the name the omni-channel's agent user is shown by, for a new omni-channel.
	AgentName string
	Person    Person
	// Source is the card's source: store.EpisodeCall or store.EpisodeSlack.
	Source string
	// ThreadChannel is the cid of the channel with the raw text.
	ThreadChannel string
	// CallID and SessionID are a call's Stream call and router session.
	CallID    string
	SessionID string
	StartedAt time.Time
	// StreamAppPK is the app a new omni-channel is made in: the thread's or the call's.
	StreamAppPK int64
}

// Opened is an episode the contact map and the episode store hold.
type Opened struct {
	Episode   store.Episode
	Contact   store.ContactMapEntry
	AgentName string
	// New is whether this opened the episode, and so whether its card is still to be written.
	New bool
}

// Open finds the person's omni-channel in the contact map, making one for a person seen for
// the first time, and opens the episode unless its thread or its call session already has
// one. Postgres only, so the channel bridge calls it on the request a provider waits on.
func (c *Cards) Open(ctx context.Context, episode Episode) (Opened, error) {
	contact := store.ContactMapEntry{
		CustomerID:     episode.CustomerID,
		AgentConfigID:  episode.AgentConfigID,
		Kind:           episode.Person.Kind,
		Address:        episode.Person.Address,
		ConversationID: chatlog.ChannelType + ":" + omniChannelPrefix + uuid.NewString(),
		StreamAppPK:    episode.StreamAppPK,
	}
	if _, err := c.store.MapContact(ctx, &contact); err != nil {
		return Opened{}, err
	}
	row := store.Episode{
		CustomerID:    episode.CustomerID,
		ContactID:     contact.ID,
		Source:        episode.Source,
		ThreadChannel: episode.ThreadChannel,
		CallID:        episode.CallID,
		SessionID:     episode.SessionID,
		StartedAt:     episode.StartedAt,
		StreamAppPK:   contact.StreamAppPK,
	}
	opened, err := c.store.OpenEpisode(ctx, &row)
	if err != nil {
		return Opened{}, err
	}
	return Opened{Episode: row, Contact: contact, AgentName: episode.AgentName, New: opened}, nil
}

// Write writes the card of an episode Open opened into the person's omni-channel, making the
// channel the first time. An episode Open found already open has its card, and nothing is
// written. The card has a source, so the message hook, which answers only a message without
// one (api.addressed), starts no session for it.
func (c *Cards) Write(ctx context.Context, opened Opened) error {
	if !opened.New {
		return nil
	}
	bound, err := c.stream.ForApp(ctx, opened.Contact.CustomerID, opened.Contact.StreamAppPK)
	if err != nil {
		return err
	}
	channelID := strings.TrimPrefix(opened.Contact.ConversationID, chatlog.ChannelType+":")
	agentName := opened.AgentName
	if err := conversation.CreateMissingUsers(ctx, bound.Client, map[string]getstream.UserRequest{
		channelID: {ID: channelID, Name: &agentName},
	}); err != nil {
		return stack.Wrap(err)
	}
	// The agent channel type refuses a server-side create without a creator. No owner and no
	// member: the person reaches the omni-channel through the agent, not as a member of it.
	_, err = bound.Client.Chat().GetOrCreateChannel(ctx, chatlog.ChannelType, channelID, &getstream.GetOrCreateChannelRequest{
		Data: &getstream.ChannelInput{
			CreatedByID: &channelID,
			Custom: map[string]any{
				configField:                opened.Contact.AgentConfigID,
				conversation.CustomerField: opened.Contact.CustomerID,
				agentField:                 channelID,
			},
		},
	})
	if err != nil {
		return stack.Wrap(err)
	}
	episode := opened.Episode
	custom := map[string]any{
		chatlog.SourceField: episode.Source,
		statusField:         episode.Status,
		startedAtField:      episode.StartedAt.UTC().Format(time.RFC3339),
		threadChannelField:  episode.ThreadChannel,
		episodeField:        episode.ID,
	}
	if episode.CallID != "" {
		custom[callIDField] = episode.CallID
	}
	text := cardText
	// The message id is the episode's, so a card written twice is one message.
	_, err = bound.Client.Chat().SendMessage(ctx, chatlog.ChannelType, channelID, &getstream.SendMessageRequest{
		Message: getstream.MessageRequest{ID: &episode.CardMessageID, UserID: &channelID, Text: &text, Custom: custom},
	})
	if err != nil {
		return stack.Wrap(fmt.Errorf("omnichannel: write the card of episode %s: %w", episode.ID, err))
	}
	return nil
}
