// Package channelbridge moves messages between an external thread, such as a Slack thread,
// and the thread channel it is written into in Stream Chat (T57, AI-878; channels.md on
// connectors/planning, «Who moves messages: the channel bridge»).
//
// The inbound half takes the messages a verified provider event carries, links each external
// thread to a thread channel of its own, and writes the message there as the person who sent
// it, without a source, so Router's message hook wakes or starts the session as it does for
// any message written to an agent. The outbound half takes the agent's reply in a linked
// thread channel, which the message hook hands it, and sends it to the same external thread
// through the reply template of the connector's manifest, with the connection's credential.
package channelbridge

import (
	"bytes"
	"context"
	"crypto/sha256"
	"encoding/hex"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"log/slog"
	"net/http"
	"slices"
	"strings"
	"sync"
	"time"

	getstream "github.com/GetStream/getstream-go/v5"
	"github.com/google/uuid"

	"github.com/GetStream/Vision-Agents/acceleration/internal/chatlog"
	"github.com/GetStream/Vision-Agents/acceleration/internal/connectors/core"
	"github.com/GetStream/Vision-Agents/acceleration/internal/conversation"
	"github.com/GetStream/Vision-Agents/acceleration/internal/dlc"
	"github.com/GetStream/Vision-Agents/acceleration/internal/omnichannel"
	"github.com/GetStream/Vision-Agents/acceleration/internal/sandbox"
	"github.com/GetStream/Vision-Agents/acceleration/internal/stack"
	"github.com/GetStream/Vision-Agents/acceleration/internal/store"
	"github.com/GetStream/Vision-Agents/acceleration/internal/streamapp"
)

// configField is the custom field on an agent channel naming the agent config that answers in
// it: api.ConfigField, which the message hook reads to find who answers a channel no session
// ran on. The value is the hook's; the bridge cannot import the api package, which serves it.
const configField = "agent_config_id"

// agentField is the channel custom field a persistent conversation reads its agent from
// (conversation.ownedBy, support_agent_id).
const agentField = "support_agent_id"

// threadChannelPrefix starts every thread channel's id: conversation.ThreadChannelPrefix,
// the prefix a persistent conversation may be held on. Never support-, which durable session
// commands own and the message hook ignores (conversation.SessionCommandChannel). With a UUID
// after it the id is within Stream's «max length 64 characters» for a channel id
// (https://getstream.io/chat/docs/go-golang/creating_channels/).
const threadChannelPrefix = conversation.ThreadChannelPrefix

// writeTimeout bounds writing one inbound message into Stream Chat, off the request path. It
// is chatlog's writeTimeout for one write, 10 s, for three of them in a row (the author, the
// channel, the message).
const writeTimeout = 30 * time.Second

// sendTimeout bounds sending one reply to the provider: the 10 s the router gives one
// connector request (cmd/router connectorHTTPTimeout), twice, for the transport's one resend
// after a refused credential (core.Transports).
const sendTimeout = 20 * time.Second

// maxAnswerBytes caps what of a provider's answer to a reply is read, to check it was sent.
// chat.postMessage answers the posted message (https://docs.slack.dev/reference/methods/chat.postMessage),
// so 1 MiB is far over its size; the cap a 401's body is read with in core.Transports.
const maxAnswerBytes = 1 << 20

// episodeSources is the episode card source of each connector whose threads get a card in the
// person's omni-channel (T41), and how the contact map keys the author of a message in the
// provider unit. The manifest's channel block does not say what its threads are to the
// contact map, so each channel adds its connector here: slack_bot is Slack (T35), keyed by
// the workspace and the user; linq is iMessage (T36), keyed by the sender's number, so an
// author Linq names by an email address gets no card; telnyx is SMS (T53), keyed by the
// sender's number, with the carriers' keywords (keywords.go); whatsapp is WhatsApp (T51),
// keyed by the sender's number with a +, with the same keywords, which Meta answers none of.
// linq has them too since T62a (AI-921), as internal/channels' iMessage lines do; Linq answers
// none of them itself (keywords.go).
var episodeSources = map[string]episodeSource{
	"slack_bot": {source: store.EpisodeSlack, person: omnichannel.SlackUser},
	"linq":      {source: store.EpisodeIMessage, person: byNumber, optOuts: dlc.IMessage, chats: true},
	"telnyx":    {source: store.EpisodeSMS, person: byNumber, optOuts: dlc.SMS, answered: telnyxAnswered},
	"whatsapp":  {source: store.EpisodeWhatsApp, person: byWhatsAppNumber, optOuts: dlc.WhatsApp, recipient: whatsAppNumber},
}

// episodeSource is one connector's card source and the person its message's author is.
type episodeSource struct {
	source string
	person func(providerUnitID, authorID string) (omnichannel.Person, error)
	// optOuts is the opt-out channel (dlc.OptOutChannels) whose carrier keywords the bridge
	// answers before the agent and whose opted-out people it neither hands to the agent nor
	// replies to (keywords.go). Empty for a connector whose messages carry no such keywords.
	optOuts string
	// answered is whether the provider answered a keyword itself, read from the raw event, so
	// the bridge records it without answering it again (keywords.go). Nil is never.
	answered func(raw []byte) bool
	// recipient is the opt-out recipient an author id, or the thread key a reply goes to, is:
	// the number in E.164, the shape the opt-out API names (api.OptOut, «The number, in
	// E.164»). Nil is the id as the provider wrote it.
	recipient func(id string) string
	// chats is a thread key that is a chat, not the person a reply reaches: a Linq chat id
	// (linq.yaml). Its replies reach the person who started the thread (replyTo).
	chats bool
}

// recipientOf is the opt-out recipient an author id or a thread key is.
func (source episodeSource) recipientOf(id string) string {
	if source.recipient == nil {
		return id
	}
	return source.recipient(id)
}

// byNumber is the person an author's E.164 number is to the contact map.
func byNumber(_, author string) (omnichannel.Person, error) {
	return omnichannel.Phone(author)
}

// whatsAppNumber is a WhatsApp author's number in E.164. Meta writes messages[].from as the
// digits of the international number with no + («"from": "16505551234"»,
// https://developers.facebook.com/docs/whatsapp/cloud-api/webhooks/payload-examples, opened
// October 8, 2026), the rule internal/channels' e164 follows too.
func whatsAppNumber(from string) string {
	return "+" + from
}

// byWhatsAppNumber is the person a WhatsApp author's number is to the contact map.
func byWhatsAppNumber(_, author string) (omnichannel.Person, error) {
	return omnichannel.Phone(whatsAppNumber(author))
}

// Options configures a Bridge.
type Options struct {
	Store *store.Store
	// Stream is how the thread channels are written, in the Stream app each provider app is
	// pinned to.
	Stream *streamapp.Clients
	// Schemes are the registry's, by name: the one a connection names applies its credential
	// to a reply.
	Schemes map[string]core.Scheme
	// Transports builds each connection's outbound client, through which every reply leaves.
	Transports *core.Transports
	// Resolver is told when the provider refuses a reply's credential in an answer the
	// transport does not read, such as Slack's HTTP 200 invalid_auth.
	Resolver core.Resolver
	// RetryBackoff is how long to wait before each send of a reply again after one that
	// failed for a reason a later send can get past. Nil is defaultRetryBackoff.
	RetryBackoff []time.Duration
	// Gate is the sandbox and opt-out gate every text passes (dlc.Gate). A message on a
	// texting connector, one whose episode source names an opt-out channel, passes it before
	// it reaches the agent, and each reply before it is sent, as on internal/channels' lines
	// (T62a, AI-921). Nil lets everything through, as a nil *dlc.Gate does.
	Gate   *dlc.Gate
	Logger *slog.Logger
}

// defaultRetryBackoff is a choice, not a vendor's figure: three more sends within about 45 s,
// spaced so a provider that answered 5xx or 429 has time to recover. Slack posts «1 message
// per second to a specific channel» (https://docs.slack.dev/reference/methods/chat.postMessage),
// which the first wait already respects. Retry-After is not read yet.
var defaultRetryBackoff = []time.Duration{2 * time.Second, 10 * time.Second, 30 * time.Second}

// Bridge is the channel bridge. Its writes into Stream Chat and its replies run off the
// request that brought them, so Close waits for the ones in flight.
type Bridge struct {
	store      *store.Store
	stream     *streamapp.Clients
	schemes    map[string]core.Scheme
	transports *core.Transports
	resolver   core.Resolver
	// cards writes the episode card of a thread's first message into the person's
	// omni-channel.
	cards *omnichannel.Cards
	// retries are the waits before each send of a reply again (Options.RetryBackoff).
	retries []time.Duration
	gate    *dlc.Gate
	logger  *slog.Logger

	working sync.WaitGroup
	mu      sync.Mutex
	turns   map[string]*holder
}

// New validates the options and returns a Bridge.
func New(options Options) (*Bridge, error) {
	if options.Store == nil || options.Stream == nil || options.Transports == nil || options.Resolver == nil {
		return nil, stack.Wrap(errors.New("channelbridge: a store, Stream clients, transports and a resolver are required"))
	}
	retries := options.RetryBackoff
	if retries == nil {
		retries = defaultRetryBackoff
	}
	logger := options.Logger
	if logger == nil {
		logger = slog.Default()
	}
	cards, err := omnichannel.New(omnichannel.Options{Store: options.Store, Stream: options.Stream})
	if err != nil {
		return nil, err
	}
	return &Bridge{
		store:      options.Store,
		stream:     options.Stream,
		schemes:    options.Schemes,
		transports: options.Transports,
		resolver:   options.Resolver,
		cards:      cards,
		retries:    retries,
		gate:       options.Gate,
		logger:     logger,
		turns:      map[string]*holder{},
	}, nil
}

// Close waits for the writes and replies in flight.
func (b *Bridge) Close() {
	b.working.Wait()
}

// Deliver takes the messages one verified event of a provider app carried. For each it finds
// the app's connection of the provider unit and the agent that answers there, links the
// external thread to its thread channel, and claims the message by the provider's id for it,
// so a retried delivery is dropped. What it writes into Stream Chat it writes afterwards, off
// the request: a provider waits a few seconds at most (Slack: «respond ... within three
// seconds», https://docs.slack.dev/apis/events-api/).
//
// app is the provider app the event's route named, the zero record on a connector's own
// route, whose events name no customer. A message nobody can be found to answer is logged
// and dropped, since delivering it again finds nobody either. An error is the store failing,
// for the endpoint to answer 500, so the provider delivers again.
//
// answered is whether an agent answers at least one of the messages, now or already for a
// delivery the provider retried. When it is false the bridge handled none of them, and the
// events route hands the delivery to the customer's event destinations that take what the
// router does not handle (internal/eventforward, T46). A message whose write into its thread
// channel fails after the ack reaches no agent after all, so unanswered, when not nil, is
// called then, for those destinations to have it (AI-924).
//
// A message that links its thread brings the replies that waited for that link (take), which
// are handled after it, in this delivery. unanswered is not called for them: each came in a
// delivery of its own, which no agent answered then, so those destinations got that delivery
// when it arrived (AI-1001). This delivery is not the one that carries them.
func (b *Bridge) Deliver(ctx context.Context, app store.ConnectorOAuthClient, messages []core.InboundMessage, unanswered func()) (answered bool, err error) {
	queue := slices.Clone(messages)
	for i := 0; i < len(queue); i++ {
		message := queue[i]
		failed := unanswered
		if i >= len(messages) {
			failed = nil
		}
		thread, config, fresh, waited, err := b.take(ctx, app, &message)
		if err != nil {
			return false, err
		}
		queue = append(queue, waited...)
		// take links a thread only for a message an agent answers.
		answered = answered || thread.ChannelID != ""
		if !fresh {
			continue
		}
		// After the claim, so a retried keyword is answered once; before the episode, so a
		// keyword opens no card.
		handled, err := b.keyword(ctx, thread, message)
		if err != nil {
			return false, err
		}
		if handled {
			continue
		}
		// After the keywords, which reach a person whatever the sandbox says; before the
		// episode, so a message the gate holds back opens no card and reaches no agent.
		allowed, err := b.allowed(ctx, thread.CustomerID, message.ConnectorID, episodeSources[message.ConnectorID].recipientOf(message.AuthorID))
		if err != nil {
			return false, b.unclaim(ctx, thread, message, err)
		}
		if !allowed {
			continue
		}
		// After the claim, so a retried delivery opens nothing. A store that fails here
		// costs the card, not the message: the claim is taken, so the provider's next
		// delivery would be dropped.
		episode, err := b.episode(ctx, thread, config, message)
		if err != nil {
			b.logger.Error("could not open the episode of an inbound message", "connector", message.ConnectorID,
				"channel", thread.ChannelID, "error", err)
		}
		b.working.Add(1)
		go func() {
			defer b.working.Done()
			if !b.write(thread, config, episode, message) && failed != nil {
				failed()
			}
		}()
	}
	return answered, nil
}

// Reply sends an agent's finished reply in a thread channel to its external thread. The
// persistent conversation holding the channel calls it once the final text is stored
// (conversation.Service.OnFinishedReply), the one place a reply leaves. It claims the reply
// by its Stream Chat id first, so a reply written, and so told, again is sent once. A send
// that fails for a reason a later one can get past (no answer, a 5xx, a 429) is sent again
// after each of the bridge's retry backoffs. A reply not sent, after those or after a
// refusal, is unclaimed again, so a later hand-off of it sends it (AI-990 F28). A send that
// fails otherwise, such as a 2xx answer the bridge cannot read, keeps the claim: the provider
// may have posted it. It runs off the caller, which is the conversation's writer; a reply it
// cannot send is logged.
func (b *Bridge) Reply(reply conversation.FinishedReply) {
	b.working.Add(1)
	go func() {
		defer b.working.Done()
		if err := b.reply(reply); err != nil {
			b.logger.Error("could not send a reply to its external thread", "conversation", reply.CID, "error", err)
		}
	}()
}

// reply finds the thread a finished reply is in, claims it, and sends it, again on a failure
// a later send can get past.
func (b *Bridge) reply(reply conversation.FinishedReply) error {
	ctx, cancel := context.WithTimeout(context.Background(), sendTimeout)
	defer cancel()
	thread, err := b.store.ChannelThread(ctx, strings.TrimPrefix(reply.CID, chatlog.ChannelType+":"))
	if errors.Is(err, store.ErrNoChannelThread) {
		return nil
	}
	if err != nil {
		return err
	}
	if thread.CustomerID != reply.Customer {
		return stack.Wrap(fmt.Errorf("channelbridge: %s is another customer's thread channel", reply.CID))
	}
	// Nothing follows an opt-out's confirmation (keywords.go), not even a reply the agent
	// was writing when the person texted STOP; nor a reply past the sandbox's daily limit.
	if may, err := b.mayReply(ctx, thread); err != nil || !may {
		return err
	}
	text := withFiles(reply.Text, reply.Files)
	fresh, err := b.store.ClaimChannelThreadMessage(ctx, thread.ChannelID, store.ClaimReply, reply.MessageID)
	if err != nil || !fresh {
		return err
	}
	for attempt := 0; ; attempt++ {
		sending, done := context.WithTimeout(context.Background(), sendTimeout)
		err = b.send(sending, thread, text)
		done()
		if err == nil {
			b.sent(thread)
			return nil
		}
		var again retryable
		if !errors.As(err, &again) {
			var refused refusal
			if !errors.As(err, &refused) {
				// Neither refused nor failed in a way a later send gets past, such as a 2xx
				// answer the bridge cannot read: the provider may have posted it, so the claim
				// stays and no later hand-off posts it twice. Reply logs it.
				return err
			}
			break
		}
		if attempt == len(b.retries) {
			break
		}
		time.Sleep(b.retries[attempt])
		// Nor after a STOP that came in during the wait, or the limit another reply reached.
		checking, done := context.WithTimeout(context.Background(), sendTimeout)
		may, checked := b.mayReply(checking, thread)
		done()
		if checked != nil || !may {
			err = checked
			break
		}
	}
	// Not sent, so not claimed: whoever hands it over next sends it.
	releasing, done := context.WithTimeout(context.Background(), sendTimeout)
	defer done()
	if released := b.store.ReleaseChannelThreadMessage(releasing, thread.ChannelID, store.ClaimReply, reply.MessageID); released != nil {
		return errors.Join(err, released)
	}
	return err
}

// refusal is a send the provider answered and did not take: a non-2xx answer other than a 5xx
// or a 429, or a 2xx whose body says it refused the reply (ReplyRule.Accepted). It posted
// nothing, so the reply is unclaimed for a later hand-off to send.
type refusal struct{ err error }

func (r refusal) Error() string { return r.err.Error() }
func (r refusal) Unwrap() error { return r.err }

// retryable is a send that failed for a reason a later send can get past: no answer, a 5xx or
// a 429 (RFC 9110 sections 15.6 and 6585 section 4), or an answer the scheme classifies as
// transient or rate limited.
type retryable struct{ err error }

func (r retryable) Error() string { return r.err.Error() }
func (r retryable) Unwrap() error { return r.err }

// take finds who a message is for and claims it, and leaves the mention of the connection's
// account out of the message's text (MessageRule.WithoutMention). fresh is false for a message
// nobody answers and for one already taken. waited are the replies the message's thread link
// brought: when the message starts its thread, the replies that arrived before
// the link (wait). They come back through take, so their mention is left out too.
func (b *Bridge) take(ctx context.Context, app store.ConnectorOAuthClient, message *core.InboundMessage) (thread store.ChannelThread, config store.AgentConfig, fresh bool, waited []core.InboundMessage, err error) {
	if app.CustomerID == "" || message.ProviderUnitID == "" {
		b.logger.Info("dropped an inbound message that names no provider app or no provider unit",
			"connector", message.ConnectorID)
		return store.ChannelThread{}, store.AgentConfig{}, false, nil, nil
	}
	connection, err := b.store.AppConnectionByAccount(ctx, app.CustomerID, message.ConnectorID, message.ProviderUnitID)
	if errors.Is(err, store.ErrNoConnectorConnection) {
		b.logger.Info("dropped an inbound message: the provider app has no connection of its provider unit",
			"connector", message.ConnectorID, "customer", app.CustomerID, "provider_unit", message.ProviderUnitID)
		return store.ChannelThread{}, store.AgentConfig{}, false, nil, nil
	}
	if err != nil {
		return store.ChannelThread{}, store.AgentConfig{}, false, nil, err
	}
	configs, err := b.store.AgentConfigsBindingConnection(ctx, app.CustomerID, connection.ID)
	if err != nil {
		return store.ChannelThread{}, store.AgentConfig{}, false, nil, err
	}
	if len(configs) == 0 {
		// None answers a thread nobody bound: the customer's agent configs to fix. A test copy
		// is not listed, so it never answers (AI-1049).
		b.logger.Warn("dropped an inbound message: one agent config must bind the connection it came in on",
			"connector", message.ConnectorID, "customer", app.CustomerID, "connection", connection.ID, "configs", len(configs))
		return store.ChannelThread{}, store.AgentConfig{}, false, nil, nil
	}
	if len(configs) > 1 {
		// Two agents answering one thread would talk over each other, so the oldest answers: the
		// connection's owner. A save no longer lets a second live config bind it
		// (store.ChannelConnectionTakenError), so only configs that shared it before that rule
		// get here, and the others are named for the customer to unbind (AI-1049).
		others := make([]string, 0, len(configs)-1)
		for _, other := range configs[1:] {
			others = append(others, other.ID)
		}
		b.logger.Warn("more than one agent config binds the connection a message came in on: the oldest answers",
			"connector", message.ConnectorID, "customer", app.CustomerID, "connection", connection.ID,
			"config", configs[0].ID, "not_answering", others)
	}
	read, rule, err := b.read(ctx, connection, *message)
	if err != nil {
		return store.ChannelThread{}, store.AgentConfig{}, false, nil, err
	}
	// A message that does not speak to the connection's own account, such as one in a Slack
	// channel that does not mention the bot, starts no thread: it is answered only on a
	// thread a message that did linked before (AI-989).
	if !rule.Addresses(read, connection.Metadata) {
		linked, err := b.store.ChannelThreadLinked(ctx, app.CustomerID, message.ConnectorID, message.ProviderUnitID, message.ThreadKey)
		if err == nil && !linked && !startsThread(read) {
			linked, err = b.wait(ctx, app.CustomerID, *message)
		}
		if err != nil {
			return store.ChannelThread{}, store.AgentConfig{}, false, nil, err
		}
		if !linked {
			b.logger.Debug("dropped an inbound message that is not addressed to the connection on a thread nobody linked",
				"connector", message.ConnectorID, "connection", connection.ID)
			return store.ChannelThread{}, store.AgentConfig{}, false, nil, nil
		}
	}
	message.Text = rule.WithoutMention(read.Text, connection.Metadata)
	thread = store.ChannelThread{
		ChannelID:      threadChannelPrefix + uuid.NewString(),
		CustomerID:     app.CustomerID,
		ConnectorID:    message.ConnectorID,
		ProviderUnitID: message.ProviderUnitID,
		ThreadKey:      message.ThreadKey,
		ConnectionID:   connection.ID,
		ThreadParts:    read.ThreadParts,
		StreamAppPK:    app.StreamAppPK,
	}
	_, err = b.store.LinkChannelThread(ctx, &thread)
	if err != nil {
		return store.ChannelThread{}, store.AgentConfig{}, false, nil, err
	}
	fresh, err = b.store.ClaimChannelThreadMessage(ctx, thread.ChannelID, store.ClaimInbound, message.ProviderMessageID)
	if err != nil {
		return store.ChannelThread{}, store.AgentConfig{}, false, nil, err
	}
	if !fresh {
		b.logger.Debug("dropped a retried inbound message", "connector", message.ConnectorID, "channel", thread.ChannelID)
	}
	// Every reply in a thread comes after the message that started it, so the replies that
	// waited for that message are all to be answered. A link made by a reply, such as a
	// mention in a thread of people, leaves them waiting until they are dropped: some came
	// before the bot was spoken to, which it does not read (AI-989). The message takes them
	// whether or not it made the link now and whether or not it was taken before: a first
	// delivery that linked the thread and failed before it took them leaves them to the
	// provider's retry, and each waiting reply is handed out once.
	if startsThread(read) {
		waited, err = b.store.TakeWaitingChannelThreadMessages(ctx, app.CustomerID, message.ConnectorID, message.ProviderUnitID, message.ThreadKey)
		if err != nil {
			return store.ChannelThread{}, store.AgentConfig{}, false, nil, err
		}
	}
	return thread, configs[0], fresh, waited, nil
}

// wait keeps a reply that does not speak to the connection's account, on a thread nobody
// linked, for the message that links the thread to take (AI-990 F31a): a mention Slack retries
// after its first delivery failed arrives after the replies in its thread. linked is whether
// the thread was linked meanwhile and this reply took itself back, so it is answered now; the
// message that linked it did not see it, or took it first, in which case linked is false and
// that message answers it. The second look is after the reply is kept, so one of the two
// always finds it.
func (b *Bridge) wait(ctx context.Context, customerID string, message core.InboundMessage) (linked bool, err error) {
	if _, err := b.store.WaitChannelThreadMessage(ctx, customerID, message); err != nil {
		return false, err
	}
	linked, err = b.store.ChannelThreadLinked(ctx, customerID, message.ConnectorID, message.ProviderUnitID, message.ThreadKey)
	if err != nil || !linked {
		return false, err
	}
	return b.store.UnwaitChannelThreadMessage(ctx, customerID, message)
}

// startsThread is whether a message is the first of its thread: its own id is a part of its
// thread key, as a Slack message that starts a thread is keyed by its own ts (slack_bot.yaml,
// thread_ts falls back to the message's ts).
func startsThread(message core.ChannelMessage) bool {
	for _, part := range message.ThreadParts {
		if part == message.ProviderMessageID {
			return true
		}
	}
	return false
}

// episode opens the episode a message is in, in the omni-channel of the person who wrote it:
// the thread's first message opens it and the later ones find it open, so a thread has one
// card, in the omni-channel of whoever started it. A connector with no episode source, or an
// author the contact map cannot key, gets no card; the message is answered all the same.
func (b *Bridge) episode(ctx context.Context, thread store.ChannelThread, config store.AgentConfig, message core.InboundMessage) (omnichannel.Opened, error) {
	source, carded := episodeSources[message.ConnectorID]
	if !carded {
		return omnichannel.Opened{}, nil
	}
	person, err := source.person(message.ProviderUnitID, message.AuthorID)
	if err != nil {
		b.logger.Info("no episode card for a message whose author the contact map cannot key",
			"connector", message.ConnectorID, "channel", thread.ChannelID)
		return omnichannel.Opened{}, nil
	}
	return b.cards.Open(ctx, omnichannel.Episode{
		CustomerID:    thread.CustomerID,
		AgentConfigID: config.ID,
		AgentName:     config.Name,
		Person:        person,
		Source:        source.source,
		ThreadChannel: chatlog.ChannelType + ":" + thread.ChannelID,
		StreamAppPK:   thread.StreamAppPK,
	})
}

// read reads a message again from the raw body with the connection's manifest, by the
// provider's id for it (core.ChannelMessage), with the rule that read it: the verifier hands
// over the thread key alone, and a reply names the key's parts; whether the message
// addresses the connection is that revision's to say.
func (b *Bridge) read(ctx context.Context, connection store.ConnectorConnection, message core.InboundMessage) (core.ChannelMessage, core.MessageRule, error) {
	definition, err := b.store.ConnectorDefinition(ctx, connection.CustomerID, connection.ConnectorID, connection.DefinitionRevision)
	if err != nil {
		return core.ChannelMessage{}, core.MessageRule{}, err
	}
	if definition.Manifest.Channel == nil {
		return core.ChannelMessage{}, core.MessageRule{}, stack.Wrap(fmt.Errorf("channelbridge: revision %d of %s has no channel block", connection.DefinitionRevision, connection.ConnectorID))
	}
	read, err := definition.Manifest.Channel.Read(message.ConnectorID, message.Raw)
	if err != nil {
		return core.ChannelMessage{}, core.MessageRule{}, stack.Wrap(err)
	}
	for _, found := range read.Messages {
		if found.ProviderMessageID == message.ProviderMessageID && found.ThreadKey == message.ThreadKey {
			return found, definition.Manifest.Channel.Messages, nil
		}
	}
	return core.ChannelMessage{}, core.MessageRule{}, stack.Wrap(fmt.Errorf("channelbridge: revision %d of %s does not read message %s the way the event's did",
		connection.DefinitionRevision, connection.ConnectorID, message.ProviderMessageID))
}

// write writes one claimed message into its thread channel as its author, creating the
// channel the first time, with the agent config that answers it. One thread's messages are
// written one at a time, in the order they were taken as far as the lock keeps it. It reports
// whether the message was written.
func (b *Bridge) write(thread store.ChannelThread, config store.AgentConfig, episode omnichannel.Opened, message core.InboundMessage) bool {
	release := b.hold(thread.ChannelID)
	defer release()
	ctx, cancel := context.WithTimeout(context.Background(), writeTimeout)
	defer cancel()

	err := b.writeInto(ctx, thread, config, message)
	if err != nil {
		b.logger.Error("could not write an inbound message into its thread channel",
			"connector", thread.ConnectorID, "channel", thread.ChannelID, "error", err)
	}
	if err := b.cards.Write(ctx, episode); err != nil {
		b.logger.Error("could not write the episode card of a thread",
			"connector", thread.ConnectorID, "channel", thread.ChannelID, "error", err)
	}
	return err == nil
}

func (b *Bridge) writeInto(ctx context.Context, thread store.ChannelThread, config store.AgentConfig, message core.InboundMessage) error {
	bound, err := b.stream.ForApp(ctx, thread.CustomerID, thread.StreamAppPK)
	if err != nil {
		return err
	}
	author := authorUserID(thread.CustomerID, thread.ConnectorID, thread.ProviderUnitID, message.AuthorID)
	name, agentName := message.AuthorID, config.Name
	// The agent of a thread channel writes as a user of the channel's own id: the
	// conversation held on the channel writes as support_agent_id. The message hook finds
	// the session answering in the channel by its conversation (Manager.ByConversationWhere).
	if err := conversation.CreateMissingUsers(ctx, bound.Client, map[string]getstream.UserRequest{
		author:           {ID: author, Name: &name},
		thread.ChannelID: {ID: thread.ChannelID, Name: &agentName},
	}); err != nil {
		return stack.Wrap(err)
	}
	// The data creates the channel on the thread's first message; the agent channel type
	// refuses a server-side create without a creator. The stamps are the ones a persistent
	// conversation is opened by (conversation.ownedBy): the customer and the agent, and no
	// owner, since no one end user owns a thread several people write in.
	_, err = bound.Client.Chat().GetOrCreateChannel(ctx, chatlog.ChannelType, thread.ChannelID, &getstream.GetOrCreateChannelRequest{
		Data: &getstream.ChannelInput{
			CreatedByID: &author,
			Custom: map[string]any{
				configField:                config.ID,
				conversation.CustomerField: thread.CustomerID,
				agentField:                 thread.ChannelID,
			},
		},
	})
	if err != nil {
		return stack.Wrap(err)
	}
	// No source: the message hook answers only a message without one (api.addressed).
	_, err = bound.Client.Chat().SendMessage(ctx, chatlog.ChannelType, thread.ChannelID, &getstream.SendMessageRequest{
		Message: getstream.MessageRequest{Text: &message.Text, UserID: &author},
	})
	return stack.Wrap(err)
}

// withFiles is a reply's text with the links of the files the agent made for it, one a line
// after a blank one, so they reach the external thread on every provider: the manifests' reply
// templates carry text alone (slack_bot.yaml, linq.yaml, telnyx.yaml, whatsapp.yaml), and a
// link is what internal/channels sends where a provider takes no media. The links are where
// the conversation stored them in Stream Chat (conversation.Publish). Native media per
// provider is a later ticket.
func withFiles(text string, files []sandbox.Attachment) string {
	links := make([]string, 0, len(files))
	for _, file := range files {
		if file.URL != "" {
			links = append(links, file.URL)
		}
	}
	if len(links) == 0 {
		return text
	}
	if text == "" {
		return strings.Join(links, "\n")
	}
	return text + "\n\n" + strings.Join(links, "\n")
}

// send posts one reply through the connection's manifest reply template and transport.
func (b *Bridge) send(ctx context.Context, thread store.ChannelThread, text string) error {
	connection, err := b.store.ConnectorConnection(ctx, thread.CustomerID, thread.ConnectionID)
	if err != nil {
		return err
	}
	scheme, found := b.schemes[connection.AuthScheme]
	if !found {
		return stack.Wrap(fmt.Errorf("%w: %q", store.ErrUnregisteredScheme, connection.AuthScheme))
	}
	definition, err := b.store.ConnectorDefinition(ctx, connection.CustomerID, connection.ConnectorID, connection.DefinitionRevision)
	if err != nil {
		return err
	}
	resolved, err := definition.Manifest.Resolve(connection.AuthScheme, connection.Inputs, connection.Metadata)
	if err != nil {
		return stack.Wrap(err)
	}
	target, body, err := resolved.Reply(core.ReplyValues{
		Text:           text,
		ProviderUnitID: thread.ProviderUnitID,
		ThreadParts:    thread.ThreadParts,
		SinceInbound:   time.Since(thread.LastInboundAt),
	})
	if err != nil {
		return err
	}
	request, err := http.NewRequestWithContext(ctx, http.MethodPost, target, bytes.NewReader(body))
	if err != nil {
		return stack.Wrap(err)
	}
	request.Header.Set("Content-Type", "application/json")
	ref := core.ConnectionRef{CustomerID: connection.CustomerID, ConnectionID: connection.ID}
	response, err := b.transports.Client(ref, scheme).Do(request)
	if err != nil {
		return retryable{stack.Wrap(err)}
	}
	defer response.Body.Close() //nolint:errcheck // the answer has been read
	answer, err := io.ReadAll(io.LimitReader(response.Body, maxAnswerBytes))
	if err != nil {
		return retryable{stack.Wrap(err)}
	}
	if response.StatusCode >= http.StatusInternalServerError || response.StatusCode == http.StatusTooManyRequests {
		return retryable{stack.Wrap(fmt.Errorf("channelbridge: %s answered a reply with %d", connection.ConnectorID, response.StatusCode))}
	}
	if response.StatusCode < 200 || response.StatusCode > 299 {
		return refusal{stack.Wrap(fmt.Errorf("channelbridge: %s answered a reply with %d", connection.ConnectorID, response.StatusCode))}
	}
	sent, err := resolved.Channel.Reply.Accepts(answer)
	if err != nil {
		return stack.Wrap(fmt.Errorf("channelbridge: %s answered a reply with a body it does not read: %w", connection.ConnectorID, err))
	}
	if !sent {
		refused := stack.Wrap(fmt.Errorf("channelbridge: %s refused a reply in a %d answer%s", connection.ConnectorID, response.StatusCode, refusalCode(answer)))
		switch b.refused(ctx, ref, scheme, response, answer) {
		case core.OutcomeTransient, core.OutcomeRateLimited:
			return retryable{refused}
		}
		return refusal{refused}
	}
	return nil
}

// refusalCode names, for the log, the error a provider refused a reply with in its answer's
// top-level error member, such as Slack's «"ok": false, "error": "not_in_channel"»
// (https://docs.slack.dev/reference/methods/chat.postMessage, opened October 9, 2026), or
// nothing when the answer has no such member.
func refusalCode(answer []byte) string {
	var refusal struct {
		Error string `json:"error"`
	}
	if json.Unmarshal(answer, &refusal) != nil || refusal.Error == "" {
		return ""
	}
	return fmt.Sprintf(", error %q", refusal.Error)
}

// refused tells the resolver when the provider refused a reply's credential in an answer the
// transport does not read: it reads only a 401, and Slack refuses a revoked token with HTTP
// 200 and «"ok": false, "error": "invalid_auth"» (chat.postMessage). The scheme's own
// Classify reads the answer, as it does any provider answer. The credential is the one the
// resolver hands out now, the same one the reply was sent with unless another router
// renewed it meanwhile, in which case Invalidate leaves the connection as it is.
//
// It returns what the scheme read the refusal as, so the caller can tell one a later send
// can get past.
func (b *Bridge) refused(ctx context.Context, ref core.ConnectionRef, scheme core.Scheme, response *http.Response, answer []byte) core.OutcomeKind {
	outcome := scheme.Classify(response, answer, nil)
	if outcome.Kind != core.OutcomeInvalidGrant && outcome.Kind != core.OutcomeScopeRequired {
		return outcome.Kind
	}
	sent, err := b.resolver.Resolve(ctx, ref, core.CredentialRequest{})
	if err != nil {
		return outcome.Kind
	}
	if err := b.resolver.Invalidate(ctx, ref, sent, outcome); err != nil {
		b.logger.Error("could not mark a connection whose reply was refused", "connection", ref.ConnectionID, "error", err)
	}
	return outcome.Kind
}

// authorUserID is the Stream Chat user an external author writes as in a thread channel: one
// for each customer, connector, provider unit and author id. It is a digest so that any
// provider's ids fit the characters every Stream id here keeps to ([A-Za-z0-9_-]), and so
// that one person in two customers' workspaces is two users, never one that two customers
// write as. Who the person is across channels is the contact map's (T43), not this.
func authorUserID(customerID, connectorID, providerUnitID, authorID string) string {
	sum := sha256.Sum256([]byte(customerID + "\x00" + connectorID + "\x00" + providerUnitID + "\x00" + authorID))
	return connectorID + "-" + hex.EncodeToString(sum[:16])
}

// hold takes one thread channel's turn, so its messages are written one at a time.
func (b *Bridge) hold(key string) func() {
	b.mu.Lock()
	held, waiting := b.turns[key]
	if !waiting {
		held = &holder{}
		b.turns[key] = held
	}
	held.waiting++
	b.mu.Unlock()

	held.lock.Lock()
	return func() {
		held.lock.Unlock()
		b.mu.Lock()
		held.waiting--
		if held.waiting == 0 {
			delete(b.turns, key)
		}
		b.mu.Unlock()
	}
}

// holder is one thread channel's turn, counted so the last one out forgets it.
type holder struct {
	lock    sync.Mutex
	waiting int
}
