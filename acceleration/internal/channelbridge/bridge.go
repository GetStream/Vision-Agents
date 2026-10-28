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
	"strings"
	"sync"
	"time"

	getstream "github.com/GetStream/getstream-go/v5"
	"github.com/google/uuid"

	"github.com/GetStream/Vision-Agents/acceleration/internal/chatlog"
	"github.com/GetStream/Vision-Agents/acceleration/internal/connectors/core"
	"github.com/GetStream/Vision-Agents/acceleration/internal/conversation"
	"github.com/GetStream/Vision-Agents/acceleration/internal/dlc"
	"github.com/GetStream/Vision-Agents/acceleration/internal/eventforward"
	"github.com/GetStream/Vision-Agents/acceleration/internal/omnichannel"
	"github.com/GetStream/Vision-Agents/acceleration/internal/sandbox"
	"github.com/GetStream/Vision-Agents/acceleration/internal/stack"
	"github.com/GetStream/Vision-Agents/acceleration/internal/store"
	"github.com/GetStream/Vision-Agents/acceleration/internal/streamapp"
)

// configField is the custom field on an agent channel naming the agent config that answers in
// it: api.ConfigField, which the message hook reads to find who answers a plain channel no
// session ran on. A thread channel is answered by the line's current binding instead, so the
// value is the config the thread opened under. The value is the hook's; the bridge cannot
// import the api package, which serves it.
const configField = "agent_config_id"

// agentField is the channel custom field a persistent conversation reads its agent from
// (conversation.ownedBy, support_agent_id).
const agentField = "support_agent_id"

// BridgedField is the custom field the bridge sets, to true, on every person's message it
// writes into a thread channel: the message hook skips a message that has it, since the
// worker hands that message to the agent itself (Answer). Stream Chat keeps a message's custom
// fields and its message.new carries them at the top level of the message, as it does the
// source the agent writes on its own (chatlog.SourceField), which the hook reads the same way.
const BridgedField = "bridged"

// Written is a person's message as the worker wrote it into its thread channel: what the
// agent is handed (Answer).
type Written struct {
	// ID is its Stream message id, which its turn is claimed by.
	ID   string
	Text string
	// Author is the Stream user the person writes as (AuthorUserID).
	Author      string
	Attachments []getstream.Attachment
}

// Answer hands the agent a message the worker wrote into thread (api.Server.AnswerQueued). It
// claims the message's turn and returns the turn, which the bridge runs off the worker's slot,
// as a reply waits up to minutes; nil is a message whose turn was claimed already. An error
// is the claim failing: the worker tries the message again, and Stream Chat refuses the
// write as already done, so the next attempt claims it.
type Answer func(ctx context.Context, thread store.ChannelThread, message Written) (turn func(), err error)

// threadChannelPrefix starts every thread channel's id: conversation.ThreadChannelPrefix,
// the prefix a persistent conversation may be held on. Never support-, which durable session
// commands own and the message hook ignores (conversation.SessionCommandChannel). With a UUID
// after it the id is within Stream's «max length 64 characters» for a channel id
// (https://getstream.io/chat/docs/go-golang/creating_channels/).
const threadChannelPrefix = conversation.ThreadChannelPrefix

// writeTimeout bounds one attempt of a queued inbound message, off the request path: admitting
// it and writing it into Stream Chat. It is chatlog's writeTimeout for one write, 10 s, for
// three of them in a row (the author, the channel, the message); a keyword's answer, the one
// send an attempt may make instead, takes sendTimeout of it.
const writeTimeout = 30 * time.Second

// sendTimeout bounds sending one reply to the provider: the 10 s the router gives one
// connector request (cmd/router connectorHTTPTimeout), twice, for the transport's one resend
// after a refused credential (core.Transports).
const sendTimeout = 20 * time.Second

// defaultQueueTimeout bounds the transaction that claims and queues a message, which Deliver
// finishes off the request (detached). It is chatlog's writeTimeout for one write, 10 s; the
// provider, which waits a few seconds, may have hung up by then, and its retry finds the
// message queued or unclaimed.
const defaultQueueTimeout = 10 * time.Second

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
	"linq":      {source: store.EpisodeIMessage, person: byNumber, optOuts: dlc.IMessage, chats: true, oneToOne: linqOneToOne},
	"telnyx":    {source: store.EpisodeSMS, person: byNumber, optOuts: dlc.SMS, answered: telnyxAnswered},
	"whatsapp":  {source: store.EpisodeWhatsApp, person: byWhatsAppNumber, optOuts: dlc.WhatsApp, recipient: whatsAppNumber, window: whatsAppWindow},
}

// linqOneToOne is whether a Linq event's chat is one person's: data.chat.is_group false, as the
// 2026-02-03 payload in linq.yaml's [events] shows it. That page shows the field in its example
// only, so an event that leaves it out, or writes it as anything but a boolean, is taken as a
// group's: a task scheduled there would reach everyone in the chat.
func linqOneToOne(raw []byte) bool {
	var event struct {
		Data struct {
			Chat struct {
				IsGroup *bool `json:"is_group"`
			} `json:"chat"`
		} `json:"data"`
	}
	return json.Unmarshal(raw, &event) == nil && event.Data.Chat.IsGroup != nil && !*event.Data.Chat.IsGroup
}

// whatsAppWindow is Meta's customer service window: free text may be sent within 24 hours of
// the person's last message, and only an approved template after it
// (https://developers.facebook.com/docs/whatsapp/cloud-api/guides/send-messages#customer-service-windows),
// internal/channels' whatsappWindow.
const whatsAppWindow = 24 * time.Hour

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
	// (linq.yaml). Its replies reach the person who started the thread (replyTo) and everyone
	// else in the chat, its members as far as their messages show them
	// (store.AddChannelThreadMember), so one of them who opted out silences it (refusesReply).
	chats bool
	// oneToOne is whether a chat's raw event says the chat is one person's, not a group, so the
	// thread records its author as the chat's person (store.SetChannelThreadChatPerson). Nil is
	// never.
	oneToOne func(raw []byte) bool
	// window is how long after the person's last message the provider takes free text, when
	// it limits that; zero is no limit. A manifest whose reply has an after_window template
	// still reaches the person past it (Reachable).
	window time.Duration
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
	Gate *dlc.Gate
	// InboundRetries are the waits before each write again of a queued inbound message whose
	// write failed (Start). Nil is defaultInboundRetries.
	InboundRetries []time.Duration
	// InboundPoll is the shortest wait between looks for queued inbound messages while some are
	// queued. Zero is defaultInboundPoll. Tests shorten it.
	InboundPoll time.Duration
	// QueueTimeout bounds the transaction that claims and queues a message, which Deliver
	// finishes off the request (detached). Zero is defaultQueueTimeout. Tests shorten it.
	QueueTimeout time.Duration
	// Unanswered queues a delivery for the customer's event destinations of unhandled events
	// (eventforward.Forwarder.ForwardUnanswered), for a queued message the worker gave up. Nil
	// forwards nothing.
	Unanswered func(ctx context.Context, event eventforward.Event) error
	Logger     *slog.Logger
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

	// inboundRetries, inboundPoll and unanswered are the worker's (Options).
	inboundRetries []time.Duration
	inboundPoll    time.Duration
	unanswered     func(ctx context.Context, event eventforward.Event) error
	// answer hands the agent each message the worker writes (Start).
	answer Answer
	// queueTimeout bounds the claim and queue transaction Deliver finishes off the request
	// (Options).
	queueTimeout time.Duration
	// kick wakes the worker: a message was queued, or a slot freed.
	kick chan struct{}
	// slots holds one token for each queued message being written.
	slots chan struct{}
	// pruned is when the worker last deleted the rows past their keep.
	pruned time.Time
	// ctx ends the worker on Close.
	ctx    context.Context
	cancel context.CancelFunc

	working sync.WaitGroup
	mu      sync.Mutex
	// held are the replies callers send themselves (Hold), by the command they answer.
	held map[string]chan conversation.FinishedReply
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
	inboundRetries := options.InboundRetries
	if inboundRetries == nil {
		inboundRetries = defaultInboundRetries
	}
	inboundPoll := options.InboundPoll
	if inboundPoll <= 0 {
		inboundPoll = defaultInboundPoll
	}
	queueTimeout := options.QueueTimeout
	if queueTimeout <= 0 {
		queueTimeout = defaultQueueTimeout
	}
	cards, err := omnichannel.New(omnichannel.Options{Store: options.Store, Stream: options.Stream})
	if err != nil {
		return nil, err
	}
	ctx, cancel := context.WithCancel(context.Background())
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
		held:       map[string]chan conversation.FinishedReply{},

		inboundRetries: inboundRetries,
		inboundPoll:    inboundPoll,
		unanswered:     options.Unanswered,
		queueTimeout:   queueTimeout,
		kick:           make(chan struct{}, 1),
		slots:          make(chan struct{}, inboundInFlight),
		ctx:            ctx,
		cancel:         cancel,
	}, nil
}

// Close stops the worker (Start) and waits for the writes, the turns they handed the agent and
// the replies in flight. A reply handed over after it is still sent.
func (b *Bridge) Close() {
	b.cancel()
	b.working.Wait()
}

// Deliver takes the messages one verified event of a provider app carried. For each it finds
// the app's connection of the provider unit and the agent that answers there, links the
// external thread to its thread channel, and claims and queues the message by the provider's
// id for it in one transaction (store.AcceptChannelInbound), so a retried delivery is dropped
// and a claimed message is never left without its row. A worker on any router admits and
// writes it into Stream Chat with its episode card (Start), off the request: a provider waits
// a few seconds at most (Slack: «respond ... within three seconds»,
// https://docs.slack.dev/apis/events-api/), and a queued message outlives the router.
//
// app is the provider app the event's route named, the zero record on a connector's own
// route, whose events name no customer. A message nobody can be found to answer is logged
// and dropped, since delivering it again finds nobody either. An error is the store failing,
// for the endpoint to answer 500, so the provider delivers again.
//
// answered is whether an agent answers at least one of the messages, now or already for a
// delivery the provider retried. When it is false the bridge handled none of them, and the
// events route hands the delivery to the customer's event destinations that take what the
// router does not handle (internal/eventforward, T46). A message the worker gives up after the
// ack reaches no agent after all, so unanswered, when not nil, is stored with each message
// and the worker forwards it then, for those destinations to have it (AI-924). Its body is
// the message's Raw; its customer and connector are the thread's.
//
// A message that starts its thread takes the replies that waited for its link (take), which are
// queued after it, in the same transaction, so a failure leaves them waiting for the provider's
// retry. They are queued with no forward: each came in a delivery of its own, which no agent
// answered then, so those destinations got that delivery when it arrived (AI-1001). This
// delivery is not the one that carries them.
func (b *Bridge) Deliver(ctx context.Context, app store.ConnectorOAuthClient, messages []core.InboundMessage, unanswered *eventforward.Event) (answered bool, err error) {
	var forward *store.ChannelInboundForward
	if unanswered != nil {
		forward = &store.ChannelInboundForward{ID: unanswered.ID, Headers: unanswered.Headers}
		if !unanswered.HeadersUntil.IsZero() {
			forward.HeadersUntil = &unanswered.HeadersUntil
		}
	}
	for _, message := range messages {
		thread, config, waiting, err := b.take(ctx, app, &message)
		if err != nil {
			return false, err
		}
		// take links a thread only for a message an agent answers.
		if thread.ChannelID == "" {
			continue
		}
		answered = true
		b.logger.Info("accepted an inbound message", "connector", message.ConnectorID, "provider_app", app.ProviderAppID,
			"customer", thread.CustomerID, "config", config.ID, "channel", thread.ChannelID)
		var waited func(core.InboundMessage) (store.ChannelInbound, error)
		if waiting != nil {
			waited = func(reply core.InboundMessage) (store.ChannelInbound, error) {
				reply, err := waiting(reply)
				return inboundRow(thread, config, reply, nil), err
			}
		}
		// Off the request, so a provider that hangs up meanwhile leaves the message queued, or
		// unclaimed, never claimed with no row.
		accepting, cancel := b.detached(ctx)
		fresh, err := b.store.AcceptChannelInbound(accepting, inboundRow(thread, config, message, forward), waited)
		cancel()
		if err != nil {
			return false, err
		}
		if !fresh {
			b.logger.Info("dropped a retried inbound message", "connector", message.ConnectorID, "provider_app", app.ProviderAppID,
				"customer", app.CustomerID, "connection", thread.ConnectionID, "config", config.ID, "channel", thread.ChannelID)
		} else {
			b.logger.Info("queued an inbound message", "connector", message.ConnectorID, "provider_app", app.ProviderAppID,
				"customer", thread.CustomerID, "config", config.ID, "channel", thread.ChannelID)
		}
		b.wake()
	}
	return answered, nil
}

// inboundRow is the queue row of a message in thread, due now.
func inboundRow(thread store.ChannelThread, config store.AgentConfig, message core.InboundMessage, forward *store.ChannelInboundForward) store.ChannelInbound {
	return store.ChannelInbound{
		ChannelID: thread.ChannelID, ProviderMessageID: message.ProviderMessageID, CustomerID: thread.CustomerID,
		ConnectorID: message.ConnectorID, ProviderUnitID: message.ProviderUnitID, ThreadKey: message.ThreadKey,
		ConfigID: config.ID, ConfigName: config.Name, AuthorID: message.AuthorID, Text: message.Text,
		Media: message.Media, Raw: message.Raw, Forward: forward, NextAttemptAt: time.Now(),
	}
}

// admit runs the steps a queued message passes before it is written into its thread channel:
// it records the chat's person and member, answers a keyword, and asks the sandbox gate.
// admitted is whether the message goes on to be written, for an agent to answer. It reads
// only the queued row and the row's thread channel, never the chat person the thread arrives
// with (store.ChannelThread brings the stored one), so an attempt of the row again decides as
// the first one did. A store that fails is the attempt failing: the worker tries the row again
// after its next wait, and each step before the failure is one a second run leaves as it is.
func (b *Bridge) admit(ctx context.Context, thread store.ChannelThread, queued store.ChannelInbound) (admitted bool, err error) {
	message := queued.Message()
	// Only for a claimed message, so a retried delivery of an older message, such as a 1:1 one
	// after the chat became a group, does not name the chat's person again.
	if source := episodeSources[message.ConnectorID]; source.oneToOne != nil {
		// The latest message decides, not the person the thread arrives with.
		thread.ChatPerson = ""
		if source.oneToOne(message.Raw) {
			thread.ChatPerson = message.AuthorID
		}
		if err := b.store.SetChannelThreadChatPerson(ctx, thread.ChannelID, thread.ChatPerson); err != nil {
			return false, err
		}
	}
	// Before the keywords, so a person who opted out silences the chat with their STOP and with
	// any message after it (refusesReply). Every author counts, whatever data.chat.is_group
	// says: a reply reaches whoever is in the chat.
	if source := episodeSources[message.ConnectorID]; source.chats {
		if err := b.store.AddChannelThreadMember(ctx, thread.ChannelID, source.recipientOf(message.AuthorID)); err != nil {
			return false, err
		}
	}
	// Before the episode, so a keyword opens no card.
	handled, err := b.keyword(ctx, thread, queued)
	if err != nil || handled {
		return false, err
	}
	// After the keywords, which reach a person whatever the sandbox says; before the
	// episode, so a message the gate holds back opens no card and reaches no agent.
	return b.allowed(ctx, thread.CustomerID, message.ConnectorID, episodeSources[message.ConnectorID].recipientOf(message.AuthorID))
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
	b.mu.Lock()
	held, holding := b.held[reply.CommandID]
	delete(b.held, reply.CommandID)
	b.mu.Unlock()
	if holding {
		held <- reply
		return
	}
	b.working.Add(1)
	go func() {
		defer b.working.Done()
		// A reply nobody may be sent (refusesReply), or one whose thread is gone, is no
		// failure to log.
		if err := b.reply(reply); err != nil && !errors.Is(err, ErrOptedOut) && !errors.Is(err, ErrHeldBack) &&
			!errors.Is(err, store.ErrNoChannelThread) {
			b.logger.Error("could not send a reply to its external thread", "conversation", reply.CID, "error", err)
		}
	}()
}

// Hold hands the reply that answers commandID to the caller instead of sending it, so the
// caller sends it (Send) and learns whether it went out, as a scheduled task's delivery
// records. release stops holding it.
func (b *Bridge) Hold(commandID string) (replies <-chan conversation.FinishedReply, release func()) {
	held := make(chan conversation.FinishedReply, 1)
	b.mu.Lock()
	b.held[commandID] = held
	b.mu.Unlock()
	return held, func() {
		b.mu.Lock()
		defer b.mu.Unlock()
		if b.held[commandID] == held {
			delete(b.held, commandID)
		}
	}
}

// Send sends a reply as Reply does, on the caller, and answers why it did not go out:
// ErrOptedOut or ErrHeldBack, store.ErrNoChannelThread for a thread no longer linked, the
// provider refusing it, or the store failing. A reply sent already is nil.
func (b *Bridge) Send(reply conversation.FinishedReply) error {
	return b.reply(reply)
}

// reply finds the thread a finished reply is in, claims it, and sends it, again on a failure
// a later send can get past.
func (b *Bridge) reply(reply conversation.FinishedReply) error {
	ctx, cancel := context.WithTimeout(context.Background(), sendTimeout)
	defer cancel()
	thread, err := b.store.ChannelThread(ctx, strings.TrimPrefix(reply.CID, chatlog.ChannelType+":"))
	if err != nil {
		return err
	}
	if thread.CustomerID != reply.Customer {
		return stack.Wrap(fmt.Errorf("channelbridge: %s is another customer's thread channel", reply.CID))
	}
	// Nothing follows an opt-out's confirmation (keywords.go), not even a reply the agent
	// was writing when the person texted STOP; nor a reply past the sandbox's daily limit.
	if err := b.refusesReply(ctx, thread); err != nil {
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
		err = b.refusesReply(checking, thread)
		done()
		if err != nil {
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

// take finds who a message is for, links its thread channel, and leaves the mention of the
// connection's account out of the message's text (MessageRule.WithoutMention). thread is the
// zero one for a message nobody answers. waiting, for a message that starts its thread, reads
// a reply that waited for the link (wait) as take reads this message, so its mention is left
// out too; it is nil for any other message.
func (b *Bridge) take(ctx context.Context, app store.ConnectorOAuthClient, message *core.InboundMessage) (thread store.ChannelThread, config store.AgentConfig, waiting func(core.InboundMessage) (core.InboundMessage, error), err error) {
	if app.CustomerID == "" || message.ProviderUnitID == "" {
		b.logger.Info("dropped an inbound message that names no provider app or no provider unit",
			"connector", message.ConnectorID)
		return store.ChannelThread{}, store.AgentConfig{}, nil, nil
	}
	connection, err := b.store.AppConnectionByAccount(ctx, app.CustomerID, message.ConnectorID, message.ProviderUnitID)
	if errors.Is(err, store.ErrNoConnectorConnection) {
		b.logger.Info("dropped an inbound message: the provider app has no connection of its provider unit",
			"connector", message.ConnectorID, "customer", app.CustomerID, "provider_unit", message.ProviderUnitID)
		return store.ChannelThread{}, store.AgentConfig{}, nil, nil
	}
	if err != nil {
		return store.ChannelThread{}, store.AgentConfig{}, nil, err
	}
	configs, err := b.store.AgentConfigsBindingConnection(ctx, app.CustomerID, connection.ID)
	if err != nil {
		return store.ChannelThread{}, store.AgentConfig{}, nil, err
	}
	if len(configs) == 0 {
		// None answers a thread nobody bound: the customer's agent configs to fix. A test copy
		// is not listed, so it never answers (AI-1049).
		b.logger.Warn("dropped an inbound message: one agent config must bind the connection it came in on",
			"connector", message.ConnectorID, "customer", app.CustomerID, "connection", connection.ID, "configs", len(configs))
		return store.ChannelThread{}, store.AgentConfig{}, nil, nil
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
	definition, err := b.store.ConnectorDefinition(ctx, connection.CustomerID, connection.ConnectorID, connection.DefinitionRevision)
	if err != nil {
		return store.ChannelThread{}, store.AgentConfig{}, nil, err
	}
	read, rule, err := b.read(connection, definition, *message)
	if err != nil {
		return store.ChannelThread{}, store.AgentConfig{}, nil, err
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
			return store.ChannelThread{}, store.AgentConfig{}, nil, err
		}
		if !linked {
			// waiting is a reply kept for a message that links its thread to take (wait).
			b.logger.Info("dropped an inbound message that is not addressed to the connection on a thread nobody linked",
				"connector", message.ConnectorID, "provider_app", app.ProviderAppID, "customer", app.CustomerID,
				"connection", connection.ID, "config", configs[0].ID, "waiting", !startsThread(read))
			return store.ChannelThread{}, store.AgentConfig{}, nil, nil
		}
	}
	message.Text = rule.WithoutMention(read.Text, connection.Metadata)
	message.Media = read.Media
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
		return store.ChannelThread{}, store.AgentConfig{}, nil, err
	}
	// Every reply in a thread comes after the message that started it, so the replies that
	// waited for that message are all to be answered. A link made by a reply, such as a
	// mention in a thread of people, leaves them waiting until they are dropped: some came
	// before the bot was spoken to, which it does not read (AI-989). The message takes them
	// whether or not it made the link now and whether or not it was taken before, and each
	// waiting reply is handed out once (store.AcceptChannelInbound).
	if startsThread(read) {
		waiting = func(reply core.InboundMessage) (core.InboundMessage, error) {
			read, rule, err := b.read(connection, definition, reply)
			reply.Text, reply.Media = rule.WithoutMention(read.Text, connection.Metadata), read.Media
			return reply, err
		}
	}
	return thread, configs[0], waiting, nil
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
// started is when an episode the message opens starts; zero is now.
func (b *Bridge) episode(ctx context.Context, thread store.ChannelThread, config store.AgentConfig, message core.InboundMessage, started time.Time) (omnichannel.Opened, error) {
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
		StartedAt:     started,
		StreamAppPK:   thread.StreamAppPK,
	})
}

// read reads a message again from the raw body with the manifest of the connection's
// definition, by the provider's id for it (core.ChannelMessage), with the rule that read it:
// the verifier hands over the thread key alone, and a reply names the key's parts; whether the
// message addresses the connection is that revision's to say.
func (b *Bridge) read(connection store.ConnectorConnection, definition store.ConnectorDefinition, message core.InboundMessage) (core.ChannelMessage, core.MessageRule, error) {
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

// writeInto writes message into its thread channel as its author under the Stream message id
// id, creating the channel the first time, with the agent config that answers it.
func (b *Bridge) writeInto(ctx context.Context, thread store.ChannelThread, config store.AgentConfig, message core.InboundMessage, id string) (Written, error) {
	bound, err := b.stream.ForApp(ctx, thread.CustomerID, thread.StreamAppPK)
	if err != nil {
		return Written{}, err
	}
	author := AuthorUserID(thread.CustomerID, thread.ConnectorID, thread.ProviderUnitID, message.AuthorID)
	name, agentName := message.AuthorID, config.Name
	// The agent of a thread channel writes as a user of the channel's own id: the
	// conversation held on the channel writes as support_agent_id. The message hook finds
	// the session answering in the channel by its conversation (Manager.ByConversationWhere).
	if err := conversation.CreateMissingUsers(ctx, bound.Client, map[string]getstream.UserRequest{
		author:           {ID: author, Name: &name},
		thread.ChannelID: {ID: thread.ChannelID, Name: &agentName},
	}); err != nil {
		return Written{}, stack.Wrap(err)
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
		return Written{}, stack.Wrap(err)
	}
	// No source: the conversation reads a message without one as the person's
	// (conversation.messageFromThread). Bridged: the message hook leaves it to the worker.
	request := getstream.MessageRequest{ID: &id, Text: &message.Text, UserID: &author, Attachments: photos(message.Media),
		Custom: map[string]any{BridgedField: true}}
	sent, err := bound.Client.Chat().SendMessage(ctx, chatlog.ChannelType, thread.ChannelID, &getstream.SendMessageRequest{Message: request})
	if err != nil {
		return Written{}, stack.Wrap(err)
	}
	stored := sent.Data.Message
	return Written{ID: stored.ID, Text: stored.Text, Author: stored.User.ID, Attachments: stored.Attachments}, nil
}

// rewritten is the message an attempt whose write Stream Chat refused as already done wrote
// (alreadyWritten), rebuilt from its row as writeInto wrote it: that refusal carries no
// message.
func rewritten(thread store.ChannelThread, message core.InboundMessage, id string) Written {
	return Written{ID: id, Text: message.Text, Author: AuthorUserID(thread.CustomerID, thread.ConnectorID, thread.ProviderUnitID, message.AuthorID),
		Attachments: photos(message.Media)}
}

// photos are a message's image files as Stream Chat image attachments, the shape the message
// hook hands the model (api.messageEvent.photos), as conversation.chatFileAttachments writes
// one. Any other file, such as a video, is left out: the agent gets the text alone (AI-2094).
func photos(media []core.Media) []getstream.Attachment {
	var attachments []getstream.Attachment
	for _, file := range media {
		if strings.HasPrefix(file.MIMEType, "image/") {
			attachments = append(attachments, getstream.Attachment{
				Type: getstream.PtrTo("image"), ImageUrl: &file.URL, Custom: map[string]any{"mime_type": file.MIMEType},
			})
		}
	}
	return attachments
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

// AuthorUserID is the Stream Chat user an external author writes as in a thread channel: one
// for each customer, connector, provider unit and author id. It is a digest so that any
// provider's ids fit the characters every Stream id here keeps to ([A-Za-z0-9_-]), and so
// that one person in two customers' workspaces is two users, never one that two customers
// write as. Who the person is across channels is the contact map's (T43), not this.
func AuthorUserID(customerID, connectorID, providerUnitID, authorID string) string {
	sum := sha256.Sum256([]byte(customerID + "\x00" + connectorID + "\x00" + providerUnitID + "\x00" + authorID))
	return connectorID + "-" + hex.EncodeToString(sum[:16])
}
