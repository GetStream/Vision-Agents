package channels

import (
	"context"
	"crypto/rand"
	"encoding/json"
	"errors"
	"fmt"
	"log/slog"
	"math/big"
	"net/http"
	"net/url"
	"strings"
	"sync"
	"time"

	"github.com/GetStream/Vision-Agents/acceleration/internal/agent"
	"github.com/GetStream/Vision-Agents/acceleration/internal/auth"
	"github.com/GetStream/Vision-Agents/acceleration/internal/dlc"
	"github.com/GetStream/Vision-Agents/acceleration/internal/options"
	"github.com/GetStream/Vision-Agents/acceleration/internal/routing"
	"github.com/GetStream/Vision-Agents/acceleration/internal/sandbox"
	"github.com/GetStream/Vision-Agents/acceleration/internal/session"
	"github.com/GetStream/Vision-Agents/acceleration/internal/stack"
	"github.com/GetStream/Vision-Agents/acceleration/internal/store"
)

// runTimeout bounds the turn one message earns.
const runTimeout = 10 * time.Minute

// settleGap is how long the agent stays quiet, with nothing left to do, before a message is
// taken as answered.
const settleGap = 2 * time.Second

// LinkValidFor is how long a code to claim a number is good for.
const LinkValidFor = 15 * time.Minute

// Options configures a Service. Store, Sessions and Secrets are required.
type Options struct {
	Store    *store.Store
	Sessions *session.Manager
	// Secrets opens the credentials a connected line was stored with.
	Secrets *auth.Sealer
	// Transport reaches the providers. Nil is http.DefaultClient.
	Transport *http.Client
	// Gate keeps the agent from writing to somebody who opted out, or past the sandbox.
	Gate   *dlc.Gate
	Logger *slog.Logger
}

// Service answers what arrives on an app's channels.
type Service struct {
	store     *store.Store
	sessions  *session.Manager
	secrets   *auth.Sealer
	transport *http.Client
	gate      *dlc.Gate
	logger    *slog.Logger

	// mu guards answering, keyed by who is writing: two messages from one person are
	// answered one after the other, since both carry on the same conversation.
	mu      sync.Mutex
	turns   map[string]*holder
	ctx     context.Context
	cancel  context.CancelFunc
	working sync.WaitGroup
}

// Answer is what a delivery to a webhook is answered with, which is not what the agent says:
// that goes back over the channel, not in this response.
type Answer struct {
	Status int
	Body   any
	// Text is answered instead of Body when it is set, which the challenge a provider
	// checks a webhook with needs: it is echoed back as itself, not as JSON.
	Text string
}

// New validates the options and returns a Service.
func New(options Options) (*Service, error) {
	if options.Store == nil {
		return nil, errors.New("channels: a database is required")
	}
	if options.Sessions == nil {
		return nil, errors.New("channels: a session manager is required")
	}
	if options.Secrets == nil {
		return nil, errors.New("channels: a key encryption key is required to open a line's credentials")
	}
	if options.Logger == nil {
		options.Logger = slog.Default()
	}
	ctx, cancel := context.WithCancel(context.Background())
	return &Service{
		store:     options.Store,
		sessions:  options.Sessions,
		secrets:   options.Secrets,
		transport: options.Transport,
		gate:      options.Gate,
		logger:    options.Logger,
		turns:     map[string]*holder{},
		ctx:       ctx,
		cancel:    cancel,
	}, nil
}

// Close stops answering and waits for the turns already running to finish.
func (s *Service) Close() {
	if s == nil {
		return
	}
	s.cancel()
	s.working.Wait()
}

// Challenge answers the GET a provider checks a webhook with before it delivers to it.
// WhatsApp is the one that does this: it asks for the verify token back along with a
// challenge to echo, so an URL cannot be pointed at somebody else's router.
func (s *Service) Challenge(ctx context.Context, token string, query url.Values) Answer {
	line, _, err := s.line(ctx, token)
	if err != nil {
		return Answer{Status: http.StatusGone, Body: failure(err.Error())}
	}
	if query.Get("hub.mode") != "subscribe" || query.Get("hub.verify_token") != line.Challenge {
		return Answer{Status: http.StatusForbidden, Body: failure("that is not this line's verify token")}
	}
	return Answer{Status: http.StatusOK, Text: query.Get("hub.challenge")}
}

// Receive answers one delivery: every message in it that has not been seen before earns a
// turn, in the background, since the provider is waiting on this request and a model takes
// seconds.
func (s *Service) Receive(ctx context.Context, token string, header http.Header, body []byte) Answer {
	line, account, err := s.line(ctx, token)
	if err != nil {
		return Answer{Status: http.StatusGone, Body: failure(err.Error())}
	}
	provider, ok := For(line.Kind, s.transport)
	if !ok {
		return Answer{Status: http.StatusGone, Body: failure("no channel called " + string(line.Kind))}
	}
	if err := provider.Verify(line, header, body, time.Now()); err != nil {
		return Answer{Status: http.StatusUnauthorized, Body: failure(err.Error())}
	}

	delivered, err := provider.Parse(body)
	if err != nil {
		return Answer{Status: http.StatusBadRequest, Body: failure(err.Error())}
	}
	for _, message := range delivered {
		fresh, err := s.store.ClaimChannelMessage(ctx, account.ID, message.ID)
		if err != nil {
			return Answer{Status: http.StatusInternalServerError, Body: failure(err.Error())}
		}
		// A provider that did not hear back in time sends the same message again. Answering
		// it twice is the agent talking over itself, so a message already taken is dropped.
		if !fresh {
			continue
		}
		s.working.Add(1)
		go func(message Message) {
			defer s.working.Done()
			s.answer(account, line, provider, message)
		}(message)
	}
	return Answer{Status: http.StatusAccepted, Body: map[string]string{}}
}

// Link mints a code somebody signed in may text from their phone to claim it. It is how an
// agent that keeps something personal learns which end user a number belongs to.
func (s *Service) Link(ctx context.Context, customerID, configID, userID string) (store.ChannelLink, error) {
	code, err := linkCode()
	if err != nil {
		return store.ChannelLink{}, err
	}
	link := store.ChannelLink{
		Code:       code,
		CustomerID: customerID,
		ConfigID:   configID,
		UserID:     userID,
		ExpiresAt:  time.Now().UTC().Add(LinkValidFor),
	}
	if err := s.store.SaveChannelLink(ctx, &link); err != nil {
		return store.ChannelLink{}, err
	}
	return link, nil
}

// Open reads back the credentials a line was stored with. The customer is bound into the
// sealing, so ciphertext moved to another app's row does not open.
func Open(secrets *auth.Sealer, stored store.ChannelAccount) (Account, error) {
	account := Account{
		Kind:      Kind(stored.Kind),
		E164:      stored.E164,
		AccountID: stored.AccountID,
	}
	if secrets == nil || len(stored.SecretsSealed) == 0 {
		return account, stack.Wrap(errors.New("channels: this line has no stored credentials"))
	}
	raw, err := secrets.OpenWithAADVersion(stored.SecretsSealed,
		[]byte(stored.CustomerID), stored.SecretsKEKVersion)
	if err != nil {
		return account, err
	}
	var opened Secrets
	if err := json.Unmarshal([]byte(raw), &opened); err != nil {
		return account, stack.Wrap(err)
	}
	account.Token = opened.Token
	account.Signing = opened.Signing
	account.Challenge = opened.Challenge
	return account, nil
}

// line opens the credentials of the account a delivery is addressed to.
func (s *Service) line(ctx context.Context, token string) (Account, store.ChannelAccount, error) {
	account, err := s.store.ChannelAccountByToken(ctx, token)
	if err != nil {
		return Account{}, store.ChannelAccount{}, errors.New("no line is connected at this address")
	}
	line, err := Open(s.secrets, account)
	if err != nil {
		return Account{}, account, errors.New("this line's credentials cannot be opened: connect it again")
	}
	return line, account, nil
}

// answer has the agent on a line take one message and writes back what it says.
func (s *Service) answer(account store.ChannelAccount, line Account, provider Provider, message Message) {
	ctx, cancel := context.WithTimeout(s.ctx, runTimeout)
	defer cancel()

	if s.keyword(ctx, account, line, provider, message) {
		return
	}
	config, found, err := s.store.ConfigOnChannel(ctx, account.CustomerID, account.Kind, account.E164)
	if err != nil {
		s.logger.Error("could not find the agent on a channel", "channel", account.Kind,
			"number", account.E164, "error", err)
		return
	}
	if !found {
		s.logger.Warn("a message arrived on a line no agent answers on", "channel", account.Kind,
			"number", account.E164, "hint", "name it under channels in the agent's config")
		return
	}

	unlock := s.hold(account.ID + "\x00" + message.From)
	defer unlock()

	if err := s.gate.Allow(ctx, account.CustomerID, account.Kind, message.From); err != nil {
		s.logger.Info("not answering a channel message", "channel", account.Kind, "reason", err)
		return
	}

	identity, ok := s.identify(ctx, config, account, line, provider, message)
	if !ok {
		return
	}

	spec := session.FromConfig(config)
	spec.Text = true
	spec.CallID = ""
	spec.STSTarget = ""
	spec.Greeting = ""
	spec.PersistConversation = true
	spec.ConversationID = identity.ConversationID
	spec.Caller = routing.Caller{UserID: identity.UserID}
	spec.CallerKind = auth.KindAuthenticated
	// The conversation is kept in Stream Chat like any other, and this is what says it was
	// held over a channel, so a dashboard or an inbox can tell it from one typed in an app.
	if spec.Custom == nil {
		spec.Custom = map[string]any{}
	}
	spec.Custom["channel"] = account.Kind
	spec.Custom["channel_number"] = account.E164
	spec.Custom["channel_from"] = message.From

	created, err := s.sessions.Create(ctx, spec)
	if err != nil {
		s.logger.Error("could not open a conversation for a channel message", "channel", account.Kind,
			"agent", config.Name, "error", err)
		return
	}
	// The session ends with the message it answered, which is what saves the conversation:
	// the next message reopens it by its id rather than starting over.
	defer func() {
		if _, err := s.sessions.Close(created.ID(), session.OwnerOf(created.Spec())); err != nil {
			s.logger.Debug("could not close a channel conversation", "session", created.ID(), "error", err)
		}
	}()
	events, detach := created.Watch()
	defer detach()

	// The conversation a session opened is the one the next message carries on, so it is
	// recorded before the turn rather than after: a turn that fails halfway still leaves the
	// person's history where they can see it.
	if identity.ConversationID == "" {
		identity.ConversationID = created.Spec().ConversationID
		if err := s.store.SaveChannelIdentity(ctx, &identity); err != nil {
			s.logger.Error("could not record a channel conversation", "agent", config.Name, "error", err)
		}
	}

	sending := make(chan struct{})
	go func() {
		defer close(sending)
		s.send(ctx, created, account.CustomerID, line, provider, message, events)
	}()
	// The provider's own message id is the command id, so the durable conversation refuses a
	// repeat for the same reason the delivery table does.
	if _, _, err := created.RespondCommand(ctx, message.ID, message.Text, "", options.LLM{}); err != nil {
		s.logger.Error("the agent could not take a channel message", "session", created.ID(), "error", err)
	}
	<-sending
}

// identify is who the number writing belongs to, and false when nobody is to be answered.
//
// Under phone, which is the default, each number is an end user of its own: anybody who
// writes is answered and what they say is theirs alone. Under link, a number means nothing
// until somebody signed in has tied it to themselves with a code, which is what an agent
// that reads a person's own calendar or orders needs before it says a word.
func (s *Service) identify(ctx context.Context, config store.AgentConfig, account store.ChannelAccount,
	line Account, provider Provider, message Message) (store.ChannelIdentity, bool) {
	identity, err := s.store.ChannelIdentity(ctx, account.CustomerID, config.ID, account.Kind, message.From)
	if err == nil {
		return identity, true
	}
	if !errors.Is(err, store.ErrUnknownChannelIdentity) {
		s.logger.Error("could not tell who is writing", "channel", account.Kind, "error", err)
		return store.ChannelIdentity{}, false
	}

	identity = store.ChannelIdentity{
		CustomerID: account.CustomerID,
		ConfigID:   config.ID,
		Kind:       account.Kind,
		Address:    message.From,
		UserID:     "phone:" + message.From,
	}
	if config.Channels.Identity == store.ChannelIdentityLink {
		link, err := s.store.ClaimChannelLink(ctx, account.CustomerID, strings.TrimSpace(message.Text))
		if err != nil {
			s.say(ctx, line, provider, message, Reply{Text: "Send me the code from your account to " +
				"get started. It ties this number to you, so I can answer about what is yours."})
			return store.ChannelIdentity{}, false
		}
		if link.ConfigID != config.ID {
			s.say(ctx, line, provider, message, Reply{Text: "That code is for another agent."})
			return store.ChannelIdentity{}, false
		}
		identity.UserID = link.UserID
	}
	if err := s.store.SaveChannelIdentity(ctx, &identity); err != nil {
		s.logger.Error("could not record who is writing", "channel", account.Kind, "error", err)
		return store.ChannelIdentity{}, false
	}
	if config.Channels.Identity == store.ChannelIdentityLink {
		s.say(ctx, line, provider, message, Reply{Text: "This number is yours now. What can I do for you?"})
		return identity, false
	}
	return identity, true
}

// send writes back what the agent says, as it says it, and returns once the turn has
// settled: one message can earn several replies, and work it hands off produces files.
func (s *Service) send(ctx context.Context, created *session.Session, customerID string, line Account,
	provider Provider, message Message, events <-chan session.Event) {
	ticker := time.NewTicker(250 * time.Millisecond)
	defer ticker.Stop()
	last := time.Now()
	for {
		var reply Reply
		select {
		case <-ctx.Done():
			return
		case event, open := <-events:
			if !open {
				return
			}
			last = time.Now()
			reply = replyOf(event)
		case <-ticker.C:
			if time.Since(last) >= settleGap && !created.Busy() {
				return
			}
			continue
		}
		if reply.Empty() {
			continue
		}
		// A sandboxed app can reach its daily limit halfway through a turn.
		if err := s.gate.Allow(ctx, customerID, string(line.Kind), message.From); err != nil {
			s.logger.Info("not writing back on a channel", "channel", line.Kind, "reason", err)
			continue
		}
		s.say(ctx, line, provider, message, reply)
		s.gate.Sent(ctx, customerID)
	}
}

// replyOf is what an event is worth writing back, if anything.
func replyOf(event session.Event) Reply {
	switch happened := event.(type) {
	case agent.Responded:
		return Reply{Text: happened.Text}
	case agent.TaskSettled:
		return Reply{Text: happened.Text, Files: filesOf(happened.Files)}
	}
	return Reply{}
}

func filesOf(attached []sandbox.Attachment) []File {
	files := make([]File, 0, len(attached))
	for _, one := range attached {
		if one.URL == "" {
			continue
		}
		files = append(files, File{Name: one.Name, MimeType: one.MIME, URL: one.URL})
	}
	return files
}

func (s *Service) say(ctx context.Context, line Account, provider Provider, message Message, reply Reply) {
	if err := provider.Send(ctx, line, message.Thread, reply); err != nil {
		s.logger.Error("could not write back on a channel", "channel", line.Kind,
			"number", line.E164, "error", err)
	}
}

// hold serializes the turns for one person, so their second message carries on the
// conversation their first one opened rather than racing it.
func (s *Service) hold(key string) func() {
	s.mu.Lock()
	held, waiting := s.turns[key]
	if !waiting {
		held = &holder{}
		s.turns[key] = held
	}
	held.waiting++
	s.mu.Unlock()

	held.lock.Lock()
	return func() {
		held.lock.Unlock()
		s.mu.Lock()
		held.waiting--
		if held.waiting == 0 {
			delete(s.turns, key)
		}
		s.mu.Unlock()
	}
}

// holder is one person's turn, counted so the last one out forgets them again.
type holder struct {
	lock    sync.Mutex
	waiting int
}

// linkCode is six digits, which is short enough to read off a screen and type into a text.
func linkCode() (string, error) {
	digits := make([]byte, 6)
	for i := range digits {
		drawn, err := rand.Int(rand.Reader, big.NewInt(10))
		if err != nil {
			return "", stack.Wrap(fmt.Errorf("channels: minting a code: %w", err))
		}
		digits[i] = byte('0' + drawn.Int64())
	}
	return string(digits), nil
}

func failure(message string) map[string]string {
	return map[string]string{"error": message}
}
