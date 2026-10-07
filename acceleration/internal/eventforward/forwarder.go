// Package eventforward forwards a connector's raw provider events to the customer's own URLs,
// its event destinations (T46, AI-875; channels.md on connectors/planning, «Integration modes:
// full platform, customizations, pass-through», modes B and C).
//
// The events route verifies a provider delivery with the provider app's secret and acts on it,
// then hands it to Forward, which queues it in Postgres for each destination that takes it and
// returns: the provider's ack never waits on a customer's URL. A worker on every router takes
// the queued forwards and sends each one with the provider's body and verification headers as
// they came, signed on top in the Standard Webhooks shape with the destination's own secret,
// never with a deployment secret (https://www.standardwebhooks.com/).
package eventforward

import (
	"bytes"
	"context"
	"crypto/sha256"
	"encoding/hex"
	"errors"
	"fmt"
	"io"
	"log/slog"
	"net/http"
	"strconv"
	"strings"
	"sync"
	"time"

	"github.com/GetStream/Vision-Agents/acceleration/internal/auth"
	"github.com/GetStream/Vision-Agents/acceleration/internal/connectors/core"
	"github.com/GetStream/Vision-Agents/acceleration/internal/egress"
	"github.com/GetStream/Vision-Agents/acceleration/internal/plugins"
	"github.com/GetStream/Vision-Agents/acceleration/internal/stack"
	"github.com/GetStream/Vision-Agents/acceleration/internal/store"
)

// The Standard Webhooks spec, opened October 6, 2026:
// https://github.com/standard-webhooks/standard-webhooks/blob/main/spec/standard-webhooks.md
// («Webhook headers», «Deliverability and reliability», «Request timeouts»).
const (
	headerID        = "webhook-id"
	headerTimestamp = "webhook-timestamp"
	headerSignature = "webhook-signature"
)

// attemptTimeout bounds one send: the spec's «A recommended request timeout value for webhooks
// is somewhere between 15 and 30s», its low end, so one slow customer holds a worker slot the
// least the spec allows.
const attemptTimeout = 15 * time.Second

// lease is how far a worker pushes a delivery's next attempt when it takes one, so no other
// router takes it while it is sent: longer than one attempt's attemptTimeout, so a live worker
// never loses one, and short enough that one a stopped router held is sent again within a
// minute. A choice, not a spec value.
const lease = time.Minute

// defaultPoll is how often a worker looks for deliveries due, besides when Forward queues one,
// which it sends at once. The shortest wait in defaultRetries is 5 s, so a retry goes out at
// most a second late. A choice, not a spec value.
const defaultPoll = time.Second

// inFlight is how many sends one router runs at once: one slow destination, at attemptTimeout,
// takes one slot and leaves the rest to the others. A choice, not a spec value. # unverified
// against real load
const inFlight = 16

// maxAnswerBytes is how much of a destination's answer is read: none of it is used, so it is
// read only to let the connection be reused; 64 KiB is a choice.
const maxAnswerBytes = 64 << 10

// defaultRetries are the waits before each attempt after the first: the spec's example retry
// schedule («Example retry schedule»: immediately, 5 seconds, 5 minutes, 30 minutes, 2 hours),
// its first five rows. After the fifth attempt fails, at 2 h 35 min 5 s, the forward is given
// up and logged. The spec's later rows, 5 h to 24 h, go with a delivery log the customer can
// read and with telling the customer by other means, which this router does not have yet; a
// raw interactive event, such as a button click, is moot long before then. A choice.
var defaultRetries = []time.Duration{5 * time.Second, 5 * time.Minute, 30 * time.Minute, 2 * time.Hour}

// rotationOverlap is how long the secret a rotation replaced still signs beside the new one,
// so a receiver can move to the new one without a delivery failing its check («the webhook is
// signed both using the current key, and using an old key (for a set period of time)»). The
// spec names no period; a day is a choice, and covers the last retry of defaultRetries.
const rotationOverlap = 24 * time.Hour

// errNotPublic is a destination URL egress refuses: not https, or a host that is or resolves
// to a private, loopback or otherwise not public address.
var errNotPublic = errors.New("eventforward: the destination is not a public https URL")

// Options configures a Forwarder. Store and Secrets are required.
type Options struct {
	Store *store.Store
	// Secrets seals and opens the destinations' signing secrets.
	Secrets *auth.Sealer
	// HTTP sends every forward. Nil is egress.NewClient, which dials public addresses alone.
	// Tests pass a client that reaches their local destinations. Redirects are never followed,
	// whatever client it is.
	HTTP *http.Client
	// PublicURL checks a destination URL before it is stored. Nil is
	// egress.ValidatePublicHTTPSURL.
	PublicURL func(ctx context.Context, raw string) error
	// Retries are the waits before each attempt after the first. Nil is defaultRetries.
	Retries []time.Duration
	// Poll is how often due deliveries are looked for. Zero is defaultPoll.
	Poll   time.Duration
	Logger *slog.Logger
}

// Forwarder queues provider deliveries for the customer's event destinations and sends them.
type Forwarder struct {
	store     *store.Store
	secrets   *auth.Sealer
	http      *http.Client
	publicURL func(ctx context.Context, raw string) error
	retries   []time.Duration
	poll      time.Duration
	logger    *slog.Logger

	// kick wakes the worker: a delivery was queued, or a slot freed.
	kick chan struct{}
	// slots holds one token for each send running.
	slots   chan struct{}
	ctx     context.Context
	cancel  context.CancelFunc
	working sync.WaitGroup
}

// Event is one verified provider delivery of one customer's provider app.
type Event struct {
	CustomerID  string
	ConnectorID string
	// Headers are the provider's headers a receiver verifies the body with (ProviderHeaders).
	Headers map[string]string
	// Body is the raw request body, as verified.
	Body []byte
	// Handled is whether the router acted on the delivery: a signal, or a message an agent of
	// the customer answers. A destination that forwards unhandled takes it only when false.
	Handled bool
}

// Secret is a destination's signing secret, made for it and sealed bound to it.
type Secret struct {
	// Plain is the whsec_ secret, shown to the customer once.
	Plain   string
	Sealed  []byte
	Version int
}

// New validates the options and returns a Forwarder. It sends nothing until Start.
func New(options Options) (*Forwarder, error) {
	if options.Store == nil || options.Secrets == nil {
		return nil, stack.Wrap(errors.New("eventforward: a store and a sealer are required"))
	}
	client := options.HTTP
	if client == nil {
		client = egress.NewClient(attemptTimeout, nil)
	}
	// A copy, so the caller's client keeps its own policy. The spec: «3xx: Failure. Following
	// redirects causes unnecessary load on both the sender and the receiver».
	unredirected := *client
	unredirected.CheckRedirect = func(*http.Request, []*http.Request) error { return http.ErrUseLastResponse }
	publicURL := options.PublicURL
	if publicURL == nil {
		publicURL = egress.ValidatePublicHTTPSURL
	}
	retries := options.Retries
	if retries == nil {
		retries = defaultRetries
	}
	poll := options.Poll
	if poll <= 0 {
		poll = defaultPoll
	}
	logger := options.Logger
	if logger == nil {
		logger = slog.Default()
	}
	ctx, cancel := context.WithCancel(context.Background())
	return &Forwarder{
		store:     options.Store,
		secrets:   options.Secrets,
		http:      &unredirected,
		publicURL: publicURL,
		retries:   retries,
		poll:      poll,
		logger:    logger,
		kick:      make(chan struct{}, 1),
		slots:     make(chan struct{}, inFlight),
		ctx:       ctx,
		cancel:    cancel,
	}, nil
}

// Start runs the worker until Close.
func (f *Forwarder) Start() {
	f.working.Add(1)
	go func() {
		defer f.working.Done()
		ticker := time.NewTicker(f.poll)
		defer ticker.Stop()
		for {
			f.sendDue()
			select {
			case <-f.ctx.Done():
				return
			case <-ticker.C:
			case <-f.kick:
			}
		}
	}()
}

// Close stops the worker and cancels the sends in flight. A send cut short is not counted as
// an attempt: its lease runs out and a router sends it again.
func (f *Forwarder) Close() {
	f.cancel()
	f.working.Wait()
}

// Forward queues one delivery for each of the customer's destinations of the connector that
// takes it, and wakes the worker. It touches Postgres alone and sends nothing, so it is safe on
// the request path of a provider that waits a few seconds at most (Slack: «respond ... within
// three seconds», https://docs.slack.dev/apis/events-api/). An error is the store failing.
func (f *Forwarder) Forward(ctx context.Context, event Event) error {
	if event.CustomerID == "" {
		return nil
	}
	queued, err := f.store.QueueEventDeliveries(ctx, event.CustomerID, event.ConnectorID, event.Handled, store.EventDelivery{
		ID: deliveryID(event.Body), Headers: event.Headers, Body: event.Body, NextAttemptAt: time.Now(),
	})
	if err != nil {
		return err
	}
	if queued > 0 {
		f.wake()
	}
	return nil
}

// CheckURL is errNotPublic for a destination URL a forward could not be sent to.
func (f *Forwarder) CheckURL(ctx context.Context, raw string) error {
	if err := f.publicURL(ctx, raw); err != nil {
		return stack.Wrap(fmt.Errorf("%w: %w", errNotPublic, err))
	}
	return nil
}

// NewSecret is a fresh signing secret for one destination, sealed with the destination's
// customer, connector and id as AAD, so a sealed secret copied onto another row does not open.
func (f *Forwarder) NewSecret(customerID, connectorID, id string) (Secret, error) {
	// 32 random bytes, inside the spec's «Between 24 bytes (192 bits) and 64 bytes (512 bits)»,
	// base64 after whsec_, the form MCP Events subscriptions already use.
	plain, err := plugins.NewWebhookSecret()
	if err != nil {
		return Secret{}, stack.Wrap(err)
	}
	sealed, err := f.secrets.SealWithAAD(plain, secretAAD(customerID, connectorID, id))
	if err != nil {
		return Secret{}, stack.Wrap(err)
	}
	return Secret{Plain: plain, Sealed: sealed, Version: f.secrets.CurrentVersion()}, nil
}

// RotationOverlap is how long a secret a rotation replaced still signs, from now.
func RotationOverlap(now time.Time) time.Time {
	return now.Add(rotationOverlap)
}

// deliveryID is the webhook-id of a delivery: msg_ and the first 128 bits of the body's
// SHA-256, so the same provider body is the same id, on every attempt and when the provider
// delivers it again. The spec: the id «remains the same no matter how many times a webhook
// that has failed is retried», and it must hold no «.», which hex never does.
func deliveryID(body []byte) string {
	sum := sha256.Sum256(body)
	return "msg_" + hex.EncodeToString(sum[:16])
}

// ProviderHeaders are the request headers a destination is sent beside the raw body, as they
// came: Content-Type, and the signature and timestamp headers the connector's channel.verifier
// names (Slack: X-Slack-Signature and X-Slack-Request-Timestamp), so a receiver that holds the
// app's own signing secret, such as Slack Bolt, verifies the body itself. Nothing else is
// copied: a proxy's or a load balancer's headers say where the router runs, and a provider's
// retry headers (X-Slack-Retry-Num) are about the router's answer, not the customer's;
// webhook-id is what tells the receiver a delivery again.
func ProviderHeaders(m core.Manifest, header http.Header) map[string]string {
	names := []string{"Content-Type"}
	if m.Channel != nil {
		names = append(names, m.Channel.Verifier.Header, m.Channel.Verifier.TimestampHeader)
	}
	kept := map[string]string{}
	for _, name := range names {
		if value := header.Get(name); name != "" && value != "" {
			kept[http.CanonicalHeaderKey(name)] = value
		}
	}
	return kept
}

// sign is the webhook-signature header of one send: v1, and the base64 HMAC-SHA256 of
// id.timestamp.body under each secret, space-separated, so a receiver holding either secret of
// a rotation verifies it («Webhook headers»: «The signature header is a space delimited list of
// signatures»).
func sign(id string, at time.Time, body []byte, secrets ...string) (string, error) {
	signatures := make([]string, 0, len(secrets))
	for _, secret := range secrets {
		signature, err := plugins.SignWebhook(secret, id, at, body)
		if err != nil {
			return "", stack.Wrap(err)
		}
		signatures = append(signatures, signature)
	}
	return strings.Join(signatures, " "), nil
}

// wake tells the worker there may be something to send, without waiting.
func (f *Forwarder) wake() {
	select {
	case f.kick <- struct{}{}:
	default:
	}
}

// sendDue takes as many due deliveries as there are free slots and sends each on its own
// goroutine, until none is due or no slot is free.
func (f *Forwarder) sendDue() {
	for f.ctx.Err() == nil {
		free := cap(f.slots) - len(f.slots)
		if free == 0 {
			return
		}
		now := time.Now()
		claimed, err := f.store.ClaimEventDeliveries(f.ctx, now, free, now.Add(lease))
		if err != nil {
			if f.ctx.Err() == nil {
				f.logger.Error("could not take the event forwards due", "error", err)
			}
			return
		}
		for _, delivery := range claimed {
			f.slots <- struct{}{}
			f.working.Add(1)
			go func() {
				defer f.working.Done()
				defer func() {
					<-f.slots
					f.wake()
				}()
				f.attempt(delivery)
			}()
		}
		if len(claimed) < free {
			return
		}
	}
}

// attempt sends a delivery once and records what came of it: gone when the destination took
// it or refused it for good, else due again after the next wait, or gone when the waits ran
// out.
func (f *Forwarder) attempt(delivery store.ClaimedEventDelivery) {
	ctx, cancel := context.WithTimeout(f.ctx, attemptTimeout)
	err := f.send(ctx, delivery)
	cancel()
	if f.ctx.Err() != nil {
		// Close cut it short: the lease runs out and a router sends it again.
		return
	}
	writing, done := context.WithTimeout(context.WithoutCancel(f.ctx), attemptTimeout)
	defer done()
	log := f.logger.With("connector", delivery.ConnectorID, "customer", delivery.CustomerID,
		"destination", delivery.DestinationID, "webhook_id", delivery.ID)
	var again retryable
	attempts := delivery.Attempts + 1
	switch {
	case err == nil:
		err = f.store.FinishEventDelivery(writing, delivery.DestinationID, delivery.ID)
	case errors.As(err, &again) && attempts <= len(f.retries):
		log.Info("an event forward failed and is sent again", "attempts", attempts, "error", err)
		err = f.store.RetryEventDelivery(writing, delivery.DestinationID, delivery.ID, attempts, time.Now().Add(f.retries[attempts-1]))
	default:
		log.Warn("gave up an event forward", "attempts", attempts, "error", err)
		err = f.store.FinishEventDelivery(writing, delivery.DestinationID, delivery.ID)
	}
	if err != nil {
		log.Error("could not record an event forward's attempt", "error", err)
	}
}

// send posts the raw body with the provider's headers and the Standard Webhooks ones. A 2xx is
// sent. No answer, a 5xx or a 429 is retryable: a 429 is «Too Many Requests», which the spec
// says to throttle, not drop (RFC 6585 section 4). Any other answer, a 3xx or a 4xx, is the
// destination refusing it for good, and is not retried. The spec counts a 4xx as a failure to
// retry; a destination that answers 400 to a body it cannot read answers the same body the same
// way every time, so this router does not send it again.
func (f *Forwarder) send(ctx context.Context, delivery store.ClaimedEventDelivery) error {
	secrets, err := f.openSecrets(delivery)
	if err != nil {
		return err
	}
	at := time.Now()
	signature, err := sign(delivery.ID, at, delivery.Body, secrets...)
	if err != nil {
		return err
	}
	request, err := http.NewRequestWithContext(ctx, http.MethodPost, delivery.URL, bytes.NewReader(delivery.Body))
	if err != nil {
		return stack.Wrap(err)
	}
	for name, value := range delivery.Headers {
		request.Header.Set(name, value)
	}
	request.Header.Set(headerID, delivery.ID)
	request.Header.Set(headerTimestamp, strconv.FormatInt(at.Unix(), 10))
	request.Header.Set(headerSignature, signature)
	response, err := f.http.Do(request)
	if err != nil {
		return retryable{stack.Wrap(err)}
	}
	defer response.Body.Close() //nolint:errcheck // the answer is drained and dropped
	_, _ = io.Copy(io.Discard, io.LimitReader(response.Body, maxAnswerBytes))
	switch {
	case response.StatusCode >= 200 && response.StatusCode <= 299:
		return nil
	case response.StatusCode >= http.StatusInternalServerError || response.StatusCode == http.StatusTooManyRequests:
		return retryable{stack.Wrap(fmt.Errorf("eventforward: the destination answered %d", response.StatusCode))}
	default:
		return stack.Wrap(fmt.Errorf("eventforward: the destination refused the forward with %d", response.StatusCode))
	}
}

// openSecrets are the secrets a send is signed with: the destination's, and the one its last
// rotation replaced while that still signs.
func (f *Forwarder) openSecrets(delivery store.ClaimedEventDelivery) ([]string, error) {
	aad := secretAAD(delivery.CustomerID, delivery.ConnectorID, delivery.DestinationID)
	current, err := f.secrets.OpenWithAADVersion(delivery.SecretSealed, aad, delivery.KEKVersion)
	if err != nil {
		return nil, stack.Wrap(fmt.Errorf("eventforward: the signing secret of destination %s does not open: %w", delivery.DestinationID, err))
	}
	secrets := []string{current}
	if len(delivery.PreviousSecretSealed) > 0 && delivery.PreviousUntil != nil && time.Now().Before(*delivery.PreviousUntil) {
		previous, err := f.secrets.OpenWithAADVersion(delivery.PreviousSecretSealed, aad, delivery.PreviousKEKVersion)
		if err != nil {
			return nil, stack.Wrap(fmt.Errorf("eventforward: the previous signing secret of destination %s does not open: %w", delivery.DestinationID, err))
		}
		secrets = append(secrets, previous)
	}
	return secrets, nil
}

// retryable is a send a later attempt can get past: no answer, a 5xx or a 429.
type retryable struct{ err error }

func (r retryable) Error() string { return r.err.Error() }
func (r retryable) Unwrap() error { return r.err }

// secretAAD binds a sealed signing secret to its customer, connector and destination, so a blob
// copied onto another destination's row does not open. Each part is length-prefixed, as
// api.oauthClientAAD's are, with a prefix of its own, so no other secret's blob opens as this
// one. v1 changes with the layout.
func secretAAD(customerID, connectorID, id string) []byte {
	return fmt.Appendf(nil, "accelerate:connector-event-destination-secret:v1:%d:%s:%d:%s:%d:%s",
		len(customerID), customerID, len(connectorID), connectorID, len(id), id)
}
