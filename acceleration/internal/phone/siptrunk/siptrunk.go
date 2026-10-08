// Package siptrunk places calls through a customer's own SIP trunk.
//
// Every other vendor is asked over HTTP to place a call and holds it on its own network.
// Here this process places it: it opens one SIP dialog to the customer's trunk and one to
// the Stream trunk made for the call, and copies SDP between them, so the audio goes from
// the carrier to Stream directly and never through here. The dialogs live on the node that
// placed the call until it ends, which is why stopping the node hangs them all up.
package siptrunk

import (
	"context"
	"errors"
	"fmt"
	"log/slog"
	"net/http"
	"sync"
	"time"

	"github.com/google/uuid"

	"github.com/GetStream/Vision-Agents/acceleration/internal/phone"
	"github.com/GetStream/Vision-Agents/acceleration/internal/phone/sipbridge"
	"github.com/GetStream/Vision-Agents/acceleration/internal/stack"
)

// hangupTimeout bounds the BYEs sent to one call when the node stops.
const hangupTimeout = 5 * time.Second

var errNotSold = errors.New("phone: numbers on a customer's own sip trunk are the customer's, not bought here")

// call is what the provider needs of a connected call, so tests can stand in for one.
type call interface {
	Done() <-chan struct{}
	Err() error
	Hangup(ctx context.Context) error
}

type dialer func(ctx context.Context, config sipbridge.Config) (call, error)

// Options configures a Provider.
type Options struct {
	Logger *slog.Logger
}

// Provider is the sip_trunk vendor.
type Provider struct {
	dial    dialer
	logger  *slog.Logger
	ctx     context.Context
	stop    context.CancelFunc
	running sync.WaitGroup

	mu    sync.Mutex
	calls map[string]call
}

// New returns a Provider that dials with sipbridge.
func New(options Options) *Provider {
	return newProvider(options, func(ctx context.Context, config sipbridge.Config) (call, error) {
		connected, err := sipbridge.Dial(ctx, config)
		if err != nil {
			return nil, err
		}
		return connected, nil
	})
}

func newProvider(options Options, dial dialer) *Provider {
	if options.Logger == nil {
		options.Logger = slog.Default()
	}
	ctx, stop := context.WithCancel(context.Background())
	return &Provider{dial: dial, logger: options.Logger, ctx: ctx, stop: stop, calls: map[string]call{}}
}

// Dial starts the call and returns before anyone answers, as every vendor's API does. What
// happens after, including a call that never connects, is in the logs under its
// vendor_call_id.
func (p *Provider) Dial(_ context.Context, outbound phone.Outbound) (phone.Dialed, error) {
	if outbound.Trunk == nil {
		return phone.Dialed{}, stack.Wrap(errors.New("phone: a call through a customer's sip trunk needs the trunk"))
	}
	if outbound.Trunk.Password == "" {
		return phone.Dialed{}, stack.Wrap(errors.New("phone: the sip trunk has no password; set one before calling through it"))
	}
	if err := outbound.Validate(); err != nil {
		return phone.Dialed{}, err
	}

	id := uuid.NewString()
	trunk := outbound.Trunk
	config := sipbridge.Config{
		CustomerTrunk: sipbridge.CustomerTrunk{
			Host: trunk.Host, Port: trunk.Port, Transport: trunk.Transport, Username: trunk.Username,
			Password: trunk.Password, LateOffer: trunk.LateOffer, Codecs: trunk.Codecs,
		},
		Stream: sipbridge.StreamTrunk{
			URI: outbound.Bridge.URI, Username: outbound.Bridge.Username, Password: outbound.Bridge.Password,
		},
		Call:   sipbridge.CallParams{From: outbound.From, To: outbound.To, RingTimeout: outbound.RingTimeout},
		Logger: p.logger.With("vendor", phone.SIPTrunkVendor, "vendor_call_id", id),
	}.WithDefaults()
	if err := config.Validate(); err != nil {
		return phone.Dialed{}, stack.Wrap(fmt.Errorf("phone: %w", err))
	}

	// Checked under the lock Close cancels under, so no call starts after Close has begun
	// waiting for the ones already running.
	p.mu.Lock()
	if p.ctx.Err() != nil {
		p.mu.Unlock()
		return phone.Dialed{}, stack.Wrap(errors.New("phone: this node is shutting down and places no more calls"))
	}
	p.running.Add(1)
	p.mu.Unlock()

	go p.run(id, config)
	return phone.Dialed{VendorCallID: id, Status: "queued"}, nil
}

// run holds one call from dialling to its end. It uses the node's context, never the
// request's: the request is answered, and its context cancelled, while the phone still rings.
func (p *Provider) run(id string, config sipbridge.Config) {
	defer p.running.Done()
	log := config.Logger

	connected, err := p.dial(p.ctx, config)
	if err != nil {
		log.Warn("call through a customer's sip trunk did not connect", "error", err)
		return
	}
	p.mu.Lock()
	p.calls[id] = connected
	p.mu.Unlock()
	defer func() {
		p.mu.Lock()
		delete(p.calls, id)
		p.mu.Unlock()
	}()
	log.Info("call through a customer's sip trunk connected")

	select {
	case <-connected.Done():
		if err := connected.Err(); err != nil {
			log.Warn("call through a customer's sip trunk ended", "error", err)
			return
		}
		log.Info("call through a customer's sip trunk ended")
	case <-p.ctx.Done():
		ctx, cancel := context.WithTimeout(context.Background(), hangupTimeout)
		defer cancel()
		if err := connected.Hangup(ctx); err != nil {
			log.Warn("could not hang up a call through a customer's sip trunk", "error", err)
			return
		}
		log.Info("hung up a call through a customer's sip trunk because the node is stopping")
	}
}

// Close hangs up every call this node holds and waits for them, up to ctx.
func (p *Provider) Close(ctx context.Context) error {
	p.mu.Lock()
	p.stop()
	p.mu.Unlock()

	finished := make(chan struct{})
	go func() {
		p.running.Wait()
		close(finished)
	}()
	select {
	case <-finished:
		return nil
	case <-ctx.Done():
		return stack.Wrap(fmt.Errorf("phone: calls through customers' sip trunks still up at shutdown: %w", ctx.Err()))
	}
}

// SearchNumbers finds nothing: these numbers are the customer's own.
func (p *Provider) SearchNumbers(context.Context, phone.Search) ([]phone.Available, error) {
	return nil, stack.Wrap(errNotSold)
}

// BuyNumber buys nothing.
func (p *Provider) BuyNumber(context.Context, phone.Order) (phone.Number, error) {
	return phone.Number{}, stack.Wrap(errNotSold)
}

// ReleaseNumber gives nothing back.
func (p *Provider) ReleaseNumber(context.Context, string) error { return stack.Wrap(errNotSold) }

// ConfigureInbound is not supported: calls into these numbers reach the customer's own
// switch, not here.
func (p *Provider) ConfigureInbound(context.Context, phone.Inbound) error {
	return stack.Wrap(errors.New("phone: inbound calls on a customer's own sip trunk are not supported"))
}

// SendDigits cannot work: digits travel in the audio, which does not pass through here.
func (p *Provider) SendDigits(context.Context, string, string) error {
	return stack.Wrap(errors.New("phone: digits cannot be pressed on a call through a customer's own sip trunk"))
}

// Supports says no filter: there is nothing to search.
func (p *Provider) Supports(phone.Filter) bool { return false }

// Dials says which of a call's terms this can carry. Only the ring timeout: initial digits
// would be sent in the audio, and custom headers are not passed on.
func (p *Provider) Dials(feature phone.CallFeature) bool { return feature == phone.FeatureRingTimeout }

// Vendor is sip_trunk.
func (p *Provider) Vendor() string { return phone.SIPTrunkVendor }

// Client is the default client: nothing here speaks HTTP.
func (p *Provider) Client() *http.Client { return http.DefaultClient }

var _ phone.Provider = (*Provider)(nil)
