// Package siptrunk places calls through a customer's own SIP trunk.
//
// Every other vendor is asked over HTTP to place a call and holds it on its own network.
// Here this process places it: it opens one SIP dialog to the customer's trunk and one to
// the Stream trunk made for the call, and copies SDP between them, so the audio goes from
// the carrier to Stream directly and never through here.
//
// The dialogs live on one node at a time. With Redis, that node records each connected call
// and holds a claim on it; when it stops it offers its calls to the other nodes, and when it
// dies its claims lapse, and either way another node continues both dialogs from the record.
// Without Redis, stopping the node hangs its calls up.
package siptrunk

import (
	"context"
	"errors"
	"fmt"
	"log/slog"
	"net/http"
	"strconv"
	"sync"
	"sync/atomic"
	"time"

	"github.com/google/uuid"
	"github.com/redis/rueidis"

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
	Media() sipbridge.MediaAddrs
	State() sipbridge.CallState
	Release()
}

type dialer func(ctx context.Context, config sipbridge.Config) (call, error)

type resumer func(ctx context.Context, cfg sipbridge.ResumeConfig, state sipbridge.CallState) (call, error)

// Options configures a Provider.
type Options struct {
	Logger *slog.Logger
	// Redis is where calls are recorded so another node can continue them. Nil keeps every
	// call on this node, and stopping it hangs them up.
	Redis rueidis.Client
	// Node names this process as the owner of the calls it holds. Empty is a random name.
	Node string

	holdFor, renewEvery time.Duration
	prefix              string
}

// Provider is the sip_trunk vendor.
type Provider struct {
	dial    dialer
	resume  resumer
	logger  *slog.Logger
	ctx     context.Context
	stop    context.CancelFunc
	running sync.WaitGroup

	redis      rueidis.Client
	node       string
	keys       keys
	holdFor    time.Duration
	renewEvery time.Duration
	resumer    Resumer
	loops      sync.WaitGroup

	// notTaken is set when Close let a recorded call go that no other node had taken.
	notTaken atomic.Bool

	mu        sync.Mutex
	calls     map[string]*heldCall
	stopLoops context.CancelFunc
	// closing is Close's context once it began, so a call adopted after that is handed over too.
	closing context.Context
}

// heldCall is a connected call this node holds.
type heldCall struct {
	call call
	// ref is what the record names, nil for a call no other node can continue.
	ref *phone.TrunkCallRef
	// movable is whether the call is recorded and claimed, so Close offers it to another node
	// rather than hanging it up.
	movable bool
}

// New returns a Provider that dials and resumes with sipbridge.
func New(options Options) *Provider {
	return newProvider(options, func(ctx context.Context, config sipbridge.Config) (call, error) {
		connected, err := sipbridge.Dial(ctx, config)
		if err != nil {
			return nil, err
		}
		return connected, nil
	}, func(ctx context.Context, cfg sipbridge.ResumeConfig, state sipbridge.CallState) (call, error) {
		resumed, err := sipbridge.Resume(ctx, cfg, state)
		if err != nil {
			return nil, err
		}
		return resumed, nil
	})
}

func newProvider(options Options, dial dialer, resume resumer) *Provider {
	if options.Logger == nil {
		options.Logger = slog.Default()
	}
	if options.Node == "" {
		options.Node = uuid.NewString()
	}
	if options.holdFor == 0 {
		options.holdFor = holdFor
	}
	if options.renewEvery == 0 {
		options.renewEvery = renewEvery
	}
	if options.prefix == "" {
		options.prefix = keyPrefix
	}
	ctx, stop := context.WithCancel(context.Background())
	return &Provider{
		dial: dial, resume: resume, logger: options.Logger, ctx: ctx, stop: stop, calls: map[string]*heldCall{},
		redis: options.Redis, node: options.Node, keys: keys(options.prefix),
		holdFor: options.holdFor, renewEvery: options.renewEvery,
	}
}

// StreamLeg is the number's own line, with the call id on the leg.
func (p *Provider) StreamLeg() phone.StreamLeg { return phone.StreamLegShared }

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
		Call:   sipbridge.CallParams{From: outbound.From, To: outbound.To, RingTimeout: outbound.RingTimeout, CallID: outbound.CallID},
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

	go p.run(id, config, outbound.Connected, outbound.Ended, outbound.Resume)
	return phone.Dialed{VendorCallID: id, Status: "queued"}, nil
}

// run holds one call from dialling to its end. It uses the node's context, never the
// request's: the request is answered, and its context cancelled, while the phone still rings.
// connected, when set, is told where each side's audio comes from once the call is up, and
// ended, when set, how the call ended. ref, when set, is what another node needs to
// continue the call.
func (p *Provider) run(id string, config sipbridge.Config, connected func(sipbridge.MediaAddrs), ended func(phone.CallStatus), ref *phone.TrunkCallRef) {
	defer p.running.Done()
	log := config.Logger
	if ended == nil {
		ended = func(phone.CallStatus) {}
	}

	// The bridge tells the delay on whichever goroutine reads the INVITE's responses.
	var postDialDelay atomic.Int64
	config.Call.Responded = func(d time.Duration) { postDialDelay.Store(int64(d)) }
	report := func(err error) {
		status := statusOf(err)
		status.PostDialDelay = time.Duration(postDialDelay.Load())
		ended(status)
	}
	changed := make(chan struct{}, 1)
	if p.redis != nil && ref != nil {
		config.OnChange = notify(changed)
	}

	bridged, err := p.dial(p.ctx, config)
	if err != nil {
		log.Warn("call through a customer's sip trunk did not connect", "error", err)
		report(err)
		return
	}
	log.Info("call through a customer's sip trunk connected")
	if connected != nil {
		// Its own goroutine, so a slow write of where the audio goes cannot hold up a hang-up.
		p.running.Go(func() { connected(bridged.Media()) })
	}
	if held := p.adopt(id, bridged, ref, false, log); held != nil {
		p.hold(id, held, changed, log, func() { report(nil) })
	}
}

// adopt records and claims a connected call, and adds it to the calls this node holds. It
// lets the call go and returns nil when another node holds it. takenOver is a call this node
// already won the claim on. A call adopted while Close runs is handed over at once.
func (p *Provider) adopt(id string, bridged call, ref *phone.TrunkCallRef, takenOver bool, log *slog.Logger) *heldCall {
	held := &heldCall{call: bridged, ref: ref}
	if p.redis != nil && ref != nil {
		claimed, err := p.claim(id, held)
		switch {
		case err != nil && takenOver:
			// The claim won before resuming is still this node's: renewal extends it, or lets
			// the call go if another node holds it.
			log.Warn("could not record a call through a customer's sip trunk taken over", "error", err)
			held.movable = true
		case err != nil:
			// The script may have run with only its reply lost. A record left behind would
			// let another node resume the call while this one still relays it, so none is
			// left, and Close hangs the call up as it would without Redis.
			log.Warn("could not record a call through a customer's sip trunk; it ends when this node stops", "error", err)
			p.forget(id, log)
		case !claimed:
			log.Info("another node holds a call through a customer's sip trunk; letting it go")
			bridged.Release()
			return nil
		default:
			held.movable = true
		}
	}
	p.mu.Lock()
	p.calls[id] = held
	closing := p.closing
	p.mu.Unlock()
	if closing != nil && held.movable {
		go p.handOver(closing, id, held)
	}
	return held
}

// hold keeps an adopted call until it ends or this node lets it go, and reports its end.
// changed is told when the call's dialogs changed, so its record is written again.
func (p *Provider) hold(id string, held *heldCall, changed <-chan struct{}, log *slog.Logger, report func()) {
	bridged := held.call
	defer func() {
		p.mu.Lock()
		delete(p.calls, id)
		p.mu.Unlock()
	}()

	// A movable call is offered to another node by Close instead.
	var stopping <-chan struct{}
	if !held.movable {
		stopping = p.ctx.Done()
	}
	for {
		select {
		case <-changed:
			if held.movable {
				p.rewrite(id, held, log)
			}
		case <-bridged.Done():
			err := bridged.Err()
			if errors.Is(err, sipbridge.ErrReleased) {
				log.Info("let a call through a customer's sip trunk go to another node")
				return
			}
			if err != nil {
				log.Warn("call through a customer's sip trunk ended", "error", err)
			} else {
				log.Info("call through a customer's sip trunk ended")
			}
			report()
			if held.movable {
				p.forget(id, log)
			}
			return
		case <-stopping:
			ctx, cancel := context.WithTimeout(context.Background(), hangupTimeout)
			err := bridged.Hangup(ctx)
			cancel()
			if err != nil {
				log.Warn("could not hang up a call through a customer's sip trunk", "error", err)
			} else {
				log.Info("hung up a call through a customer's sip trunk because the node is stopping")
			}
			report()
			return
		}
	}
}

// notify is an OnChange that never blocks the bridge: one pending change stands for any
// number, since each rewrite reads the call's whole state.
func notify(changed chan<- struct{}) func() {
	return func() {
		select {
		case changed <- struct{}{}:
		default:
		}
	}
}

// statusOf is how a call ended, from what dialling it returned: nil once an answered call
// has ended, or why it never connected.
// context.Canceled is the node stopping, or Stream hanging up while the customer rang.
func statusOf(err error) phone.CallStatus {
	if err == nil {
		// The answer was a 2xx, and RFC 3261 §21.2 defines no 2xx but 200 OK.
		return phone.CallStatus{Result: phone.CallCompleted, Word: "bye", SIPCode: 200}
	}
	if u, ok := errors.AsType[*sipbridge.Unanswered](err); ok {
		switch u.Code {
		case 0:
			return phone.CallStatus{Result: phone.CallNoAnswer, Word: "ring timeout"}
		case 486, 603:
			return phone.CallStatus{Result: phone.CallBusy, Word: strconv.Itoa(u.Code), SIPCode: u.Code}
		case 408, 480:
			return phone.CallStatus{Result: phone.CallNoAnswer, Word: strconv.Itoa(u.Code), SIPCode: u.Code}
		}
		return phone.CallStatus{Result: phone.CallFailed, Word: strconv.Itoa(u.Code), SIPCode: u.Code}
	}
	if errors.Is(err, context.Canceled) {
		return phone.CallStatus{Result: phone.CallCanceled, Word: "canceled"}
	}
	return phone.CallStatus{Result: phone.CallFailed, Word: "error"}
}

// Close stops placing and taking calls, offers each recorded call to another node and lets
// it go once one took it, hangs up the rest, and waits for them, up to ctx. A recorded call
// no node took by then is let go too, and its record stays for the next node to scan.
func (p *Provider) Close(ctx context.Context) error {
	p.mu.Lock()
	p.stop()
	p.closing = ctx
	stopLoops := p.stopLoops
	// Under the lock adopt registers under, so each call is handed over once: here, or by
	// adopt for a call that connects or is taken over from now on.
	for id, held := range p.calls {
		if held.movable {
			go p.handOver(ctx, id, held)
		}
	}
	p.mu.Unlock()
	if stopLoops != nil {
		stopLoops()
	}
	// Before running: a takeover still resuming adds its call to running.
	if err := waitFor(ctx, &p.loops); err != nil {
		return stack.Wrap(fmt.Errorf("phone: still taking over a call through a customer's sip trunk at shutdown: %w", err))
	}
	if err := waitFor(ctx, &p.running); err != nil {
		return stack.Wrap(fmt.Errorf("phone: calls through customers' sip trunks still up at shutdown: %w", err))
	}
	if p.notTaken.Load() {
		return stack.Wrap(fmt.Errorf("phone: no other node took a call through a customer's sip trunk before shutdown: %w", ctx.Err()))
	}
	return nil
}

func waitFor(ctx context.Context, wg *sync.WaitGroup) error {
	finished := make(chan struct{})
	go func() {
		wg.Wait()
		close(finished)
	}()
	select {
	case <-finished:
		return nil
	case <-ctx.Done():
		return ctx.Err()
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
	return stack.Wrap(phone.Invalid(errors.New("phone: digits cannot be pressed on a call through a customer's own sip trunk")))
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
