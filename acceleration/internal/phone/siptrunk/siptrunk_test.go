package siptrunk

import (
	"context"
	"errors"
	"log/slog"
	"sync"
	"testing"
	"time"

	"github.com/google/uuid"
	"github.com/stretchr/testify/suite"

	"github.com/GetStream/Vision-Agents/acceleration/internal/phone"
	"github.com/GetStream/Vision-Agents/acceleration/internal/phone/sipbridge"
)

// fakeCall is a connected call that ends when it is hung up.
type fakeCall struct {
	done     chan struct{}
	once     sync.Once
	hungUp   bool
	hangupMu sync.Mutex
}

func newFakeCall() *fakeCall { return &fakeCall{done: make(chan struct{})} }

func (c *fakeCall) Done() <-chan struct{} { return c.done }
func (c *fakeCall) Err() error            { return nil }
func (c *fakeCall) Hangup(context.Context) error {
	c.hangupMu.Lock()
	c.hungUp = true
	c.hangupMu.Unlock()
	c.once.Do(func() { close(c.done) })
	return nil
}

func (c *fakeCall) wasHungUp() bool {
	c.hangupMu.Lock()
	defer c.hangupMu.Unlock()
	return c.hungUp
}

// fakeDialer records what it was asked to dial and answers with call or err.
type fakeDialer struct {
	mu      sync.Mutex
	ctx     context.Context
	config  sipbridge.Config
	call    *fakeCall
	err     error
	blocks  bool // the dial waits for ctx to end, like a phone nobody picks up
	dialled chan struct{}
}

func (d *fakeDialer) dial(ctx context.Context, config sipbridge.Config) (call, error) {
	d.mu.Lock()
	d.ctx, d.config = ctx, config
	d.mu.Unlock()
	close(d.dialled)
	if d.blocks {
		<-ctx.Done()
		return nil, ctx.Err()
	}
	if d.err != nil {
		return nil, d.err
	}
	return d.call, nil
}

type SIPTrunkSuite struct {
	suite.Suite
	dialer   *fakeDialer
	provider *Provider
}

func TestSIPTrunkSuite(t *testing.T) { suite.Run(t, new(SIPTrunkSuite)) }

func (s *SIPTrunkSuite) SetupTest() {
	s.dialer = &fakeDialer{call: newFakeCall(), dialled: make(chan struct{})}
	s.provider = newProvider(Options{Logger: slog.New(slog.DiscardHandler)}, s.dialer.dial)
}

func (s *SIPTrunkSuite) TearDownTest() {
	ctx, cancel := context.WithTimeout(context.Background(), time.Second)
	defer cancel()
	s.Require().NoError(s.provider.Close(ctx))
}

func outbound() phone.Outbound {
	return phone.Outbound{
		From: "+15550000301",
		To:   "+15550000302",
		Bridge: phone.Bridge{
			URI: "sip:+15550000301@bridge.sip.example.com", Username: "stream-user", Password: "stream-pass",
		},
		RingTimeout: 20 * time.Second,
		Trunk: &phone.SIPTrunk{
			Host: "trunk.example.com", Port: 5060, Transport: "tcp", Username: "agent",
			Password: "s3cret", LateOffer: true, Codecs: []string{"PCMU", "PCMA"},
		},
	}
}

func (s *SIPTrunkSuite) waitForDial() {
	select {
	case <-s.dialer.dialled:
	case <-time.After(time.Second):
		s.FailNow("the call was never dialled")
	}
}

// live takes the provider rather than reading s.provider, because an Eventually check can
// still be running when the next test's SetupTest replaces it.
func live(p *Provider, id string) bool {
	p.mu.Lock()
	defer p.mu.Unlock()
	_, ok := p.calls[id]
	return ok
}

// waitForCalls waits until every call's goroutine has returned, without closing the provider.
func (s *SIPTrunkSuite) waitForCalls() {
	finished := make(chan struct{})
	go func() {
		s.provider.running.Wait()
		close(finished)
	}()
	select {
	case <-finished:
	case <-time.After(time.Second):
		s.FailNow("a call is still running")
	}
}

func (s *SIPTrunkSuite) TestDialReturnsAtOnceAsQueued() {
	dialed, err := s.provider.Dial(context.Background(), outbound())

	s.Require().NoError(err)
	s.Equal("queued", dialed.Status)
	_, err = uuid.Parse(dialed.VendorCallID)
	s.NoError(err)
}

func (s *SIPTrunkSuite) TestTheBridgeIsGivenTheTrunkAndTheStreamLeg() {
	_, err := s.provider.Dial(context.Background(), outbound())
	s.Require().NoError(err)
	s.waitForDial()

	s.dialer.mu.Lock()
	got := s.dialer.config
	s.dialer.mu.Unlock()
	got.Logger = nil
	want := sipbridge.Config{
		CustomerTrunk: sipbridge.CustomerTrunk{
			Host: "trunk.example.com", Port: 5060, Transport: "tcp", Username: "agent",
			Password: "s3cret", LateOffer: true, Codecs: []string{"PCMU", "PCMA"},
		},
		Stream: sipbridge.StreamTrunk{
			URI: "sip:+15550000301@bridge.sip.example.com", Username: "stream-user", Password: "stream-pass",
		},
		Call: sipbridge.CallParams{From: "+15550000301", To: "+15550000302", RingTimeout: 20 * time.Second},
	}.WithDefaults()
	want.Logger = nil
	s.Equal(want, got)
}

func (s *SIPTrunkSuite) TestCancellingTheRequestDoesNotEndTheCall() {
	request, cancel := context.WithCancel(context.Background())
	dialed, err := s.provider.Dial(request, outbound())
	s.Require().NoError(err)
	cancel()
	s.waitForDial()

	s.dialer.mu.Lock()
	dialCtx := s.dialer.ctx
	s.dialer.mu.Unlock()
	s.NoError(dialCtx.Err(), "the call runs on the node's context, not the request's")
	p := s.provider
	s.Eventually(func() bool { return live(p, dialed.VendorCallID) }, time.Second, 10*time.Millisecond)
	s.False(s.dialer.call.wasHungUp())
}

func (s *SIPTrunkSuite) TestAnEndedCallIsForgotten() {
	dialed, err := s.provider.Dial(context.Background(), outbound())
	s.Require().NoError(err)
	p := s.provider
	s.Eventually(func() bool { return live(p, dialed.VendorCallID) }, time.Second, 10*time.Millisecond)

	s.Require().NoError(s.dialer.call.Hangup(context.Background()))
	s.waitForCalls()
	s.False(live(p, dialed.VendorCallID))
}

func (s *SIPTrunkSuite) TestAFailedDialIsForgotten() {
	s.dialer.err = errors.New("407 Proxy Authentication Required")

	dialed, err := s.provider.Dial(context.Background(), outbound())
	s.Require().NoError(err, "the failure happens after the request has been answered")
	s.waitForCalls()
	s.False(live(s.provider, dialed.VendorCallID))
}

func (s *SIPTrunkSuite) TestCloseHangsUpEveryCall() {
	dialed, err := s.provider.Dial(context.Background(), outbound())
	s.Require().NoError(err)
	p := s.provider
	s.Eventually(func() bool { return live(p, dialed.VendorCallID) }, time.Second, 10*time.Millisecond)

	ctx, cancel := context.WithTimeout(context.Background(), time.Second)
	defer cancel()
	s.Require().NoError(s.provider.Close(ctx))

	s.True(s.dialer.call.wasHungUp())
	s.False(live(p, dialed.VendorCallID))
}

func (s *SIPTrunkSuite) TestCloseCancelsACallStillBeingSetUp() {
	s.dialer.blocks = true

	dialed, err := s.provider.Dial(context.Background(), outbound())
	s.Require().NoError(err)
	s.Equal("queued", dialed.Status)
	s.waitForDial()

	ctx, cancel := context.WithTimeout(context.Background(), time.Second)
	defer cancel()
	s.Require().NoError(s.provider.Close(ctx))

	s.dialer.mu.Lock()
	dialCtx := s.dialer.ctx
	s.dialer.mu.Unlock()
	s.ErrorIs(dialCtx.Err(), context.Canceled)
	s.False(live(s.provider, dialed.VendorCallID))
	s.False(s.dialer.call.wasHungUp(), "a call that never connected has nothing to hang up")
}

func (s *SIPTrunkSuite) TestADialAfterCloseIsRefused() {
	ctx, cancel := context.WithTimeout(context.Background(), time.Second)
	defer cancel()
	s.Require().NoError(s.provider.Close(ctx))

	_, err := s.provider.Dial(context.Background(), outbound())
	s.EqualError(err, "phone: this node is shutting down and places no more calls")
}

func (s *SIPTrunkSuite) TestACallNeedsATrunkWithAPassword() {
	withoutTrunk := outbound()
	withoutTrunk.Trunk = nil
	_, err := s.provider.Dial(context.Background(), withoutTrunk)
	s.EqualError(err, "phone: a call through a customer's sip trunk needs the trunk")

	withoutPassword := outbound()
	withoutPassword.Trunk.Password = ""
	_, err = s.provider.Dial(context.Background(), withoutPassword)
	s.EqualError(err, "phone: the sip trunk has no password; set one before calling through it")
}

func (s *SIPTrunkSuite) TestOnlyTheRingTimeoutCanBeExpressed() {
	s.True(s.provider.Dials(phone.FeatureRingTimeout))
	s.False(s.provider.Dials(phone.FeatureInitialDigits))
	s.False(s.provider.Dials(phone.FeatureCustomHeaders))
	s.Equal(phone.SIPTrunkVendor, s.provider.Vendor())
}

func (s *SIPTrunkSuite) TestNothingIsBoughtOrPressedHere() {
	ctx := context.Background()
	const notSold = "phone: numbers on a customer's own sip trunk are the customer's, not bought here"
	_, err := s.provider.SearchNumbers(ctx, phone.Search{})
	s.EqualError(err, notSold)
	_, err = s.provider.BuyNumber(ctx, phone.Order{})
	s.EqualError(err, notSold)
	s.EqualError(s.provider.ReleaseNumber(ctx, "+15550000301"), notSold)
	s.EqualError(s.provider.ConfigureInbound(ctx, phone.Inbound{}),
		"phone: inbound calls on a customer's own sip trunk are not supported")
	s.EqualError(s.provider.SendDigits(ctx, "id", "1"),
		"phone: digits cannot be pressed on a call through a customer's own sip trunk")
}

func (s *SIPTrunkSuite) TestClosingTheServiceHangsUpThisProvidersCalls() {
	service, err := phone.NewService(phone.ServiceOptions{
		Registry: phone.NewRegistry(phone.Config{}), SIPTrunks: s.provider,
	})
	s.Require().NoError(err)
	dialed, err := s.provider.Dial(context.Background(), outbound())
	s.Require().NoError(err)
	p := s.provider
	s.Eventually(func() bool { return live(p, dialed.VendorCallID) }, time.Second, 10*time.Millisecond)

	ctx, cancel := context.WithTimeout(context.Background(), time.Second)
	defer cancel()
	s.Require().NoError(service.Close(ctx))

	s.True(s.dialer.call.wasHungUp())
	s.False(live(p, dialed.VendorCallID))
}
