package main

import (
	"context"
	"fmt"
	"io"
	"log/slog"
	"net"
	"net/http"
	"net/netip"
	"sync"
	"testing"
	"time"

	"github.com/emiago/sipgo"
	"github.com/emiago/sipgo/sip"
	"github.com/stretchr/testify/suite"

	"github.com/GetStream/Vision-Agents/acceleration/internal/phone"
	"github.com/GetStream/Vision-Agents/acceleration/internal/phone/sipbridge"
)

// EndpointSuite runs real calls into the bench over SIP and RTP on loopback, through a
// carrier that delays each direction by a known amount.
type EndpointSuite struct {
	suite.Suite
	ctx      context.Context
	cancel   context.CancelFunc
	host     netip.Addr
	port     int
	endpoint *Endpoint
}

func TestEndpointSuite(t *testing.T) {
	suite.Run(t, new(EndpointSuite))
}

func (s *EndpointSuite) SetupTest() {
	s.ctx, s.cancel = context.WithTimeout(context.Background(), 30*time.Second)
	s.host = netip.MustParseAddr("127.0.0.1")
	s.port = freeUDPPort(s.T())
	low := freeUDPPort(s.T())
	rtpPorts, err := newPorts(low, min(low+50, 65535))
	s.Require().NoError(err)
	s.endpoint, err = NewEndpoint(fmt.Sprintf("127.0.0.1:%d", s.port), s.host, rtpPorts, quiet())
	s.Require().NoError(err)
}

func (s *EndpointSuite) TearDownTest() {
	s.cancel()
	s.NoError(s.endpoint.Close())
}

// prober measures calls placed by carrier, from one number to another.
func (s *EndpointSuite) prober(carrier phone.Provider, patience time.Duration) *Prober {
	return &Prober{
		Endpoint:  s.endpoint,
		Providers: map[string]phone.Provider{"loopback": carrier},
		Host:      s.host,
		Port:      s.port,
		Stamps:    3,
		Gap:       500 * time.Millisecond,
		Patience:  patience,
	}
}

func (s *EndpointSuite) TestACallMeasuresTheDelayEachWayThroughTheCarrier() {
	carrier := newLoopbackCarrier(s.T(), s.host, s.port, 120*time.Millisecond, 60*time.Millisecond)
	caller := Line{Vendor: "loopback", City: "Chicago", E164: "+13125550001"}
	callee := Line{Vendor: "loopback", City: "Chicago", E164: "+13125550002"}

	result := s.prober(carrier, 5*time.Second).Probe(s.ctx, caller, callee)

	s.Require().Empty(result.Error)
	s.Equal(3, result.Sent)
	s.Require().Len(result.ForwardMs, 3)
	s.Require().Len(result.BackwardMs, 3)
	for _, ms := range result.ForwardMs {
		s.InDelta(135, ms, 20, "caller → callee is the carrier's 120 ms plus loopback scheduling")
	}
	for _, ms := range result.BackwardMs {
		s.InDelta(75, ms, 20, "callee → caller is the carrier's 60 ms plus loopback scheduling")
	}
	s.Zero(result.Garbled)
	s.Equal(carrier.calleeMedia(), result.CalleeMedia)
	s.Equal(carrier.callerMedia(), result.CallerMedia)
	s.Positive(result.RingMs)
}

func (s *EndpointSuite) TestACallWhoseLegNeverArrivesIsReportedNotLost() {
	carrier := &loopbackCarrier{silent: true}
	caller := Line{Vendor: "loopback", City: "Chicago", E164: "+13125550001"}
	callee := Line{Vendor: "loopback", City: "Chicago", E164: "+13125550002"}

	result := s.prober(carrier, 300*time.Millisecond).Probe(s.ctx, caller, callee)

	s.Contains(result.Error, "the callee's leg did not arrive")
	s.Empty(result.ForwardMs)
}

func (s *EndpointSuite) TestACallNobodyIsWaitingForIsTurnedAway() {
	client := newSIPClient(s.T())
	recipient := sip.Uri{User: "+19995550000", Host: "127.0.0.1", Port: s.port}

	dialog, err := client.dialogs.Invite(s.ctx, recipient, offer(s.host, 40000))
	s.Require().NoError(err)
	err = dialog.WaitAnswer(s.ctx, sipgo.AnswerOptions{})

	var refused *sipgo.ErrDialogResponse
	s.Require().ErrorAs(err, &refused)
	s.Equal(sip.StatusNotFound, refused.Res.StatusCode)
}

func (s *EndpointSuite) TestANumberIsFiledByItsLastTenDigitsWhateverTheCarrierPrefixes() {
	s.Equal("3125550001", legKey("+13125550001"))
	s.Equal("3125550001", legKey("13125550001"))
	s.Equal("3125550001", legKey("3125550001"))
	s.Equal("probe-ab12", legKey("probe-ab12"))
}

// loopbackCarrier is a phone vendor on this machine: dialling sends the callee's leg and
// then the caller's to the bench, and every RTP packet between them is held for a fixed
// time, which is the latency the bench should measure.
type loopbackCarrier struct {
	t        *testing.T
	sip      *sipClient
	host     netip.Addr
	benchSIP int
	// forward delays the caller's side to the callee's, backward the other way.
	forward, backward time.Duration
	// silent drops every call, as a carrier that never delivers it.
	silent bool

	mu                     sync.Mutex
	calleeConn, callerConn *net.UDPConn
}

func newLoopbackCarrier(t *testing.T, host netip.Addr, benchSIP int, forward, backward time.Duration) *loopbackCarrier {
	return &loopbackCarrier{
		t:        t,
		sip:      newSIPClient(t),
		host:     host,
		benchSIP: benchSIP,
		forward:  forward,
		backward: backward,
	}
}

func (c *loopbackCarrier) Dial(ctx context.Context, outbound phone.Outbound) (phone.Dialed, error) {
	if c.silent {
		return phone.Dialed{VendorCallID: "dropped"}, nil
	}
	var bridge sip.Uri
	if err := sip.ParseUri(outbound.Bridge.URI, &bridge); err != nil {
		return phone.Dialed{}, err
	}
	go func() {
		calleeConn, calleeBench, err := c.leg(ctx, sip.Uri{User: outbound.To, Host: c.host.String(), Port: c.benchSIP})
		if err != nil {
			c.t.Errorf("callee leg: %v", err)
			return
		}
		callerConn, callerBench, err := c.leg(ctx, bridge)
		if err != nil {
			c.t.Errorf("caller leg: %v", err)
			return
		}
		c.mu.Lock()
		c.calleeConn, c.callerConn = calleeConn, callerConn
		c.mu.Unlock()
		go relay(callerConn, calleeConn, calleeBench, c.forward)
		go relay(calleeConn, callerConn, callerBench, c.backward)
	}()
	return phone.Dialed{VendorCallID: "loopback", Status: "dialing"}, nil
}

// leg places one leg into the bench and returns the carrier's socket for it and where the
// bench wants that leg's media.
func (c *loopbackCarrier) leg(ctx context.Context, to sip.Uri) (*net.UDPConn, netip.AddrPort, error) {
	conn, err := net.ListenUDP("udp4", &net.UDPAddr{IP: net.IPv4(127, 0, 0, 1)})
	if err != nil {
		return nil, netip.AddrPort{}, err
	}
	c.t.Cleanup(func() { _ = conn.Close() })

	dialog, err := c.sip.dialogs.Invite(ctx, to, offer(c.host, conn.LocalAddr().(*net.UDPAddr).Port))
	if err != nil {
		return nil, netip.AddrPort{}, err
	}
	if err := dialog.WaitAnswer(ctx, sipgo.AnswerOptions{}); err != nil {
		return nil, netip.AddrPort{}, err
	}
	if err := dialog.Ack(ctx); err != nil {
		return nil, netip.AddrPort{}, err
	}
	answer, err := sipbridge.ParseMedia(dialog.InviteResponse.Body())
	if err != nil {
		return nil, netip.AddrPort{}, err
	}
	return conn, netip.AddrPortFrom(netip.MustParseAddr(answer.Addr), uint16(answer.Port)), nil
}

func (c *loopbackCarrier) calleeMedia() string {
	c.mu.Lock()
	defer c.mu.Unlock()
	return c.calleeConn.LocalAddr().String()
}

func (c *loopbackCarrier) callerMedia() string {
	c.mu.Lock()
	defer c.mu.Unlock()
	return c.callerConn.LocalAddr().String()
}

// relay copies what the bench sends on one leg to the bench's other leg, late by delay.
func relay(from, to *net.UDPConn, bench netip.AddrPort, delay time.Duration) {
	buffer := make([]byte, 1500)
	for {
		n, err := from.Read(buffer)
		if err != nil {
			return
		}
		packet := append([]byte(nil), buffer[:n]...)
		time.AfterFunc(delay, func() { _, _ = to.WriteToUDPAddrPort(packet, bench) })
	}
}

func (c *loopbackCarrier) SearchNumbers(context.Context, phone.Search) ([]phone.Available, error) {
	return nil, phone.ErrNotImplemented
}

func (c *loopbackCarrier) BuyNumber(context.Context, phone.Order) (phone.Number, error) {
	return phone.Number{}, phone.ErrNotImplemented
}

func (c *loopbackCarrier) ReleaseNumber(context.Context, string) error {
	return phone.ErrNotImplemented
}

func (c *loopbackCarrier) ConfigureInbound(context.Context, phone.Inbound) error { return nil }

func (c *loopbackCarrier) SendDigits(context.Context, string, string) error {
	return phone.ErrNotImplemented
}

func (c *loopbackCarrier) Supports(phone.Filter) bool { return false }

func (c *loopbackCarrier) Dials(phone.CallFeature) bool { return false }

func (c *loopbackCarrier) Vendor() string { return "loopback" }

func (c *loopbackCarrier) Client() *http.Client { return http.DefaultClient }

// sipClient places calls and answers the BYEs the bench sends to end them.
type sipClient struct {
	dialogs *sipgo.DialogClientCache
}

func newSIPClient(t *testing.T) *sipClient {
	port := freeUDPPort(t)
	ua, err := sipgo.NewUA()
	if err != nil {
		t.Fatal(err)
	}
	server, err := sipgo.NewServer(ua)
	if err != nil {
		t.Fatal(err)
	}
	client, err := sipgo.NewClient(ua, sipgo.WithClientHostname("127.0.0.1"), sipgo.WithClientPort(port))
	if err != nil {
		t.Fatal(err)
	}
	contact := sip.ContactHeader{Address: sip.Uri{User: "carrier", Host: "127.0.0.1", Port: port}}
	dialogs := sipgo.NewDialogClientCache(client, contact)
	server.OnBye(func(req *sip.Request, tx sip.ServerTransaction) {
		if err := dialogs.ReadBye(req, tx); err != nil {
			_ = tx.Respond(sip.NewResponseFromRequest(req, sip.StatusCallTransactionDoesNotExists, "Gone", nil))
		}
	})

	conn, err := net.ListenPacket("udp4", fmt.Sprintf("127.0.0.1:%d", port))
	if err != nil {
		t.Fatal(err)
	}
	go func() { _ = server.ServeUDP(conn) }()
	t.Cleanup(func() {
		_ = conn.Close()
		_ = ua.Close()
	})
	return &sipClient{dialogs: dialogs}
}

func offer(host netip.Addr, port int) []byte {
	return answerSDP(host, port)
}

func freeUDPPort(t *testing.T) int {
	conn, err := net.ListenUDP("udp4", &net.UDPAddr{IP: net.IPv4(127, 0, 0, 1)})
	if err != nil {
		t.Fatal(err)
	}
	defer conn.Close()
	return conn.LocalAddr().(*net.UDPAddr).Port
}

func quiet() *slog.Logger { return slog.New(slog.NewTextHandler(io.Discard, nil)) }
