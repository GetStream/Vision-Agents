package main

import (
	"context"
	"errors"
	"fmt"
	"log/slog"
	"net"
	"net/netip"
	"strings"
	"sync"
	"time"

	"github.com/emiago/sipgo"
	"github.com/emiago/sipgo/sip"

	"github.com/GetStream/Vision-Agents/acceleration/internal/phone/sipbridge"
)

// Endpoint is the bench's SIP address. Both legs of every call it measures end here: the
// called number's carrier sends the call it received, and the calling number's carrier
// bridges the call it placed, so the two directions of a call are timed on one clock.
//
// It only answers. A leg is told apart by the user part of the address it was sent to:
// the called number for the callee's leg, a probe id for the caller's.
type Endpoint struct {
	host  netip.Addr
	ports *ports
	log   *slog.Logger

	ua      *sipgo.UserAgent
	server  *sipgo.Server
	conn    net.PacketConn
	dialogs *sipgo.DialogServerCache

	mu       sync.Mutex
	expected map[string]chan *Leg
}

// Leg is one carrier's side of a call, answered by the bench.
type Leg struct {
	// Signal is where the carrier's INVITE came from.
	Signal netip.AddrPort
	// Media is the leg's audio.
	Media *Media
	// Answered is when the bench answered it.
	Answered time.Time

	dialog *sipgo.DialogServerSession
}

// Done is closed when the leg ends, from either side.
func (l *Leg) Done() <-chan struct{} { return l.dialog.Context().Done() }

// Hangup ends the leg. It is safe to call on a leg the carrier already hung up.
func (l *Leg) Hangup(ctx context.Context) error {
	defer l.dialog.Close()
	defer l.Media.Close()
	return l.dialog.Bye(ctx)
}

// NewEndpoint starts listening for calls on listen, a UDP host:port, and answers them with
// media on host, the address carriers reach this machine at.
func NewEndpoint(listen string, host netip.Addr, rtpPorts *ports, log *slog.Logger) (*Endpoint, error) {
	ua, err := sipgo.NewUA(sipgo.WithUserAgent("telephony-latency"))
	if err != nil {
		return nil, fmt.Errorf("sip user agent: %w", err)
	}
	server, err := sipgo.NewServer(ua)
	if err != nil {
		return nil, fmt.Errorf("sip server: %w", err)
	}
	client, err := sipgo.NewClient(ua)
	if err != nil {
		return nil, fmt.Errorf("sip client: %w", err)
	}

	port, err := listenPort(listen)
	if err != nil {
		return nil, err
	}
	contact := sip.ContactHeader{Address: sip.Uri{User: "bench", Host: host.String(), Port: port, UriParams: sip.NewParams()}}

	e := &Endpoint{
		host:     host,
		ports:    rtpPorts,
		log:      log,
		ua:       ua,
		server:   server,
		dialogs:  sipgo.NewDialogServerCache(client, contact),
		expected: map[string]chan *Leg{},
	}
	server.OnInvite(e.onInvite)
	server.OnAck(func(req *sip.Request, tx sip.ServerTransaction) {
		if err := e.dialogs.ReadAck(req, tx); err != nil {
			e.log.Debug("telephony-latency: ack outside a dialog", "err", err)
		}
	})
	server.OnBye(func(req *sip.Request, tx sip.ServerTransaction) {
		if err := e.dialogs.ReadBye(req, tx); err != nil {
			e.respond(req, tx, sip.StatusCallTransactionDoesNotExists, "Call/Transaction Does Not Exist")
		}
	})
	server.OnOptions(func(req *sip.Request, tx sip.ServerTransaction) {
		e.respond(req, tx, sip.StatusOK, "OK")
	})

	// The socket is bound here rather than by ListenAndServe, so a port already taken is an
	// error now and closing it is ours to do.
	conn, err := net.ListenPacket("udp4", listen)
	if err != nil {
		return nil, fmt.Errorf("sip listen on %s: %w", listen, err)
	}
	e.conn = conn
	go func() {
		if err := server.ServeUDP(conn); err != nil && !errors.Is(err, net.ErrClosed) {
			e.log.Error("telephony-latency: sip server stopped", "err", err)
		}
	}()
	return e, nil
}

// Expect reserves the leg that will arrive for user and returns where it will be handed
// over. It must be called before the call is placed, so a carrier that answers quickly
// does not find nobody waiting.
func (e *Endpoint) Expect(user string) <-chan *Leg {
	arrived := make(chan *Leg, 1)
	e.mu.Lock()
	defer e.mu.Unlock()
	e.expected[legKey(user)] = arrived
	return arrived
}

// Forget drops a reservation that was never claimed.
func (e *Endpoint) Forget(user string) {
	e.mu.Lock()
	defer e.mu.Unlock()
	delete(e.expected, legKey(user))
}

// Close stops answering.
func (e *Endpoint) Close() error {
	return errors.Join(e.conn.Close(), e.server.Close(), e.ua.Close())
}

func (e *Endpoint) onInvite(req *sip.Request, tx sip.ServerTransaction) {
	key := legKey(req.Recipient.User)
	e.mu.Lock()
	arrived, ok := e.expected[key]
	delete(e.expected, key)
	e.mu.Unlock()
	if !ok {
		e.log.Warn("telephony-latency: a call nobody is waiting for", "user", req.Recipient.User, "from", req.Source())
		e.respond(req, tx, sip.StatusNotFound, "Not Found")
		return
	}

	offer, err := sipbridge.ParseMedia(req.Body())
	if err != nil || !offersPCMU(offer) {
		e.log.Warn("telephony-latency: an offer without PCMU", "user", req.Recipient.User, "err", err)
		e.respond(req, tx, sip.StatusNotAcceptableHere, "Not Acceptable Here")
		return
	}
	remoteAddr, err := netip.ParseAddr(offer.Addr)
	if err != nil {
		e.log.Warn("telephony-latency: media address is not an ip", "addr", offer.Addr)
		e.respond(req, tx, sip.StatusNotAcceptableHere, "Not Acceptable Here")
		return
	}
	signal, err := netip.ParseAddrPort(req.Source())
	if err != nil {
		e.log.Warn("telephony-latency: unreadable source", "source", req.Source(), "err", err)
	}

	conn, port, err := e.ports.listen()
	if err != nil {
		e.log.Error("telephony-latency: no rtp port", "err", err)
		e.respond(req, tx, sip.StatusServiceUnavailable, "Service Unavailable")
		return
	}
	dialog, err := e.dialogs.ReadInvite(req, tx)
	if err != nil {
		_ = conn.Close()
		e.log.Error("telephony-latency: read invite", "err", err)
		return
	}
	if err := dialog.RespondSDP(answerSDP(e.host, port)); err != nil {
		_ = conn.Close()
		dialog.Close()
		e.log.Error("telephony-latency: answer", "err", err)
		return
	}

	leg := &Leg{
		Signal:   signal,
		Media:    NewMedia(dialog.Context(), conn, netip.AddrPortFrom(remoteAddr, uint16(offer.Port)), e.log),
		Answered: time.Now(),
		dialog:   dialog,
	}
	go func() {
		<-dialog.Context().Done()
		_ = leg.Media.Close()
		dialog.Close()
	}()
	arrived <- leg
}

func (e *Endpoint) respond(req *sip.Request, tx sip.ServerTransaction, status int, reason string) {
	if err := tx.Respond(sip.NewResponseFromRequest(req, status, reason, nil)); err != nil {
		e.log.Debug("telephony-latency: respond", "status", status, "err", err)
	}
}

// legKey is what a leg is filed under. A number is filed by its last ten digits, because
// carriers disagree on whether the user part of an address carries the +1.
func legKey(user string) string {
	digits := strings.TrimPrefix(user, "+")
	for _, r := range digits {
		if r < '0' || r > '9' {
			return user
		}
	}
	if len(digits) > 10 {
		return digits[len(digits)-10:]
	}
	return digits
}

func offersPCMU(offer sipbridge.Media) bool {
	for _, codec := range offer.Codecs {
		if codec.Name == "PCMU" {
			return true
		}
	}
	return false
}

// answerSDP takes G.711 μ-law and nothing else, which every US carrier offers and which
// leaves no transcoding on the bench's side of the measurement.
func answerSDP(host netip.Addr, port int) []byte {
	session := time.Now().UnixNano()
	return fmt.Appendf(nil, "v=0\r\n"+
		"o=bench %d %d IN IP4 %s\r\n"+
		"s=telephony-latency\r\n"+
		"c=IN IP4 %s\r\n"+
		"t=0 0\r\n"+
		"m=audio %d RTP/AVP %d\r\n"+
		"a=rtpmap:%d PCMU/8000\r\n"+
		"a=ptime:20\r\n"+
		"a=sendrecv\r\n",
		session, session, host, host, port, payloadPCMU, payloadPCMU)
}

// ports hands out RTP sockets from a range a firewall can be opened for.
type ports struct {
	mu        sync.Mutex
	low, high int
	next      int
}

func newPorts(low, high int) (*ports, error) {
	if low <= 0 || high < low || high > 65535 {
		return nil, fmt.Errorf("rtp port range %d-%d is not a range of ports", low, high)
	}
	return &ports{low: low, high: high, next: low}, nil
}

// listen binds the next free port in the range.
func (p *ports) listen() (*net.UDPConn, int, error) {
	p.mu.Lock()
	defer p.mu.Unlock()
	for range p.high - p.low + 1 {
		port := p.next
		p.next++
		if p.next > p.high {
			p.next = p.low
		}
		conn, err := net.ListenUDP("udp4", &net.UDPAddr{Port: port})
		if err == nil {
			return conn, port, nil
		}
	}
	return nil, 0, fmt.Errorf("every rtp port from %d to %d is in use", p.low, p.high)
}
