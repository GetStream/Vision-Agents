package sipbridge

import (
	"context"
	"crypto/tls"
	"errors"
	"fmt"
	"sync"
	"time"

	"github.com/emiago/sipgo"
	"github.com/emiago/sipgo/sip"

	"github.com/GetStream/Vision-Agents/acceleration/internal/stack"
)

// keepaliveTimeout bounds one OPTIONS round trip.
const keepaliveTimeout = 5 * time.Second

// Dial places the call and returns once both legs are up and RTP flows between the customer
// and Stream. Cancelling ctx before then cancels the call.
func Dial(ctx context.Context, cfg Config) (*Call, error) {
	cfg = cfg.WithDefaults()
	if err := cfg.Validate(); err != nil {
		return nil, err
	}
	log := cfg.Logger
	customerURI, _ := cfg.customerURI()
	streamURI, _ := cfg.streamURI()
	// Checked before Stream is called, so a trunk at a private address places no
	// call at all. Every request to the customer is checked again where it is sent.
	guard := cfg.guard.withDefaults()
	if _, err := guard.addrs(ctx, cfg.CustomerTrunk.Host); err != nil {
		return nil, stack.Wrap(fmt.Errorf("sipbridge: customer trunk: %w", err))
	}

	ua, err := sipgo.NewUA(
		sipgo.WithUserAgent("sipbridge"),
		sipgo.WithUserAgenTLSConfig(&tls.Config{MinVersion: tls.VersionTLS12}),
	)
	if err != nil {
		return nil, stack.Wrap(err)
	}
	// rport: our Via address is private, so responses must go back to where the request
	// came from.
	client, err := sipgo.NewClient(ua, sipgo.WithClientNAT())
	if err != nil {
		ua.Close()
		return nil, stack.Wrap(err)
	}
	customerClient, err := sipgo.NewClient(ua, sipgo.WithClientNAT())
	if err != nil {
		ua.Close()
		return nil, stack.Wrap(err)
	}
	customerClient.TxRequester = guardedRequester{guard: guard, tx: ua.TransactionLayer()}
	// The server is never started with ListenAndServe. It only handles requests the other
	// side sends back over the connections we opened.
	srv, err := sipgo.NewServer(ua)
	if err != nil {
		ua.Close()
		return nil, stack.Wrap(err)
	}

	reg := newRegistry()
	customer := &sipLeg{
		side: customerSide, client: customerClient, ua: &sipgo.DialogUA{Client: customerClient},
		target:    customerURI,
		from:      sip.Uri{User: cfg.Call.From, Host: cfg.CustomerTrunk.Host},
		to:        sip.Uri{User: cfg.Call.To, Host: cfg.CustomerTrunk.Host},
		transport: cfg.CustomerTrunk.Transport,
		username:  cfg.CustomerTrunk.Username, password: cfg.CustomerTrunk.Password,
		log: log.With("leg", customerSide),
	}
	// Stream matches the routing rule on the called number, which is the trunk's own
	// number, and makes the caller a participant named after From.
	stream := &sipLeg{
		side: streamSide, client: client, ua: &sipgo.DialogUA{Client: client},
		target:    streamURI,
		from:      sip.Uri{User: cfg.Call.To, Host: streamURI.Host},
		to:        sip.Uri{User: cfg.Call.From, Host: streamURI.Host},
		transport: cfg.Stream.Transport, reinviteHost: streamURI.Host,
		username: cfg.Stream.Username, password: cfg.Stream.Password,
		log: log.With("leg", streamSide),
	}
	customer.onDialog = func(id string) { reg.add(id, customer) }
	stream.onDialog = func(id string) { reg.add(id, stream) }

	call := newCall(customer, stream, log)
	handle(srv, reg, call)

	flow := flowB
	if cfg.CustomerTrunk.LateOffer {
		flow = flowA
	}
	// A BYE during setup ends the call, which must also stop the flow, so a pending
	// INVITE is cancelled instead of answered into a call nobody holds.
	flowCtx, cancelFlow := context.WithCancel(ctx)
	defer cancelFlow()
	go func() {
		select {
		case <-call.Done():
			cancelFlow()
		case <-flowCtx.Done():
		}
	}()
	if err := flow(flowCtx, cfg, customer, stream); err != nil {
		ua.Close()
		return nil, err
	}
	if err := finishSetup(ctx, call, customer, stream); err != nil {
		ua.Close()
		return nil, err
	}
	call.establish()

	for _, l := range []*sipLeg{customer, stream} {
		go keepalive(call, l, cfg.KeepaliveInterval)
	}
	go func() {
		<-call.Done()
		ua.Close()
	}()
	return call, nil
}

// finishSetup catches a call that ended while the flow was still running: the flow can
// succeed on the leg that was mid-INVITE, which then has to be hung up.
func finishSetup(ctx context.Context, call *Call, customer, stream leg) error {
	if !call.ended() {
		return nil
	}
	cctx, cancel := cleanupContext(ctx)
	defer cancel()
	err := errors.Join(customer.Bye(cctx), stream.Bye(cctx))
	return stack.Wrap(errors.Join(errors.New("call ended during setup"), err))
}

func handle(srv *sipgo.Server, reg *registry, call *Call) {
	unknown := func(req *sip.Request, tx sip.ServerTransaction) {
		_ = tx.Respond(sip.NewResponseFromRequest(req, sip.StatusCallTransactionDoesNotExists, "Call/Transaction Does Not Exist", nil))
	}

	srv.OnBye(func(req *sip.Request, tx sip.ServerTransaction) {
		l, ok := reg.lookup(req)
		if !ok {
			unknown(req, tx)
			return
		}
		if err := l.readBye(req, tx); err != nil {
			l.log.Warn("answering BYE failed", "error", err)
		}
		ctx, cancel := cleanupContext(context.Background())
		defer cancel()
		call.onBye(ctx, l.side)
	})

	srv.OnInvite(func(req *sip.Request, tx sip.ServerTransaction) {
		l, ok := reg.lookup(req)
		if !ok {
			unknown(req, tx)
			return
		}
		ctx, cancel := context.WithTimeout(context.Background(), 30*time.Second)
		defer cancel()
		forward := func(ctx context.Context, offer []byte) ([]byte, error) {
			return call.onReinvite(ctx, l.side, offer)
		}
		_ = tx.Respond(answerReinvite(ctx, l, req, forward))
	})

	// The ACK for our 2xx to a re-INVITE needs no action.
	srv.OnAck(func(*sip.Request, sip.ServerTransaction) {})

	srv.OnInfo(func(req *sip.Request, tx sip.ServerTransaction) {
		l, ok := reg.lookup(req)
		if !ok {
			unknown(req, tx)
			return
		}
		ctx, cancel := context.WithTimeout(context.Background(), keepaliveTimeout)
		defer cancel()
		contentType := ""
		if h := req.GetHeader("Content-Type"); h != nil {
			contentType = h.Value()
		}
		err := call.onInfo(ctx, l.side, contentType, req.Body())
		if err != nil {
			l.log.Warn("passing INFO on failed", "error", err)
		}
		status, reason := infoStatus(err)
		_ = tx.Respond(sip.NewResponseFromRequest(req, status, reason, nil))
	})

	srv.OnOptions(func(req *sip.Request, tx sip.ServerTransaction) {
		_ = tx.Respond(sip.NewResponseFromRequest(req, sip.StatusOK, "OK", nil))
	})
}

// answerReinvite answers a re-INVITE l received. One with an offer is passed on through
// forward. One without is a session refresh, answered here with the SDP this leg already has.
func answerReinvite(ctx context.Context, l *sipLeg, req *sip.Request, forward func(context.Context, []byte) ([]byte, error)) *sip.Response {
	ok := func(sdp []byte) *sip.Response {
		res := sip.NewResponseFromRequest(req, sip.StatusOK, "OK", sdp)
		res.AppendHeader(sip.NewHeader("Content-Type", "application/sdp"))
		res.AppendHeader(l.contact())
		return res
	}
	if len(req.Body()) == 0 {
		if last := l.getLastSDP(); last != nil {
			attrs := []any{}
			if h := req.GetHeader("Session-Expires"); h != nil {
				attrs = append(attrs, "session_expires", h.Value())
			}
			l.log.Info("re-INVITE without SDP, answering with our last SDP", attrs...)
			return ok(last)
		}
	}
	answer, err := forward(ctx, req.Body())
	if errors.Is(err, errNotEstablished) {
		// 491 asks the sender to try again shortly, by when the call is set up.
		return sip.NewResponseFromRequest(req, 491, "Request Pending", nil)
	}
	if err != nil {
		l.log.Warn("passing re-INVITE on failed", "error", err)
		return sip.NewResponseFromRequest(req, sip.StatusNotAcceptableHere, "Not Acceptable Here", nil)
	}
	l.setLastSDP(answer)
	return ok(answer)
}

// infoStatus is our answer to an INFO, given what passing it on returned. A rejection
// from the other leg goes back as it is, so the sender sees why.
func infoStatus(err error) (int, string) {
	if err == nil {
		return sip.StatusOK, "OK"
	}
	if se, ok := errors.AsType[*statusError](err); ok {
		return se.code, se.reason
	}
	return sip.StatusInternalServerError, "Server Internal Error"
}

// keepalive pings the leg until the call ends. A failed ping means the connection the other
// side would send its BYE over is gone, so the call is ended rather than left half open.
func keepalive(call *Call, l *sipLeg, interval time.Duration) {
	ticker := time.NewTicker(interval)
	defer ticker.Stop()
	for {
		select {
		case <-call.Done():
			return
		case <-ticker.C:
			ctx, cancel := context.WithTimeout(context.Background(), keepaliveTimeout)
			err := l.keepalive(ctx)
			cancel()
			if err != nil {
				cctx, ccancel := cleanupContext(context.Background())
				call.lost(cctx, l.side, err)
				ccancel()
				return
			}
		}
	}
}

// registry finds the leg a request from the other side belongs to.
type registry struct {
	mu   sync.Mutex
	legs map[string]*sipLeg
}

func newRegistry() *registry {
	return &registry{legs: map[string]*sipLeg{}}
}

func (r *registry) add(callID string, l *sipLeg) {
	r.mu.Lock()
	defer r.mu.Unlock()
	r.legs[callID] = l
}

func (r *registry) lookup(req *sip.Request) (*sipLeg, bool) {
	h := req.CallID()
	if h == nil {
		return nil, false
	}
	r.mu.Lock()
	defer r.mu.Unlock()
	l, ok := r.legs[h.Value()]
	return l, ok
}
