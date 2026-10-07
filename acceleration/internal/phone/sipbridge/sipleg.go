package sipbridge

import (
	"context"
	"errors"
	"fmt"
	"log/slog"
	"strconv"
	"strings"
	"sync"

	"github.com/emiago/sipgo"
	"github.com/emiago/sipgo/sip"
	"github.com/icholy/digest"

	"github.com/GetStream/Vision-Agents/acceleration/internal/stack"
)

// sipLeg is a leg on a real SIP dialog. We are the UAC on both legs.
type sipLeg struct {
	side      side
	client    *sipgo.Client
	ua        *sipgo.DialogUA
	target    sip.Uri // Request-URI of the initial INVITE
	from, to  sip.Uri
	transport string
	// reinviteHost, when set, replaces the host of the remote target in re-INVITEs. Only the
	// host changes; the rest of the Contact is sent back unchanged, as RFC 3261 asks. ACK, BYE
	// and OPTIONS go to the Contact as it is.
	reinviteHost string
	username     string
	password     string
	log          *slog.Logger
	// onDialog registers the leg's Call-ID, so requests the other side sends in this
	// dialog find it. It runs before the answer arrives.
	onDialog func(callID string)

	// mu serialises in-dialog requests: sipgo's CSeq update is not atomic, and an
	// OPTIONS between a re-INVITE's 2xx and its ACK would corrupt the ACK's CSeq.
	mu  sync.Mutex
	dlg *sipgo.DialogClientSession
	// answer is the 2xx to the initial INVITE, nil until the leg is answered. It is kept
	// here, under mu, because sipgo writes dlg.InviteResponse without a lock while
	// WaitAnswer runs, and handlers on other goroutines may reach this leg meanwhile.
	answer *sip.Response
	// lastSDP is the last SDP we sent this leg, repeated when it refreshes the session
	// with a re-INVITE that carries no offer.
	lastSDP []byte
}

func (l *sipLeg) Invite(ctx context.Context, body []byte) ([]byte, error) {
	req := sip.NewRequest(sip.INVITE, l.target)
	from := &sip.FromHeader{Address: l.from, Params: sip.NewParams()}
	from.Params.Add("tag", sip.GenerateTagN(16))
	req.AppendHeader(from)
	req.AppendHeader(&sip.ToHeader{Address: l.to, Params: sip.NewParams()})
	contact := &sip.ContactHeader{Address: sip.Uri{User: "sipbridge", UriParams: sip.NewParams()}}
	contact.Address.UriParams.Add("transport", l.transport)
	req.AppendHeader(contact)
	if body != nil {
		req.SetBody(body)
		req.AppendHeader(sip.NewHeader("Content-Type", "application/sdp"))
	}
	req.SetTransport(strings.ToUpper(l.transport))

	dlg, err := l.ua.WriteInvite(ctx, req)
	if err != nil {
		return nil, stack.Wrap(err)
	}
	// WriteInvite has now filled the Contact with this connection's address; later
	// in-dialog requests take it from the UA. Handlers on other goroutines read both
	// fields, so they are set under mu, which is not held across WaitAnswer.
	callID := req.CallID().Value()
	l.mu.Lock()
	l.dlg = dlg
	l.ua.ContactHDR = *dlg.InviteRequest.Contact()
	if body != nil {
		l.lastSDP = body
	}
	l.mu.Unlock()
	// Set before onDialog, so handlers that find the leg through the registry see it.
	l.log = l.log.With("call_id", callID)
	l.onDialog(callID)

	err = dlg.WaitAnswer(ctx, sipgo.AnswerOptions{
		Username: l.username,
		Password: l.password,
		OnResponse: func(res *sip.Response) error {
			l.log.Info("INVITE response", "status", res.StatusCode, "reason", res.Reason)
			// WaitAnswer resends the same INVITE with credentials right after this hook,
			// so the retry already carries the address the other side can reach us on.
			if res.StatusCode == sip.StatusUnauthorized || res.StatusCode == sip.StatusProxyAuthRequired {
				natContact(dlg.InviteRequest.Contact(), res)
			}
			return nil
		},
	})
	l.mu.Lock()
	l.ua.ContactHDR = *dlg.InviteRequest.Contact()
	if err == nil {
		l.answer = dlg.InviteResponse
	}
	l.mu.Unlock()
	if err != nil {
		if ack, answered := ackForLateAnswer(body, dlg.InviteResponse); answered {
			l.hangUpLateAnswer(ctx, dlg.InviteResponse, ack)
		}
		return nil, stack.Wrap(err)
	}
	return dlg.InviteResponse.Body(), nil
}

// ackForLateAnswer says whether an INVITE that failed was answered all the same, and with
// what to ACK it. A 2xx that crosses our CANCEL is kept by sipgo as the dialog's response
// while WaitAnswer still returns an error, so the callee is left answered with nobody on
// the line unless we ACK and BYE it. offer is what our INVITE carried: without one the 2xx
// holds the offer, and the ACK must carry an answer, which rejects the audio.
func ackForLateAnswer(offer []byte, res *sip.Response) (ack []byte, answered bool) {
	if res == nil || !res.IsSuccess() {
		return nil, false
	}
	if offer == nil {
		return rejectingAnswer(res.Body()), true
	}
	return nil, true
}

// hangUpLateAnswer ends a dialog answered after we gave up on it. ctx is usually what was
// cancelled, so the requests go out on a cleanup context.
func (l *sipLeg) hangUpLateAnswer(ctx context.Context, res *sip.Response, ack []byte) {
	l.log.Warn("INVITE answered after the call was given up, hanging up", "status", res.StatusCode)
	l.mu.Lock()
	l.answer = res
	l.mu.Unlock()
	cctx, cancel := cleanupContext(ctx)
	defer cancel()
	if err := l.Ack(cctx, ack); err != nil {
		l.log.Warn("cleanup ACK failed", "error", err)
	}
	if err := l.Bye(cctx); err != nil {
		l.log.Warn("cleanup BYE failed", "error", err)
	}
}

// natContact points contact at the address the server saw our request come from, taken
// from the received and rport it added to our Via. Behind NAT our own address is private,
// and the other side only sends its requests back over our connection when Contact
// matches that connection's address exactly.
func natContact(contact *sip.ContactHeader, res *sip.Response) {
	via := res.Via()
	if contact == nil || via == nil {
		return
	}
	if host, ok := via.Params.Get("received"); ok && host != "" {
		contact.Address.Host = host
	}
	if p, ok := via.Params.Get("rport"); ok && p != "" {
		if port, err := strconv.Atoi(p); err == nil {
			contact.Address.Port = port
		}
	}
}

func (l *sipLeg) Ack(ctx context.Context, body []byte) error {
	l.mu.Lock()
	defer l.mu.Unlock()
	if err := l.dlg.WriteAck(ctx, l.ack(body)); err != nil {
		return stack.Wrap(err)
	}
	if body != nil {
		l.lastSDP = body
	}
	return nil
}

func (l *sipLeg) Reinvite(ctx context.Context, body []byte) ([]byte, error) {
	l.mu.Lock()
	defer l.mu.Unlock()
	req := l.inDialog(sip.INVITE, "application/sdp", body)
	res, err := l.dlg.Do(ctx, req)
	if err != nil {
		return nil, stack.Wrap(err)
	}
	if (res.StatusCode == sip.StatusUnauthorized || res.StatusCode == sip.StatusProxyAuthRequired) && l.password != "" {
		if err := authorize(req, res, l.username, l.password); err != nil {
			return nil, err
		}
		req.RemoveHeader("Via")
		for req.RemoveHeader("Route") {
		}
		if res, err = l.dlg.Do(ctx, req); err != nil {
			return nil, stack.Wrap(err)
		}
	}
	if !res.IsSuccess() {
		return nil, stack.Wrap(fmt.Errorf("re-INVITE: %d %s", res.StatusCode, res.Reason))
	}
	// A re-INVITE's 2xx is ACKed like the first one, but outside the initial INVITE
	// transaction, so WriteAck does not fit.
	if err := l.dlg.WriteRequest(newACK(req, res, nil)); err != nil {
		return nil, stack.Wrap(fmt.Errorf("re-INVITE ack: %w", err))
	}
	l.lastSDP = body
	return res.Body(), nil
}

func (l *sipLeg) Info(ctx context.Context, contentType string, body []byte) error {
	l.mu.Lock()
	defer l.mu.Unlock()
	res, err := l.dlg.Do(ctx, l.inDialog(sip.INFO, contentType, body))
	if err != nil {
		return stack.Wrap(err)
	}
	if !res.IsSuccess() {
		return stack.Wrap(&statusError{method: sip.INFO, code: res.StatusCode, reason: res.Reason})
	}
	return nil
}

// statusError is a final non-2xx answer to a request we sent.
type statusError struct {
	method sip.RequestMethod
	code   int
	reason string
}

func (e *statusError) Error() string {
	return fmt.Sprintf("%s: %d %s", e.method, e.code, e.reason)
}

func (l *sipLeg) Bye(ctx context.Context) error {
	l.mu.Lock()
	defer l.mu.Unlock()
	if l.answer == nil {
		return nil
	}
	// WriteBye rather than Bye: Bye copies Route from the initial INVITE, which carries
	// the route set when a 401 with Record-Route was retried, and WriteBye adds it again.
	return stack.Wrap(l.dlg.WriteBye(ctx, l.inDialog(sip.BYE, "", nil)))
}

// keepalive sends one in-dialog OPTIONS. Any response means the connection is alive.
func (l *sipLeg) keepalive(ctx context.Context) error {
	l.mu.Lock()
	defer l.mu.Unlock()
	res, err := l.dlg.Do(ctx, l.inDialog(sip.OPTIONS, "", nil))
	if err != nil {
		return stack.Wrap(err)
	}
	if res.StatusCode == sip.StatusCallTransactionDoesNotExists {
		return stack.Wrap(fmt.Errorf("keep-alive: %d %s", res.StatusCode, res.Reason))
	}
	return nil
}

func (l *sipLeg) readBye(req *sip.Request, tx sip.ServerTransaction) error {
	return stack.Wrap(l.dlg.ReadBye(req, tx))
}

func (l *sipLeg) setLastSDP(body []byte) {
	l.mu.Lock()
	defer l.mu.Unlock()
	l.lastSDP = body
}

func (l *sipLeg) getLastSDP() []byte {
	l.mu.Lock()
	defer l.mu.Unlock()
	return l.lastSDP
}

// State is a snapshot of the leg's dialog. It is empty before the leg is answered.
func (l *sipLeg) State() LegState {
	l.mu.Lock()
	defer l.mu.Unlock()
	if l.answer == nil {
		return LegState{}
	}
	return legState(l.dlg.InviteRequest, l.answer, l.dlg.CSEQ(), l.lastSDP)
}

func (l *sipLeg) contact() *sip.ContactHeader {
	l.mu.Lock()
	defer l.mu.Unlock()
	return sip.HeaderClone(&l.ua.ContactHDR).(*sip.ContactHeader)
}

// inDialog builds a request in the dialog. The dialog fills in the dialog headers when it
// sends it.
func (l *sipLeg) inDialog(method sip.RequestMethod, contentType string, body []byte) *sip.Request {
	target := l.dlg.InviteRequest.Recipient
	if c := l.answer.Contact(); c != nil {
		target = c.Address
	}
	target = *target.Clone()
	if method == sip.INVITE && l.reinviteHost != "" {
		target.Host = l.reinviteHost
	}
	req := sip.NewRequest(method, target)
	if body != nil {
		req.SetBody(body)
		req.AppendHeader(sip.NewHeader("Content-Type", contentType))
	}
	return req
}

// ack builds the ACK for the initial 2xx.
func (l *sipLeg) ack(body []byte) *sip.Request {
	return newACK(l.dlg.InviteRequest, l.answer, body)
}

// newACK builds the ACK for a 2xx, like sipgo's unexported newAckRequestUAC, but with a
// body: in flow A the ACK carries our SDP answer. It has no Route headers: the INVITE may
// already carry the route set, and WriteAck and WriteRequest add it from the dialog.
func newACK(invite *sip.Request, res *sip.Response, body []byte) *sip.Request {
	recipient := invite.Recipient
	if c := res.Contact(); c != nil {
		recipient = c.Address
	}
	ack := sip.NewRequest(sip.ACK, *recipient.Clone())
	ack.SipVersion = invite.SipVersion
	if h := invite.From(); h != nil {
		ack.AppendHeader(sip.HeaderClone(h))
	}
	if h := res.To(); h != nil {
		ack.AppendHeader(sip.HeaderClone(h))
	}
	if h := invite.CallID(); h != nil {
		ack.AppendHeader(sip.HeaderClone(h))
	}
	if h := invite.CSeq(); h != nil {
		cseq := sip.HeaderClone(h).(*sip.CSeqHeader)
		cseq.MethodName = sip.ACK
		ack.AppendHeader(cseq)
	}
	maxForwards := sip.MaxForwardsHeader(70)
	ack.AppendHeader(&maxForwards)
	if h := invite.Contact(); h != nil {
		ack.AppendHeader(sip.HeaderClone(h))
	}
	if body != nil {
		ack.AppendHeader(sip.NewHeader("Content-Type", "application/sdp"))
	}
	ack.SetBody(body)
	ack.SetTransport(invite.Transport())
	ack.SetSource(invite.Source())
	ack.Laddr = invite.Laddr
	return ack
}

// authorize answers a 401 or 407 on req, for requests WaitAnswer does not cover.
func authorize(req *sip.Request, res *sip.Response, username, password string) error {
	challengeName, credentialsName := "WWW-Authenticate", "Authorization"
	if res.StatusCode == sip.StatusProxyAuthRequired {
		challengeName, credentialsName = "Proxy-Authenticate", "Proxy-Authorization"
	}
	h := res.GetHeader(challengeName)
	if h == nil {
		return stack.Wrap(fmt.Errorf("%d without a %s header", res.StatusCode, challengeName))
	}
	chal, err := digest.ParseChallenge(h.Value())
	if err != nil {
		return stack.Wrap(fmt.Errorf("parse %s: %w", challengeName, err))
	}
	chal.Algorithm = strings.ToUpper(chal.Algorithm)
	cred, err := digest.Digest(chal, digest.Options{
		Method:   req.Method.String(),
		URI:      req.Recipient.Addr(),
		Username: username,
		Password: password,
	})
	if err != nil {
		return stack.Wrap(errors.Join(errors.New("build digest"), err))
	}
	req.RemoveHeader(credentialsName)
	req.AppendHeader(sip.NewHeader(credentialsName, cred.String()))
	return nil
}
