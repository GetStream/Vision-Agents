package sipbridge

import (
	"context"
	"errors"
	"fmt"
	"log/slog"
	"slices"
	"strings"
	"time"

	"github.com/emiago/sipgo"
	"github.com/emiago/sipgo/sip"

	"github.com/GetStream/Vision-Agents/acceleration/internal/stack"
)

// ResumeConfig is what continuing a call set up elsewhere needs besides its CallState. Each
// leg keeps the transport in its LegState; the trunks' Transport is not read.
type ResumeConfig struct {
	// CustomerTrunk gives the host the guard checks and the credentials a challenge to the
	// customer leg's re-INVITE is answered with.
	CustomerTrunk CustomerTrunk
	// Stream gives the host swapped into the Stream leg's re-INVITEs, and its credentials.
	Stream            StreamTrunk
	KeepaliveInterval time.Duration
	// CSeqJump is added to each leg's stored CSeq. The node that held the call may have
	// sent requests after the state was stored, and the peer refuses a CSeq that goes back.
	CSeqJump uint32
	// OnChange is Config.OnChange.
	OnChange func()
	Logger   *slog.Logger
	guard    guard
}

// WithDefaults returns c with every unset optional field filled in.
func (c ResumeConfig) WithDefaults() ResumeConfig {
	if c.KeepaliveInterval == 0 {
		c.KeepaliveInterval = 20 * time.Second
	}
	if c.CSeqJump == 0 {
		c.CSeqJump = 1000
	}
	if c.Logger == nil {
		c.Logger = slog.Default()
	}
	return c
}

// Validate reports every problem at once.
func (c ResumeConfig) Validate() error {
	var errs []error
	if c.CustomerTrunk.Host == "" {
		errs = append(errs, errors.New("customer_trunk.host is required"))
	}
	if c.Stream.URI == "" {
		errs = append(errs, errors.New("stream.uri is required"))
	} else if err := sip.ParseUri(c.Stream.URI, &sip.Uri{}); err != nil {
		errs = append(errs, fmt.Errorf("stream.uri: %w", err))
	}
	if c.KeepaliveInterval <= 0 {
		errs = append(errs, errors.New("keepalive_interval must be positive"))
	}
	return stack.Wrap(errors.Join(errs...))
}

// Resume continues a call whose dialogs were set up by another node, from the state that node
// stored. It sends each side a re-INVITE in its old dialog from a new connection, customer
// first, and returns the call once both answered; from then on it relays, keeps alive and
// ends like one from Dial. On failure both legs get a BYE.
func Resume(ctx context.Context, cfg ResumeConfig, state CallState) (*Call, error) {
	cfg = cfg.WithDefaults()
	if err := cfg.Validate(); err != nil {
		return nil, err
	}
	var streamURI sip.Uri
	_ = sip.ParseUri(cfg.Stream.URI, &streamURI)
	guard := cfg.guard.withDefaults()
	if _, err := guard.addrs(ctx, cfg.CustomerTrunk.Host); err != nil {
		return nil, stack.Wrap(fmt.Errorf("sipbridge: customer trunk: %w", err))
	}

	ua, customerClient, streamClient, srv, err := newUA(guard)
	if err != nil {
		return nil, err
	}
	customer, err := resumedLeg(customerSide, customerClient, state.Customer, cfg.CSeqJump, cfg.Logger)
	if err != nil {
		ua.Close()
		return nil, err
	}
	customer.username, customer.password = cfg.CustomerTrunk.Username, cfg.CustomerTrunk.Password
	stream, err := resumedLeg(streamSide, streamClient, state.Stream, cfg.CSeqJump, cfg.Logger)
	if err != nil {
		ua.Close()
		return nil, err
	}
	stream.username, stream.password = cfg.Stream.Username, cfg.Stream.Password
	stream.reinviteHost = streamURI.Host

	reg := newRegistry()
	reg.add(state.Customer.CallID, customer)
	reg.add(state.Stream.CallID, stream)
	call := newCall(customer, stream, cfg.Logger)
	call.onChange = cfg.OnChange
	call.media = state.Media
	handle(srv, reg, call)

	for _, l := range []*sipLeg{customer, stream} {
		if err := l.resume(ctx); err != nil {
			cctx, cancel := cleanupContext(ctx)
			_ = customer.Bye(cctx)
			_ = stream.Bye(cctx)
			cancel()
			ua.Close()
			return nil, err
		}
	}
	if err := finishSetup(ctx, call, customer, stream); err != nil {
		ua.Close()
		return nil, err
	}
	run(call, ua, customer, stream, cfg.KeepaliveInterval)
	return call, nil
}

// resumedLeg rebuilds a leg's confirmed dialog from its state. sipgo takes the dialog's From,
// Call-ID and CSeq from the INVITE, and its To, remote target and route set from the 2xx, so
// the two are made up to hold them.
func resumedLeg(s side, client *sipgo.Client, state LegState, cseqJump uint32, log *slog.Logger) (*sipLeg, error) {
	var err error
	from := &sip.FromHeader{Params: sip.NewParams()}
	if from.DisplayName, err = sip.ParseAddressValue(state.From, &from.Address, &from.Params); err != nil {
		return nil, stack.Wrap(fmt.Errorf("sipbridge: %s leg From: %w", s, err))
	}
	to := &sip.ToHeader{Params: sip.NewParams()}
	if to.DisplayName, err = sip.ParseAddressValue(state.To, &to.Address, &to.Params); err != nil {
		return nil, stack.Wrap(fmt.Errorf("sipbridge: %s leg To: %w", s, err))
	}
	var target sip.Uri
	if err := sip.ParseUri(state.RemoteTarget, &target); err != nil {
		return nil, stack.Wrap(fmt.Errorf("sipbridge: %s leg remote target: %w", s, err))
	}
	callID := sip.CallIDHeader(state.CallID)
	// The first request in the dialog adds one, so it carries LocalCSeq+cseqJump.
	cseq := &sip.CSeqHeader{SeqNo: state.LocalCSeq + cseqJump - 1, MethodName: sip.INVITE}

	inv := sip.NewRequest(sip.INVITE, *target.Clone())
	inv.AppendHeader(from)
	untagged := sip.HeaderClone(to).(*sip.ToHeader)
	untagged.Params = to.Params.Clone()
	untagged.Params.Remove("tag")
	inv.AppendHeader(untagged)
	inv.AppendHeader(&callID)
	inv.AppendHeader(cseq)
	inv.SetTransport(strings.ToUpper(state.Transport))

	res := sip.NewResponse(sip.StatusOK, "OK")
	res.AppendHeader(sip.HeaderClone(from))
	res.AppendHeader(to)
	res.AppendHeader(sip.HeaderClone(&callID))
	res.AppendHeader(sip.HeaderClone(cseq))
	res.AppendHeader(&sip.ContactHeader{Address: target})
	// sipgo reverses the 2xx's Record-Route into the route set, so they go in reversed.
	for _, route := range slices.Backward(state.RouteSet) {
		res.AppendHeader(sip.NewHeader("Record-Route", route))
	}

	ua := &sipgo.DialogUA{Client: client, ContactHDR: bridgeContact(client.Hostname(), 0, state.Transport)}
	dlg := &sipgo.DialogClientSession{
		Dialog: sipgo.Dialog{ID: state.CallID, InviteRequest: inv, InviteResponse: res},
		UA:     ua,
	}
	dlg.InitWithState(sip.DialogStateConfirmed)
	return &sipLeg{
		side: s, client: client, ua: ua,
		target: target, transport: state.Transport,
		log: log.With("leg", s, "call_id", state.CallID),
		dlg: dlg, answer: res,
		lastSDP: slices.Clone(state.LastSDP),
		resumed: true,
	}, nil
}

// resume moves the leg's dialog onto this node's connection: it learns the connection's
// address, then sends the last SDP again in a re-INVITE whose Contact names it, so the peer
// sends its requests in the dialog here.
func (l *sipLeg) resume(ctx context.Context) error {
	if err := l.learnContact(ctx); err != nil {
		return err
	}
	_, err := l.Reinvite(ctx, l.getLastSDP())
	return err
}

// learnContact sends one OPTIONS outside the dialog to where the re-INVITE will go, and takes
// the address the peer saw it come from, from the received and rport on our Via, as the Contact
// of later requests. Behind NAT nothing else knows it. Any response will do.
func (l *sipLeg) learnContact(ctx context.Context) error {
	l.mu.Lock()
	dest := l.inDialog(sip.INVITE, "", nil).Recipient
	if routes := l.answer.GetHeaders("Record-Route"); len(routes) > 0 {
		var route sip.Uri
		params := sip.NewParams()
		if _, err := sip.ParseAddressValue(routes[len(routes)-1].Value(), &route, &params); err != nil {
			l.mu.Unlock()
			return stack.Wrap(fmt.Errorf("sipbridge: %s leg route: %w", l.side, err))
		}
		// The proxy itself, not the dialog it routes.
		dest = sip.Uri{Scheme: route.Scheme, Host: route.Host, Port: route.Port, UriParams: sip.NewParams()}
		if t, ok := route.UriParams.Get("transport"); ok {
			dest.UriParams.Add("transport", t)
		}
	}
	l.mu.Unlock()

	req := sip.NewRequest(sip.OPTIONS, dest)
	req.AppendHeader(&sip.CSeqHeader{SeqNo: 1, MethodName: sip.OPTIONS})
	req.SetTransport(strings.ToUpper(l.transport))
	res, err := l.client.Do(ctx, req)
	if err != nil {
		return stack.Wrap(fmt.Errorf("sipbridge: %s leg OPTIONS: %w", l.side, err))
	}
	via := res.Via()
	if via == nil {
		return stack.Wrap(fmt.Errorf("sipbridge: %s leg OPTIONS: response without Via", l.side))
	}
	contact := bridgeContact(via.Host, via.Port, l.transport)
	natContact(&contact, res)
	l.mu.Lock()
	l.ua.ContactHDR = contact
	l.mu.Unlock()
	return nil
}

func bridgeContact(host string, port int, transport string) sip.ContactHeader {
	contact := sip.ContactHeader{Address: sip.Uri{User: "sipbridge", Host: host, Port: port, UriParams: sip.NewParams()}}
	contact.Address.UriParams.Add("transport", transport)
	return contact
}
