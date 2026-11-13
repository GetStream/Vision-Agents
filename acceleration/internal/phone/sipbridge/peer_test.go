package sipbridge

import (
	"context"
	"net"
	"strconv"
	"sync"
	"testing"
	"time"

	"github.com/emiago/sipgo"
	"github.com/emiago/sipgo/sip"
	"github.com/stretchr/testify/require"
)

type peerOptions struct {
	// reinviteStatus answers an INVITE; 0 means 200.
	reinviteStatus int
	// challengeOnce answers the first INVITE with a 401.
	challengeOnce bool
}

// peer is the far end of a dialog on loopback: a trunk or Stream after the bridge that held
// the call is gone. It records every request it gets.
type peer struct {
	t      *testing.T
	port   int
	sdp    []byte
	client *sipgo.Client

	mu         sync.Mutex
	reqs       []*sip.Request
	challenged bool
}

func startPeer(t *testing.T, opts peerOptions) *peer {
	t.Helper()
	ua, err := sipgo.NewUA()
	require.NoError(t, err)
	t.Cleanup(func() { _ = ua.Close() })
	srv, err := sipgo.NewServer(ua)
	require.NoError(t, err)
	client, err := sipgo.NewClient(ua)
	require.NoError(t, err)
	listener, err := net.Listen("tcp", "127.0.0.1:0")
	require.NoError(t, err)
	t.Cleanup(func() { _ = listener.Close() })
	p := &peer{t: t, port: listener.Addr().(*net.TCPAddr).Port, client: client}
	p.sdp = []byte("v=0 peer " + strconv.Itoa(p.port) + "\r\n")
	if opts.reinviteStatus == 0 {
		opts.reinviteStatus = sip.StatusOK
	}

	srv.OnInvite(func(req *sip.Request, tx sip.ServerTransaction) {
		p.record(req)
		if opts.challengeOnce && p.challenge() {
			res := sip.NewResponseFromRequest(req, sip.StatusUnauthorized, "Unauthorized", nil)
			res.AppendHeader(sip.NewHeader("WWW-Authenticate", `Digest realm="x",nonce="n"`))
			_ = tx.Respond(res)
			return
		}
		if opts.reinviteStatus != sip.StatusOK {
			_ = tx.Respond(sip.NewResponseFromRequest(req, opts.reinviteStatus, "Refused", nil))
			return
		}
		res := sip.NewResponseFromRequest(req, sip.StatusOK, "OK", p.sdp)
		contact := &sip.ContactHeader{Address: sip.Uri{User: "peer", Host: "127.0.0.1", Port: p.port, UriParams: sip.NewParams()}}
		contact.Address.UriParams.Add("transport", "tcp")
		res.AppendHeader(contact)
		res.AppendHeader(sip.NewHeader("Content-Type", "application/sdp"))
		_ = tx.Respond(res)
	})
	srv.OnAck(func(req *sip.Request, _ sip.ServerTransaction) { p.record(req) })
	ok := func(req *sip.Request, tx sip.ServerTransaction) {
		p.record(req)
		_ = tx.Respond(sip.NewResponseFromRequest(req, sip.StatusOK, "OK", nil))
	}
	srv.OnBye(ok)
	srv.OnOptions(ok)
	go func() { _ = srv.ServeTCP(listener) }()
	return p
}

func (p *peer) record(req *sip.Request) {
	p.mu.Lock()
	defer p.mu.Unlock()
	p.reqs = append(p.reqs, req)
}

// challenge reports whether this is the first INVITE, the one to challenge.
func (p *peer) challenge() bool {
	p.mu.Lock()
	defer p.mu.Unlock()
	first := !p.challenged
	p.challenged = true
	return first
}

// requests is every request with method in the dialog callID, in the order they came.
func (p *peer) requests(method sip.RequestMethod, callID string) []*sip.Request {
	p.mu.Lock()
	defer p.mu.Unlock()
	var reqs []*sip.Request
	for _, req := range p.reqs {
		if req.Method == method && req.CallID().Value() == callID {
			reqs = append(reqs, req)
		}
	}
	return reqs
}

// methods is the method of every request in the dialog callID.
func (p *peer) methods(callID string) []sip.RequestMethod {
	p.mu.Lock()
	defer p.mu.Unlock()
	var methods []sip.RequestMethod
	for _, req := range p.reqs {
		if req.CallID().Value() == callID {
			methods = append(methods, req.Method)
		}
	}
	return methods
}

// sendBye hangs up the dialog callID, sending BYE back over the connection its last INVITE
// came in on, as a trunk does to a bridge with no listening port.
func (p *peer) sendBye(callID string) {
	p.t.Helper()
	invites := p.requests(sip.INVITE, callID)
	require.NotEmpty(p.t, invites)
	last := invites[len(invites)-1]
	bye := sip.NewRequest(sip.BYE, *last.Contact().Address.Clone())
	bye.AppendHeader(&sip.FromHeader{Address: last.To().Address, Params: last.To().Params.Clone()})
	bye.AppendHeader(&sip.ToHeader{Address: last.From().Address, Params: last.From().Params.Clone()})
	bye.AppendHeader(sip.HeaderClone(last.CallID()))
	bye.AppendHeader(&sip.CSeqHeader{SeqNo: 900, MethodName: sip.BYE})
	bye.SetTransport("TCP")
	bye.SetDestination(last.Source())
	ctx, cancel := context.WithTimeout(context.Background(), 5*time.Second)
	defer cancel()
	res, err := p.client.Do(ctx, bye)
	require.NoError(p.t, err)
	require.Equal(p.t, sip.StatusOK, res.StatusCode)
}
