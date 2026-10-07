package sipbridge

import (
	"context"
	"log/slog"
	"net"
	"net/netip"
	"strconv"
	"sync"
	"testing"
	"time"

	"github.com/emiago/sipgo"
	"github.com/emiago/sipgo/sip"
	"github.com/stretchr/testify/require"
)

// answers resolves each name to its lists in turn, repeating the last, and counts lookups.
type answers struct {
	mu      sync.Mutex
	byHost  map[string][][]string
	lookups int
}

func (a *answers) LookupNetIP(_ context.Context, _, host string) ([]netip.Addr, error) {
	a.mu.Lock()
	defer a.mu.Unlock()
	lists := a.byHost[host]
	if len(lists) == 0 {
		return nil, &net.DNSError{Err: "no such host", Name: host, IsNotFound: true}
	}
	list := lists[min(a.lookups, len(lists)-1)]
	a.lookups++
	ips := make([]netip.Addr, 0, len(list))
	for _, s := range list {
		ips = append(ips, netip.MustParseAddr(s))
	}
	return ips, nil
}

func requestTo(host string, port int, transport string) *sip.Request {
	req := sip.NewRequest(sip.INVITE, sip.Uri{User: "+15550002222", Host: host, Port: port})
	req.SetTransport(transport)
	return req
}

func TestGuardRefusesPrivateAddresses(t *testing.T) {
	g := guard{}.withDefaults()

	for _, host := range []string{"10.0.0.1", "127.0.0.1", "169.254.169.254"} {
		err := g.check(t.Context(), requestTo(host, 5060, "TCP"))

		require.ErrorContains(t, err, host+" is not a public address", host)
	}
}

func TestGuardLetsAPublicAddressThrough(t *testing.T) {
	g := guard{}.withDefaults()
	req := requestTo("8.8.8.8", 5060, "TCP")

	require.NoError(t, g.check(t.Context(), req))
	require.Equal(t, "8.8.8.8:5060", req.Destination())
}

func TestGuardRefusesANameWithAnyPrivateAddress(t *testing.T) {
	g := guard{resolver: &answers{byHost: map[string][][]string{
		"trunk.example.com": {{"8.8.8.8", "10.0.0.1"}},
	}}}.withDefaults()

	err := g.check(t.Context(), requestTo("trunk.example.com", 5060, "UDP"))

	require.ErrorContains(t, err, `"trunk.example.com" resolves to 10.0.0.1, which is not a public address`)
}

// DNS rebinding: the name is public when it is checked and loopback a moment later.
func TestGuardSendsToTheAddressItChecked(t *testing.T) {
	resolver := &answers{byHost: map[string][][]string{
		"trunk.example.com": {{"8.8.8.8"}, {"127.0.0.1"}},
	}}
	g := guard{resolver: resolver}.withDefaults()
	req := requestTo("trunk.example.com", 0, "TCP")

	require.NoError(t, g.check(t.Context(), req))

	require.Equal(t, "8.8.8.8:5060", req.Destination())
	require.Equal(t, "sip:+15550002222@trunk.example.com", req.Recipient.String())
	require.Equal(t, 1, resolver.lookups)
}

func TestGuardKeepsTheNameOverTLSForTheCertificate(t *testing.T) {
	g := guard{resolver: &answers{byHost: map[string][][]string{
		"trunk.example.com": {{"8.8.8.8"}},
	}}}.withDefaults()
	req := requestTo("trunk.example.com", 5061, "TLS")

	require.NoError(t, g.check(t.Context(), req))
	require.Equal(t, "trunk.example.com:5061", req.Destination())
}

func TestDialRefusesATrunkAtAPrivateAddressBeforeCallingStream(t *testing.T) {
	for _, host := range []string{"10.0.0.1", "127.0.0.1", "169.254.169.254"} {
		cfg := validConfig()
		cfg.CustomerTrunk.Host = host

		_, err := Dial(t.Context(), cfg)

		require.ErrorContains(t, err, "sipbridge: customer trunk: "+host+" is not a public address", host)
	}
}

// sipServer answers INVITE with 200 and contact as its Contact, and records each request.
type sipServer struct {
	addr string
	mu   sync.Mutex
	reqs []*sip.Request
}

func (s *sipServer) record(req *sip.Request) {
	s.mu.Lock()
	defer s.mu.Unlock()
	s.reqs = append(s.reqs, req)
}

func (s *sipServer) methods() []sip.RequestMethod {
	s.mu.Lock()
	defer s.mu.Unlock()
	var methods []sip.RequestMethod
	for _, req := range s.reqs {
		methods = append(methods, req.Method)
	}
	return methods
}

func startSIPServer(t *testing.T, contactHost string) *sipServer {
	t.Helper()
	ua, err := sipgo.NewUA()
	require.NoError(t, err)
	t.Cleanup(func() { _ = ua.Close() })
	srv, err := sipgo.NewServer(ua)
	require.NoError(t, err)
	listener, err := net.Listen("tcp", "127.0.0.1:0")
	require.NoError(t, err)
	s := &sipServer{addr: listener.Addr().String()}
	_, portText, _ := net.SplitHostPort(s.addr)
	port, _ := strconv.Atoi(portText)
	if contactHost == "" {
		contactHost = "127.0.0.1"
	}

	srv.OnInvite(func(req *sip.Request, tx sip.ServerTransaction) {
		s.record(req)
		res := sip.NewResponseFromRequest(req, sip.StatusOK, "OK", []byte("v=0 theirs\r\n"))
		res.To().Params.Add("tag", "servertag")
		res.AppendHeader(&sip.ContactHeader{Address: sip.Uri{User: "peer", Host: contactHost, Port: port, UriParams: sip.NewParams()}})
		res.AppendHeader(sip.NewHeader("Content-Type", "application/sdp"))
		_ = tx.Respond(res)
	})
	srv.OnAck(func(req *sip.Request, _ sip.ServerTransaction) { s.record(req) })
	srv.OnBye(func(req *sip.Request, tx sip.ServerTransaction) {
		s.record(req)
		_ = tx.Respond(sip.NewResponseFromRequest(req, sip.StatusOK, "OK", nil))
	})
	go func() { _ = srv.ServeTCP(listener) }()
	t.Cleanup(func() { _ = listener.Close() })
	return s
}

// guardedLeg is a customer leg as Dial builds it, with a guard that lets only loopback through.
func guardedLeg(t *testing.T, server *sipServer) *sipLeg {
	t.Helper()
	ua, err := sipgo.NewUA()
	require.NoError(t, err)
	t.Cleanup(func() { _ = ua.Close() })
	client, err := sipgo.NewClient(ua, sipgo.WithClientNAT())
	require.NoError(t, err)
	g := guard{allow: func(ip netip.Addr) bool { return ip.IsLoopback() }}.withDefaults()
	client.TxRequester = guardedRequester{guard: g, tx: ua.TransactionLayer()}

	host, portText, _ := net.SplitHostPort(server.addr)
	port, _ := strconv.Atoi(portText)
	target := sip.Uri{User: "+15550002222", Host: host, Port: port, UriParams: sip.NewParams()}
	target.UriParams.Add("transport", "tcp")
	return &sipLeg{
		side: customerSide, client: client, ua: &sipgo.DialogUA{Client: client},
		target: target, transport: "tcp",
		from:     sip.Uri{User: "+15550001111", Host: host},
		to:       sip.Uri{User: "+15550002222", Host: host},
		log:      slog.New(slog.DiscardHandler),
		onDialog: func(string) {},
	}
}

func TestAGuardedLegCompletesACall(t *testing.T) {
	server := startSIPServer(t, "")
	l := guardedLeg(t, server)
	ctx, cancel := context.WithTimeout(t.Context(), 5*time.Second)
	defer cancel()

	answer, err := l.Invite(ctx, []byte("v=0 ours\r\n"))
	require.NoError(t, err)
	require.Equal(t, []byte("v=0 theirs\r\n"), answer)
	require.NoError(t, l.Ack(ctx, nil))
	require.NoError(t, l.Bye(ctx))

	require.Eventually(t, func() bool { return len(server.methods()) == 3 }, 2*time.Second, 10*time.Millisecond)
	// The server handles each request on its own goroutine, so ACK and BYE may be recorded
	// in either order.
	require.ElementsMatch(t, []sip.RequestMethod{sip.INVITE, sip.ACK, sip.BYE}, server.methods())
	contact := server.reqs[0].Contact()
	require.NotNil(t, contact)
	require.NotEmpty(t, contact.Address.Host, "the INVITE's Contact gets the connection's address")
	require.NotZero(t, contact.Address.Port)
}

func TestAGuardedLegRefusesAnAnswerThatPointsToAPrivateAddress(t *testing.T) {
	server := startSIPServer(t, "10.0.0.1")
	l := guardedLeg(t, server)
	ctx, cancel := context.WithTimeout(t.Context(), 5*time.Second)
	defer cancel()

	_, err := l.Invite(ctx, []byte("v=0 ours\r\n"))
	require.NoError(t, err)

	require.ErrorContains(t, l.Ack(ctx, nil), "10.0.0.1 is not a public address")
	require.Equal(t, []sip.RequestMethod{sip.INVITE}, server.methods())
}
