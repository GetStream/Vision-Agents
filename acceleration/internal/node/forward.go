package node

import (
	"bytes"
	"context"
	"io"
	"log/slog"
	"net"
	"net/http"
	"strconv"
	"strings"
	"sync"

	"golang.org/x/net/http2"
	"golang.org/x/net/http2/h2c"
	"google.golang.org/grpc"
	"google.golang.org/grpc/codes"
	"google.golang.org/grpc/credentials/insecure"
	"google.golang.org/grpc/status"
)

// ForwardedHeader marks a request a peer handed over, so the node it arrives at answers
// it rather than looking for somewhere else to send it.
const ForwardedHeader = "X-Acceleration-Forwarded"

// Serve returns a handler that answers this deployment's own requests and the Forward
// calls its peers make, on one port.
//
// One port rather than two because a second listener is a second thing to open between
// nodes, and the only traffic here is between a deployment's own processes. gRPC wants
// HTTP/2, which a peer speaks without TLS by prior knowledge, so h2c sits outside: an
// HTTP/1.1 request passes through it untouched, sockets included.
func Serve(api http.Handler) http.Handler {
	server := grpc.NewServer()
	RegisterNodeServer(server, &forwarded{api: api})

	return h2c.NewHandler(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if r.ProtoMajor == 2 && strings.HasPrefix(r.Header.Get("Content-Type"), "application/grpc") {
			server.ServeHTTP(w, r)
			return
		}
		api.ServeHTTP(w, r)
	}), &http2.Server{})
}

// Address is where this node's peers reach it, given where it listens.
//
// The port is the one being listened on, and the host is whichever of this machine's
// addresses reaches the outside: in a pod that is the address its peers use, and on a
// laptop it is the one on the network. Nothing is dialled -- a UDP socket is given a
// destination and asked which address it would leave from.
func Address(listen string) (string, error) {
	_, port, err := net.SplitHostPort(listen)
	if err != nil {
		return "", err
	}
	// Any routable address will do; this one is never sent to.
	outbound, err := net.Dial("udp", "192.0.2.1:9")
	if err != nil {
		return "", err
	}
	defer outbound.Close()

	host, _, err := net.SplitHostPort(outbound.LocalAddr().String())
	if err != nil {
		return "", err
	}
	return net.JoinHostPort(host, port), nil
}

// Forwarder hands requests to this deployment's other nodes, keeping one connection to
// each for as long as the process runs.
type Forwarder struct {
	logger *slog.Logger

	mu    sync.Mutex
	peers map[string]*grpc.ClientConn
}

// NewForwarder returns a Forwarder that has not connected to anything yet.
func NewForwarder(logger *slog.Logger) *Forwarder {
	if logger == nil {
		logger = slog.Default()
	}
	return &Forwarder{logger: logger, peers: map[string]*grpc.ClientConn{}}
}

// Forward hands a request to the node at an address and writes back what it answered.
func (f *Forwarder) Forward(ctx context.Context, address string, w http.ResponseWriter, r *http.Request) error {
	body, err := io.ReadAll(r.Body)
	if err != nil {
		return err
	}
	answer, err := f.Ask(ctx, address, r, body)
	if err != nil {
		return err
	}

	for _, header := range answer.GetHeaders() {
		w.Header()[http.CanonicalHeaderKey(header.GetName())] = header.GetValues()
	}
	// The length is set because the answer is whole and because middleware here would
	// otherwise take the body for one of its own and stamp its timing into it a second
	// time. The node that answered has already said how long it took.
	w.Header().Set("Content-Length", strconv.Itoa(len(answer.GetBody())))
	w.WriteHeader(int(answer.GetStatus()))
	if _, err := w.Write(answer.GetBody()); err != nil {
		// The answer is already going out, so a failure here is the caller having left
		// rather than anything the caller can be told about. Reporting it would have the
		// caller told twice.
		f.logger.Debug("could not write back what another node answered",
			"peer", address, "error", err)
	}

	return nil
}

// Ask hands a request over and returns the answer without writing it, which is what the
// callers that took the body apart before they knew where it belonged need.
func (f *Forwarder) Ask(
	ctx context.Context,
	address string,
	r *http.Request,
	body []byte,
) (*ForwardResponse, error) {
	peer, err := f.peer(address)
	if err != nil {
		return nil, err
	}

	headers := headersOf(r.Header)
	// Carried as a header because it is one, and because the request is rebuilt on the
	// other side from nothing but these: a handler reading the host it was reached at
	// would otherwise see none.
	if r.Host != "" {
		headers = append(headers, &Header{Name: "Host", Values: []string{r.Host}})
	}

	return NewNodeClient(peer).Forward(ctx, &ForwardRequest{
		Method:     r.Method,
		Url:        r.URL.RequestURI(),
		Headers:    headers,
		Body:       body,
		RemoteAddr: r.RemoteAddr,
	})
}

// peer returns the connection to a node, opening one the first time it is asked for.
// Nothing is dialled here: a gRPC client connects when it is first used and reconnects
// on its own, so a peer that is down costs the request that needs it rather than this.
func (f *Forwarder) peer(address string) (*grpc.ClientConn, error) {
	f.mu.Lock()
	defer f.mu.Unlock()

	if existing, ok := f.peers[address]; ok {
		return existing, nil
	}
	// Insecure because both ends are this deployment's own processes on its own network,
	// which is also what reaching them without TLS on the API port depends on.
	peer, err := grpc.NewClient(address, grpc.WithTransportCredentials(insecure.NewCredentials()))
	if err != nil {
		return nil, err
	}
	f.peers[address] = peer

	return peer, nil
}

// forwarded answers what a peer hands over by running it through this node's own handler.
type forwarded struct {
	UnimplementedNodeServer
	api http.Handler
}

func (f *forwarded) Forward(ctx context.Context, request *ForwardRequest) (*ForwardResponse, error) {
	replayed, err := http.NewRequestWithContext(
		ctx, request.GetMethod(), request.GetUrl(), bytes.NewReader(request.GetBody()))
	if err != nil {
		return nil, status.Error(codes.InvalidArgument, err.Error())
	}
	for _, header := range request.GetHeaders() {
		replayed.Header[http.CanonicalHeaderKey(header.GetName())] = header.GetValues()
	}
	if host := replayed.Header.Get("Host"); host != "" {
		replayed.Host = host
		replayed.Header.Del("Host")
	}
	replayed.Header.Set(ForwardedHeader, "1")
	replayed.RemoteAddr = request.GetRemoteAddr()
	replayed.ContentLength = int64(len(request.GetBody()))

	answered := &answer{headers: http.Header{}}
	f.api.ServeHTTP(answered, replayed)

	return &ForwardResponse{
		Status:  int32(answered.status),
		Headers: headersOf(answered.headers),
		Body:    answered.body.Bytes(),
	}, nil
}

// answer collects what a handler answered a forwarded request with. net/http has one of
// these in httptest, which is not for a server to import.
type answer struct {
	headers http.Header
	body    bytes.Buffer
	status  int
}

func (a *answer) Header() http.Header { return a.headers }

func (a *answer) WriteHeader(status int) {
	if a.status == 0 {
		a.status = status
	}
}

func (a *answer) Write(body []byte) (int, error) {
	a.WriteHeader(http.StatusOK)
	return a.body.Write(body)
}

// headersOf renders headers for the wire, which protobuf has no map of lists for.
func headersOf(headers http.Header) []*Header {
	carried := make([]*Header, 0, len(headers))
	for name, values := range headers {
		carried = append(carried, &Header{Name: name, Values: values})
	}
	return carried
}
