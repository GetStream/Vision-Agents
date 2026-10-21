// Package egress is the router's egress proxy for code in a sandbox: the sandbox holds a
// placeholder, and the proxy puts the connection's token in its place on the way out, only
// in the Authorization header of a request to one of the grant's hosts. The token never
// enters the sandbox. A host of the config's allowed domains gets the sandbox's bytes as they
// are.
package egress

import (
	"bytes"
	"cmp"
	"context"
	"crypto"
	"crypto/ecdsa"
	"crypto/elliptic"
	"crypto/rand"
	"crypto/tls"
	"crypto/x509"
	"crypto/x509/pkix"
	"encoding/base64"
	"errors"
	"fmt"
	"io"
	"log"
	"log/slog"
	"math/big"
	"net"
	"net/http"
	"net/http/httptrace"
	"net/http/httputil"
	"strings"
	"sync"
	"sync/atomic"
	"time"

	publicegress "github.com/GetStream/Vision-Agents/acceleration/internal/egress"
	"github.com/GetStream/Vision-Agents/acceleration/internal/sandbox"
	"github.com/GetStream/Vision-Agents/acceleration/internal/stack"
)

const (
	// handshakeTimeout bounds a sandbox's TLS handshake and each request's headers in a
	// tunnel: http.DefaultTransport's TLSHandshakeTimeout (net/http/transport.go).
	handshakeTimeout = 10 * time.Second
	// idleTimeout closes a tunnel no request has used for it: http.DefaultTransport's
	// IdleConnTimeout (net/http/transport.go).
	idleTimeout = 90 * time.Second
	// headerTimeout ends a request whose host sends no response header in time: nginx's
	// proxy_read_timeout default
	// (https://nginx.org/en/docs/http/ngx_http_proxy_module.html#proxy_read_timeout).
	headerTimeout = 60 * time.Second
	// leafValidity is how long a host's certificate is valid. A client checks it only in the
	// handshake it is minted for; an hour each side leaves room for a sandbox clock that is
	// off by that much.
	leafValidity = time.Hour
	// authenticate is the challenge a 407 must carry (RFC 9110 section 15.5.8), for the
	// Basic credential a proxy URL's userinfo sends (RFC 7617).
	authenticate = `Basic realm="sandbox egress"`
	// httpsPort is the one port a tunnel may open: HTTPS's (RFC 9110 section 4.2.2).
	httpsPort = "443"
)

// Options configure a Proxy.
type Options struct {
	Store  *Store
	Tokens Tokens
	// CACert and CAKey are the PEM certificate and key of the CA each granted host's
	// certificate is signed with: the CA the sandbox trusts.
	CACert, CAKey []byte
	// Transport sends a request on to its host. Nil is internal/egress's, which dials public
	// addresses only and verifies the host with the system's roots.
	Transport http.RoundTripper
	// Dial opens a tunnel to an allowed domain. Nil is internal/egress's DialPublic, which dials
	// public addresses only.
	Dial   func(ctx context.Context, network, address string) (net.Conn, error)
	Logger *slog.Logger
	// HandshakeTimeout, IdleTimeout and HeaderTimeout replace handshakeTimeout, idleTimeout
	// and headerTimeout when set, for tests.
	HandshakeTimeout, IdleTimeout, HeaderTimeout time.Duration
}

// Proxy is an HTTP proxy that takes CONNECT only.
type Proxy struct {
	store     *Store
	tokens    Tokens
	ca        *x509.Certificate
	caKey     crypto.Signer
	leafKey   *ecdsa.PrivateKey
	transport http.RoundTripper
	dial      func(ctx context.Context, network, address string) (net.Conn, error)
	logger    *slog.Logger

	handshakeTimeout, idleTimeout time.Duration
}

func New(options Options) (*Proxy, error) {
	pair, err := tls.X509KeyPair(options.CACert, options.CAKey)
	if err != nil {
		return nil, stack.Wrap(fmt.Errorf("egress: the proxy's CA certificate and key: %w", err))
	}
	caKey, signer := pair.PrivateKey.(crypto.Signer)
	if !pair.Leaf.IsCA || !signer {
		return nil, stack.Wrap(errors.New("egress: the proxy's CA certificate is not a CA's, or its key cannot sign"))
	}
	// One key for every host's certificate: it never leaves the process.
	leafKey, err := ecdsa.GenerateKey(elliptic.P256(), rand.Reader)
	if err != nil {
		return nil, stack.Wrap(err)
	}
	transport := options.Transport
	if transport == nil {
		transport = publicegress.NewClient(0, nil).Transport
	}
	dial := options.Dial
	if dial == nil {
		dial = publicegress.DialPublic
	}
	return &Proxy{store: options.Store, tokens: options.Tokens, ca: pair.Leaf, caKey: caKey, leafKey: leafKey,
		transport: outbound{next: transport, headerTimeout: cmp.Or(options.HeaderTimeout, headerTimeout)}, dial: dial, logger: options.Logger,
		handshakeTimeout: cmp.Or(options.HandshakeTimeout, handshakeTimeout), idleTimeout: cmp.Or(options.IdleTimeout, idleTimeout)}, nil
}

// ServeHTTP opens a tunnel to port 443 of a host the sandbox's grants or allowed domains cover,
// and nothing else: a request that is not CONNECT, a sandbox without a valid proxy credential,
// an IP literal, a host outside both or another port is refused. A host both cover is a grant's.
func (p *Proxy) ServeHTTP(w http.ResponseWriter, r *http.Request) {
	if r.Method != http.MethodConnect {
		w.Header().Set("Allow", http.MethodConnect)
		http.Error(w, "egress: only CONNECT is proxied", http.StatusMethodNotAllowed)
		return
	}
	credential := proxyCredential(r.Header.Get("Proxy-Authorization"))
	record, found, err := p.lookup(r.Context(), credential)
	if err != nil {
		p.logger.Warn("the sandbox egress proxy could not read a sandbox's grants", "error", err)
		http.Error(w, "egress: the sandbox's grants cannot be read just now", http.StatusServiceUnavailable)
		return
	}
	if !found {
		w.Header().Set("Proxy-Authenticate", authenticate)
		http.Error(w, "egress: no such sandbox", http.StatusProxyAuthRequired)
		return
	}
	host, port, err := net.SplitHostPort(r.Host)
	if err != nil {
		http.Error(w, "egress: CONNECT takes a host and a port", http.StatusBadRequest)
		return
	}
	// A name with a trailing dot is the name without it: matching and dialing take the same host.
	host = strings.TrimSuffix(strings.ToLower(host), ".")
	// A grant names hosts, never addresses, so an address cannot be told apart from another
	// host's (core.SandboxRule). A * is no host's name: a wildcard grant would match it, and
	// its certificate would be one for every name under the domain.
	granted := record.granted(host)
	if host == "" || net.ParseIP(host) != nil || strings.Contains(host, "*") || !granted && !record.allowed(host) {
		p.logger.Warn("the sandbox egress proxy refused a host", "session", record.Session, "host", host)
		http.Error(w, "egress: the sandbox has no grant for this host", http.StatusForbidden)
		return
	}
	// A grant is for a host's HTTPS API; another port is another service on it.
	if port != httpsPort {
		p.logger.Warn("the sandbox egress proxy refused a port", "session", record.Session, "host", host, "port", port)
		http.Error(w, "egress: only port 443 is proxied", http.StatusForbidden)
		return
	}
	if !granted {
		p.passthrough(w, r, record.Session, host)
		return
	}
	conn, buffered, err := http.NewResponseController(w).Hijack()
	if err != nil {
		http.Error(w, "egress: the connection cannot be taken over", http.StatusInternalServerError)
		return
	}
	if _, err := io.WriteString(conn, "HTTP/1.1 200 Connection established\r\n\r\n"); err != nil {
		_ = conn.Close()
		return
	}
	early, _ := buffered.Reader.Peek(buffered.Reader.Buffered())
	p.tunnel(&readFirst{Conn: conn, first: bytes.NewReader(early)}, tunnel{credential: credential, host: host})
}

// passthrough tunnels the sandbox's bytes to port 443 of host, an allowed domain, as they are:
// no certificate of the router's, no token, nothing read or logged. A host that resolves to a
// non-public address is not dialed (502).
func (p *Proxy) passthrough(w http.ResponseWriter, r *http.Request, session, host string) {
	upstream, err := p.dial(r.Context(), "tcp", net.JoinHostPort(host, httpsPort))
	if err != nil {
		// A dial's error holds no byte the host sent.
		p.logger.Warn("the sandbox egress proxy could not reach an allowed host", "session", session, "host", host, "error", err)
		http.Error(w, "egress: the host cannot be reached", http.StatusBadGateway)
		return
	}
	conn, buffered, err := http.NewResponseController(w).Hijack()
	if err != nil {
		_ = upstream.Close()
		http.Error(w, "egress: the connection cannot be taken over", http.StatusInternalServerError)
		return
	}
	if _, err := io.WriteString(conn, "HTTP/1.1 200 Connection established\r\n\r\n"); err != nil {
		_ = conn.Close()
		_ = upstream.Close()
		return
	}
	early, _ := buffered.Reader.Peek(buffered.Reader.Buffered())
	sandbox := &readFirst{Conn: conn, first: bytes.NewReader(early)}
	// Either side ending ends the tunnel: closing both stops the other copy.
	done := make(chan struct{}, 2)
	go func() { _, _ = io.Copy(upstream, sandbox); done <- struct{}{} }()
	go func() { _, _ = io.Copy(conn, upstream); done <- struct{}{} }()
	<-done
	_ = conn.Close()
	_ = upstream.Close()
}

// tunnel is one CONNECT: the sandbox's credential and the host it asked for.
type tunnel struct {
	credential, host string
}

// tunnel serves conn as host with a certificate the CA signs, and sends each request in it
// on through the proxy's transport.
func (p *Proxy) tunnel(conn net.Conn, t tunnel) {
	leaf, err := p.leaf(t.host)
	if err != nil {
		p.logger.Warn("the sandbox egress proxy could not sign a host's certificate", "host", t.host, "error", err)
		_ = conn.Close()
		return
	}
	server := &http.Server{
		Handler:           p.inner(t),
		ReadHeaderTimeout: p.handshakeTimeout,
		IdleTimeout:       p.idleTimeout,
		// What it logs is the sandbox's side of the tunnel, a failed handshake or a panic in
		// inner.
		ErrorLog: slog.NewLogLogger(p.logger.Handler(), slog.LevelInfo),
	}
	// HTTP/1.1 only: the server here serves no HTTP/2, so a sandbox offering h2 is held to
	// HTTP/1.1 rather than refused.
	tlsConn := tls.Server(conn, &tls.Config{Certificates: []tls.Certificate{leaf}, NextProtos: []string{"http/1.1"},
		MinVersion: tls.VersionTLS12})
	// Serve returns once the one connection is taken; the server goes on serving it.
	_ = server.Serve(&single{conn: tlsConn})
}

// inner forwards one request in t's tunnel. It reads the grants again for each, so after
// Delete no request goes out on a tunnel the credential opened.
func (p *Proxy) inner(t tunnel) http.HandlerFunc {
	return func(w http.ResponseWriter, r *http.Request) {
		host := strings.ToLower(r.Host)
		if name, _, err := net.SplitHostPort(host); err == nil {
			host = name
		}
		// Host decides which site of the address gets the request; one other than the
		// tunnel's could be another tenant's (domain fronting).
		if host != t.host {
			http.Error(w, "egress: Host is not the host the tunnel was opened to", http.StatusMisdirectedRequest)
			return
		}
		// An upgraded connection's bytes would pass without the response rewrite.
		if r.Header.Get("Upgrade") != "" {
			http.Error(w, "egress: protocol upgrades are not proxied", http.StatusNotImplemented)
			return
		}
		record, _, err := p.lookup(r.Context(), t.credential)
		if err != nil {
			p.logger.Warn("the sandbox egress proxy could not read a sandbox's grants", "error", err)
			http.Error(w, "egress: the sandbox's grants cannot be read just now", http.StatusServiceUnavailable)
			return
		}
		// No record, after Delete or expiry, grants nothing.
		if !record.granted(host) {
			http.Error(w, "egress: the sandbox's grant for this host has ended", http.StatusForbidden)
			return
		}
		logger := p.logger.With("session", record.Session, "host", host)
		forward := &httputil.ReverseProxy{
			Rewrite:   func(pr *httputil.ProxyRequest) { p.toHost(pr, t) },
			Transport: p.transport,
			// It logs a body lost mid-copy, whose error may quote the host's bytes: a
			// malformed trailer line that echoes the token.
			ErrorLog: log.New(fixedLine{logger: logger, message: "the sandbox egress proxy lost a host's response mid-body"}, "", 0),
			ErrorHandler: func(w http.ResponseWriter, _ *http.Request, err error) {
				failed(logger, err)
				w.WriteHeader(http.StatusBadGateway)
			},
		}
		grant, form, placed := record.placed(host, r.Header.Get("Authorization"))
		if !placed {
			forward.ServeHTTP(w, r)
			return
		}
		token, err := p.tokens(r.Context(), grant)
		if err != nil {
			// A provider's refresh error may quote its answer (resolver.statusAfter wraps it),
			// so only its class is logged.
			logger.Warn("the sandbox egress proxy has no token for a grant", "binding", grant.Binding, "class", errorClass(err))
			status := http.StatusBadGateway
			if errors.Is(err, sandbox.ErrNoCredential) {
				status = http.StatusUnauthorized
			}
			http.Error(w, "egress: the connection gives no token: "+grant.Binding, status)
			return
		}
		pairs := echoes(token, grant.Placeholder)
		if form.basic {
			pairs = append(pairs, form.credential(token), form.credential(grant.Placeholder))
		}
		// An echo of the whole Authorization value in base64 holds the token at another
		// alignment than the token's own base64.
		pairs = append(pairs, wholeInBase64(form.scheme+" "+form.credential(token), form.scheme+" "+form.credential(grant.Placeholder))...)
		forward.Rewrite = func(pr *httputil.ProxyRequest) {
			p.toHost(pr, t)
			pr.Out.Header.Set("Authorization", form.scheme+" "+form.credential(token))
			// An Accept-Encoding the request sets keeps the transport from asking for gzip and
			// decompressing it: the transport decodes the first of two Content-Encoding lines
			// and drops both, leaving encoded nothing to see (net/http Transport.DisableCompression).
			pr.Out.Header.Set("Accept-Encoding", "identity")
			// A connection reused after a host sent more than its response logs those bytes,
			// an echo of the token, to the process's log ("Unsolicited response received on
			// idle HTTP channel", net/http transport.go).
			pr.Out.Close = true
			// A range of an echo is a piece of the token no rewrite recognises.
			pr.Out.Header.Del("Range")
			pr.Out.Header.Del("If-Range")
		}
		forward.ModifyResponse = func(response *http.Response) error {
			// An encoding the transport did not undo hides the token from the rewrite, and
			// the sandbox decodes it.
			if encoded(response.Header) {
				return errEncoded
			}
			replacer := strings.NewReplacer(pairs...)
			for _, values := range response.Header {
				for i, value := range values {
					values[i] = replacer.Replace(value)
				}
			}
			// The transport fills Trailer in at the body's end, after this rewrite, and
			// ReverseProxy sends it on, so it is dropped: Trailer here, the rest on Close.
			response.Trailer = nil
			response.Body = &trailerless{ReadCloser: newRewriting(response.Body, pairs...), response: response}
			response.ContentLength = -1
			response.Header.Del("Content-Length")
			logger.Info("a sandbox request carried a connection's token", "binding", grant.Binding, "status", response.StatusCode)
			return nil
		}
		// ReverseProxy sends a 1xx's headers on before ModifyResponse sees any.
		forward.ServeHTTP(finalOnly{w}, r)
	}
}

// errEncoded is a substituted request's response in an encoding the rewrite cannot read.
var errEncoded = errors.New("egress: the host's response is encoded")

// encoded says whether header names a Content-Encoding other than identity: net/http's
// transport undoes the gzip it asked for itself and removes the header.
func encoded(header http.Header) bool {
	for _, value := range header.Values("Content-Encoding") {
		for coding := range strings.SplitSeq(value, ",") {
			if coding = strings.TrimSpace(coding); coding != "" && !strings.EqualFold(coding, "identity") {
				return true
			}
		}
	}
	return false
}

// failed logs why a request got no answer from its host. Only an error from before the
// connection was had, a dial or a TLS handshake, is logged as it is: any later one may quote
// the host's answer, an echo of the token in a malformed header line.
func failed(logger *slog.Logger, err error) {
	var early unconnected
	switch {
	case errors.Is(err, errEncoded):
		logger.Warn("the sandbox egress proxy refused an encoded response to a request with a token")
	case errors.As(err, &early):
		logger.Warn("the sandbox egress proxy could not reach a host", "error", early.err)
	default:
		logger.Warn("the sandbox egress proxy got no answer it could read from a host", "class", errorClass(err))
	}
}

// errorClass is the kind of err, for a log that must not hold its text.
func errorClass(err error) string {
	var netErr net.Error
	switch {
	case errors.Is(err, errNoHeader):
		return "header_timeout"
	case errors.Is(err, sandbox.ErrNoCredential):
		return "no_credential"
	case errors.Is(err, context.Canceled):
		return "canceled"
	case errors.Is(err, context.DeadlineExceeded):
		return "deadline"
	case errors.As(err, &netErr):
		return "network"
	}
	return "other"
}

// fixedLine is a log.Logger's writer that logs message, never what it was given.
type fixedLine struct {
	logger  *slog.Logger
	message string
}

func (f fixedLine) Write(p []byte) (int, error) {
	f.logger.Warn(f.message)
	return len(p), nil
}

// trailerless is a response body that leaves its response no trailer once it is closed,
// which is when ReverseProxy reads it.
type trailerless struct {
	io.ReadCloser
	response *http.Response
}

func (b *trailerless) Close() error {
	err := b.ReadCloser.Close()
	b.response.Trailer = nil
	return err
}

// finalOnly is a ResponseWriter that sends no informational (1xx) response.
type finalOnly struct {
	http.ResponseWriter
}

func (w finalOnly) WriteHeader(status int) {
	if status >= http.StatusOK {
		w.ResponseWriter.WriteHeader(status)
	}
}

func (w finalOnly) Unwrap() http.ResponseWriter { return w.ResponseWriter }

// errNoHeader is a host that sent no response header within the header timeout.
var errNoHeader = errors.New("egress: the host sent no response header in time")

// outbound sends a request with next, ends it when the host sends no response header
// within headerTimeout, and marks an error from before a connection was had as unconnected.
type outbound struct {
	next          http.RoundTripper
	headerTimeout time.Duration
}

func (o outbound) RoundTrip(request *http.Request) (*http.Response, error) {
	ctx, cancel := context.WithCancel(request.Context())
	var connected atomic.Bool
	ctx = httptrace.WithClientTrace(ctx, &httptrace.ClientTrace{GotConn: func(httptrace.GotConnInfo) { connected.Store(true) }})
	timer := time.AfterFunc(o.headerTimeout, cancel)
	response, err := o.next.RoundTrip(request.WithContext(ctx))
	if !timer.Stop() {
		if err == nil {
			_ = response.Body.Close()
		}
		cancel()
		return nil, stack.Wrap(errNoHeader)
	}
	if err != nil {
		cancel()
		if !connected.Load() {
			return nil, unconnected{err: err}
		}
		return nil, err
	}
	response.Body = cancelling{ReadCloser: response.Body, cancel: cancel}
	return response, nil
}

// unconnected is an error from before a connection to the host was had, which holds no byte
// the host sent.
type unconnected struct {
	err error
}

func (u unconnected) Error() string { return u.err.Error() }
func (u unconnected) Unwrap() error { return u.err }

// cancelling is a response body that ends its request's context once closed.
type cancelling struct {
	io.ReadCloser
	cancel context.CancelFunc
}

func (b cancelling) Close() error {
	err := b.ReadCloser.Close()
	b.cancel()
	return err
}

// toHost sends pr to port 443 of the tunnel's host, whatever its URL says.
func (p *Proxy) toHost(pr *httputil.ProxyRequest, t tunnel) {
	pr.Out.URL.Scheme, pr.Out.URL.Host, pr.Out.Host = "https", t.host, ""
}

// placed is the grant covering host whose placeholder value carries, and how it carries it.
func (r Record) placed(host, value string) (Grant, authorization, bool) {
	for _, grant := range r.Grants {
		if !grant.covers(host) {
			continue
		}
		if form, ok := placed(value, grant.Placeholder); ok {
			return grant, form, true
		}
	}
	return Grant{}, authorization{}, false
}

func (p *Proxy) lookup(ctx context.Context, credential string) (Record, bool, error) {
	if credential == "" {
		return Record{}, false, nil
	}
	return p.store.lookup(ctx, credential)
}

// leaf is host's certificate, signed by the CA.
func (p *Proxy) leaf(host string) (tls.Certificate, error) {
	// RFC 5280 section 4.1.2.2: a positive serial of at most 20 octets.
	serial, err := rand.Int(rand.Reader, new(big.Int).Lsh(big.NewInt(1), 128))
	if err != nil {
		return tls.Certificate{}, stack.Wrap(err)
	}
	now := time.Now()
	template := &x509.Certificate{
		SerialNumber: serial,
		Subject:      pkix.Name{CommonName: host},
		DNSNames:     []string{host},
		NotBefore:    now.Add(-leafValidity),
		NotAfter:     now.Add(leafValidity),
		KeyUsage:     x509.KeyUsageDigitalSignature,
		ExtKeyUsage:  []x509.ExtKeyUsage{x509.ExtKeyUsageServerAuth},
	}
	der, err := x509.CreateCertificate(rand.Reader, template, p.ca, &p.leafKey.PublicKey, p.caKey)
	if err != nil {
		return tls.Certificate{}, stack.Wrap(err)
	}
	return tls.Certificate{Certificate: [][]byte{der}, PrivateKey: p.leafKey}, nil
}

// proxyCredential is the password of a Basic Proxy-Authorization, which is where a client
// sends a proxy URL's userinfo; the user name is not read.
func proxyCredential(value string) string {
	scheme, encoded, _ := strings.Cut(value, " ")
	if !strings.EqualFold(scheme, "Basic") {
		return ""
	}
	raw, err := base64.StdEncoding.DecodeString(encoded)
	if err != nil {
		return ""
	}
	_, password, _ := strings.Cut(string(raw), ":")
	return password
}

// readFirst is a hijacked connection that first gives what the server had already read of
// it: a client may send its TLS hello without waiting for the 200.
type readFirst struct {
	net.Conn
	first io.Reader
}

func (c *readFirst) Read(p []byte) (int, error) {
	if c.first != nil {
		n, err := c.first.Read(p)
		if n > 0 || !errors.Is(err, io.EOF) {
			return n, err
		}
		c.first = nil
	}
	return c.Conn.Read(p)
}

// single is a listener of one connection, for a tunnel's server.
type single struct {
	once sync.Once
	conn net.Conn
}

func (l *single) Accept() (net.Conn, error) {
	var conn net.Conn
	l.once.Do(func() { conn = l.conn })
	if conn == nil {
		return nil, net.ErrClosed
	}
	return conn, nil
}

func (l *single) Close() error   { return nil }
func (l *single) Addr() net.Addr { return l.conn.LocalAddr() }
