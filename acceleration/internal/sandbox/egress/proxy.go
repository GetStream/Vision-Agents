// Package egress is the router's egress proxy for code in a sandbox: the sandbox holds a
// placeholder, and the proxy puts the connection's token in its place on the way out, only
// in the Authorization header of a request to one of the grant's hosts. The token never
// enters the sandbox.
package egress

import (
	"bytes"
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
	"log/slog"
	"math/big"
	"net"
	"net/http"
	"net/http/httputil"
	"strings"
	"sync"
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
	// leafValidity is how long a host's certificate is valid. A client checks it only in the
	// handshake it is minted for; an hour each side leaves room for a sandbox clock that is
	// off by that much.
	leafValidity = time.Hour
	// authenticate is the challenge a 407 must carry (RFC 9110 section 15.5.8), for the
	// Basic credential a proxy URL's userinfo sends (RFC 7617).
	authenticate = `Basic realm="sandbox egress"`
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
	Logger    *slog.Logger
}

// Proxy is an HTTP proxy that takes CONNECT only.
type Proxy struct {
	store     *Store
	tokens    Tokens
	ca        *x509.Certificate
	caKey     crypto.Signer
	leafKey   *ecdsa.PrivateKey
	transport http.RoundTripper
	logger    *slog.Logger
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
	return &Proxy{store: options.Store, tokens: options.Tokens, ca: pair.Leaf, caKey: caKey, leafKey: leafKey,
		transport: transport, logger: options.Logger}, nil
}

// ServeHTTP opens a tunnel to a host the sandbox's grants cover, and nothing else: a
// request that is not CONNECT, a sandbox without a valid proxy credential, an IP literal
// or a host outside its grants is refused.
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
	host = strings.ToLower(host)
	// A grant names hosts, never addresses, so an address cannot be told apart from another
	// host's (core.SandboxRule).
	if net.ParseIP(host) != nil || !record.granted(host) {
		p.logger.Warn("the sandbox egress proxy refused a host", "session", record.Session, "host", host)
		http.Error(w, "egress: the sandbox has no grant for this host", http.StatusForbidden)
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
	p.tunnel(&readFirst{Conn: conn, first: bytes.NewReader(early)}, tunnel{credential: credential, host: host, port: port})
}

// tunnel is one CONNECT: the sandbox's credential and the host and port it asked for.
type tunnel struct {
	credential, host, port string
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
		ReadHeaderTimeout: handshakeTimeout,
		IdleTimeout:       idleTimeout,
		ErrorLog:          slog.NewLogLogger(p.logger.Handler(), slog.LevelDebug),
	}
	// HTTP/1.1 only, so each request goes through inner, not an HTTP/2 stream it never sees.
	tlsConn := tls.Server(conn, &tls.Config{Certificates: []tls.Certificate{leaf}, NextProtos: []string{"http/1.1"},
		MinVersion: tls.VersionTLS12})
	// Serve returns once the one connection is taken; the server goes on serving it.
	_ = server.Serve(&single{conn: tlsConn})
}

// inner forwards one request in t's tunnel. It reads the grants again for each, so a
// credential Delete ends works in no tunnel it opened.
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
		forward := &httputil.ReverseProxy{
			Rewrite:   func(pr *httputil.ProxyRequest) { p.toHost(pr, t) },
			Transport: p.transport,
			ErrorLog:  slog.NewLogLogger(p.logger.Handler(), slog.LevelDebug),
			ErrorHandler: func(w http.ResponseWriter, _ *http.Request, err error) {
				p.logger.Warn("the sandbox egress proxy could not reach a host", "session", record.Session, "host", host, "error", err)
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
			p.logger.Warn("the sandbox egress proxy has no token for a grant", "session", record.Session,
				"binding", grant.Binding, "host", host, "error", err)
			status := http.StatusBadGateway
			if errors.Is(err, sandbox.ErrNoCredential) {
				status = http.StatusUnauthorized
			}
			http.Error(w, "egress: the connection gives no token: "+grant.Binding, status)
			return
		}
		pairs := []string{token, grant.Placeholder}
		if form.basic {
			pairs = append(pairs, form.credential(token), form.credential(grant.Placeholder))
		}
		forward.Rewrite = func(pr *httputil.ProxyRequest) {
			p.toHost(pr, t)
			pr.Out.Header.Set("Authorization", form.scheme+" "+form.credential(token))
			// Without it the transport asks for gzip and decompresses, so the rewrite reads
			// the body as sent (net/http Transport.DisableCompression).
			pr.Out.Header.Del("Accept-Encoding")
		}
		forward.ModifyResponse = func(response *http.Response) error {
			replacer := strings.NewReplacer(pairs...)
			for _, values := range response.Header {
				for i, value := range values {
					values[i] = replacer.Replace(value)
				}
			}
			response.Body = newRewriting(response.Body, pairs...)
			response.ContentLength = -1
			response.Header.Del("Content-Length")
			p.logger.Info("a sandbox request carried a connection's token", "session", record.Session,
				"binding", grant.Binding, "host", host, "status", response.StatusCode)
			return nil
		}
		forward.ServeHTTP(w, r)
	}
}

// toHost sends pr to the tunnel's host and port, whatever its URL says.
func (p *Proxy) toHost(pr *httputil.ProxyRequest, t tunnel) {
	pr.Out.URL.Scheme, pr.Out.URL.Host, pr.Out.Host = "https", t.host, ""
	if t.port != "443" {
		pr.Out.URL.Host = net.JoinHostPort(t.host, t.port)
	}
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
