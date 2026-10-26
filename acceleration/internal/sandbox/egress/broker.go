package egress

import (
	"context"
	"crypto/rand"
	"encoding/pem"
	"errors"
	"fmt"
	"log/slog"
	"maps"
	"net/url"
	"strings"
	"sync"
	"time"

	"github.com/GetStream/Vision-Agents/acceleration/internal/config"
	"github.com/GetStream/Vision-Agents/acceleration/internal/sandbox"
	"github.com/GetStream/Vision-Agents/acceleration/internal/stack"
)

// BrokerName is the code_sandbox.broker setting that picks Broker.
const BrokerName = "proxy"

// caPath is where the sandbox finds the proxy CA's certificate, and bundlePath the sandbox's
// own roots with it: /tmp, since every user code may run as can write there.
const (
	caPath     = "/tmp/va-egress-ca.pem"
	bundlePath = "/tmp/va-egress-ca-bundle.pem"
)

// bundle writes bundlePath from the first roots file the sandbox has, and fails without one: the
// system roots files Go reads on Linux (https://go.dev/src/crypto/x509/root_linux.go). Daytona's
// own image is Debian, whose file is the first (live probe, 2026-10-21: 150 roots there). A host
// of the config's allowed domains, which the proxy tunnels to as it is, is verified with them.
const bundle = `for roots in /etc/ssl/certs/ca-certificates.crt /etc/pki/tls/certs/ca-bundle.crt /etc/ssl/ca-bundle.pem ` +
	`/etc/pki/tls/cacert.pem /etc/pki/ca-trust/extracted/pem/tls-ca-bundle.pem /etc/ssl/cert.pem; do ` +
	`if [ -s "$roots" ]; then { cat "$roots"; echo; cat ` + caPath + `; } > ` + bundlePath + ` && exit 0; exit 1; fi; done; ` +
	`echo "no system CA bundle" >&2; exit 1`

// recordTTL is how long a sandbox's grants last after its last run started. BeforeRun renews
// them, so it need only outlast the longest run, and a sandbox left idle loses nothing it
// cannot get back on its next run.
const recordTTL = sandbox.MaxTimeout + time.Minute

// bundleEnv are the env vars that point OpenSSL, Python requests, curl and git at the roots they
// trust instead of the system's, so they name the bundle: the ones Daytona's netleash sets (live
// probe, 2026-10-21). Node's NODE_EXTRA_CA_CERTS adds to its own roots, so it names the CA alone.
var bundleEnv = []string{"SSL_CERT_FILE", "REQUESTS_CA_BUNDLE", "CURL_CA_BUNDLE", "GIT_SSL_CAINFO"}

// proxyEnv are the env vars that send curl, Python and git through the proxy, in both cases:
// curl reads http_proxy in lower case only (https://curl.se/docs/manpage.html#ENVIRONMENT). The
// sandbox reaches the proxy alone and has no DNS, so a client that ignores them reaches nothing.
var proxyEnv = []string{"HTTPS_PROXY", "HTTP_PROXY", "https_proxy", "http_proxy"}

// gitEnv has git send the proxy credential on its first CONNECT. By default it sends none,
// waits for the 407 and connects again (http.proxyAuthMethod «anyauth»: «It is assumed that
// the proxy answers an unauthenticated request with a 407»); basic saves that round trip
// (tunnel probe 2, 2026-10-22: one CONNECT without auth before each authenticated one).
// GIT_HTTP_PROXY_AUTHMETHOD overrides http.proxyAuthMethod, and a user's own GIT_CONFIG_* does not
// replace it: git-config(1), http.proxyAuthMethod,
// https://git-scm.com/docs/git-config#Documentation/git-config.txt-httpproxyAuthMethod.
var gitEnv = map[string]string{"GIT_HTTP_PROXY_AUTHMETHOD": "basic"}

// noProxy keeps the sandbox's own loopback off the proxy, as Daytona's netleash does (live probe,
// 2026-10-21).
const noProxy = "localhost,127.0.0.1,::1"

// BrokerOptions are what Broker needs: the proxy and its store, and the session's tokens.
type BrokerOptions struct {
	Store *Store
	// URL is the proxy as a sandbox reaches it, code_sandbox.proxy_public_url: http or https, an
	// IPv4 address and a port. With https the sandbox's clients reach it through its TLS tunnel.
	URL string
	// CACert is the PEM certificate of the CA the proxy signs each host's certificate with,
	// which the sandbox trusts. Never its key.
	CACert []byte
	// Session is the session the sandbox is for, and Allowed its agent config's allowed_domains,
	// which the proxy tunnels to without a token.
	Session string
	Allowed []string
	Tokens  sandbox.Tokens
	// Bind gives a late grant its connection before each run. A late grant needs it.
	Bind   sandbox.Bind
	Logger *slog.Logger
}

// Broker is the sandbox.CredentialBroker over the router's egress proxy: the sandbox holds a
// placeholder per grant and a proxy credential, and the proxy asks for the token on each
// request. Nothing the provider keeps holds a token.
type Broker struct {
	options BrokerOptions
	proxy   *url.URL
	// allowed are options.Allowed as a Record's hosts.
	allowed []string

	mu sync.Mutex
	// credential is the sandbox's proxy credential while its record is in the store, and held
	// the grants in it. closed is a Close since the last Prepare, whose Delete a run retries.
	credential string
	held       []held
	closed     bool
}

// held is a grant in the record, and the sandbox's grant it comes from.
type held struct {
	grant sandbox.Grant
	entry Grant
}

func NewBroker(o BrokerOptions) (*Broker, error) {
	// The sandbox gets the CA's certificates alone, whatever else the setting holds.
	o.CACert = certificates(o.CACert)
	if o.Store == nil || len(o.CACert) == 0 || o.Session == "" || o.Tokens == nil {
		return nil, stack.Wrap(errors.New("egress: the proxy broker needs redis, the proxy's CA certificate, a session and tokens"))
	}
	proxy, ok := config.ProxyURL(o.URL)
	if !ok {
		return nil, stack.Wrap(errors.New("egress: the proxy's public URL is http or https://<public IPv4 address>:port, with no user, path, query or fragment"))
	}
	if o.Logger == nil {
		o.Logger = slog.Default()
	}
	var allowed []string
	for _, domain := range o.Allowed {
		// A domain with no name in it (*, *., ..) takes no host, since *. would take any name
		// that ends in a dot.
		if host := allowedHost(domain); host != "" {
			allowed = append(allowed, host)
		}
	}
	return &Broker{options: o, proxy: proxy, allowed: allowed}, nil
}

// allowedHost is a config's allowed domain as a Record's host: lower case, without a trailing
// dot, and a leading *. or . as *., which takes the domain and every name under it. A domain
// with no name left is "".
func allowedHost(domain string) string {
	domain = strings.TrimSuffix(strings.ToLower(domain), ".")
	name := strings.TrimLeft(domain, "*.")
	if name != domain && name != "" {
		return "*." + name
	}
	return name
}

func (b *Broker) Prepare(ctx context.Context, grants []sandbox.Grant) (sandbox.Mount, error) {
	b.mu.Lock()
	defer b.mu.Unlock()
	if err := b.release(ctx); err != nil {
		return sandbox.Mount{}, err
	}
	b.closed = false
	env := map[string]string{}
	var prepared []held
	for _, grant := range grants {
		if grant.Late && b.options.Bind == nil {
			return sandbox.Mount{}, stack.Wrap(fmt.Errorf("egress: connector %q is bound late, and the broker has no Bind", grant.Binding))
		}
		// A provider's refresh error may quote its answer, so only its class is logged, and a
		// required grant's error carries none of it. A late grant has no connection to ask yet:
		// the proxy refuses its placeholder until BeforeRun binds it.
		var err error
		if !grant.Late {
			_, err = b.options.Tokens(ctx, grant)
		}
		if errors.Is(err, sandbox.ErrNoCredential) {
			if grant.Required {
				return sandbox.Mount{}, fmt.Errorf("egress: required connector %q gives the sandbox no credential: %w", grant.Binding, sandbox.ErrNoCredential)
			}
			b.options.Logger.Warn("the sandbox is created without a connector's credential", "connector", grant.Binding,
				"connection", grant.Connection.ConnectionID, "class", errorClass(err))
			continue
		}
		if err != nil {
			b.options.Logger.Warn("a connector's credential cannot be had just now; the proxy asks again on each request",
				"connector", grant.Binding, "connection", grant.Connection.ConnectionID, "class", errorClass(err))
		}
		placeholder := "va_ph_" + rand.Text()
		prepared = append(prepared, held{grant: grant, entry: Grant{Connection: grant.Connection, Binding: grant.Binding,
			Config: grant.Config, Connector: grant.Connector, Selection: grant.Selection, Scheme: grant.Scheme,
			Placeholder: placeholder, Hosts: grant.Hosts}})
		env[grant.Env] = placeholder
	}
	if len(prepared) == 0 {
		return sandbox.Mount{}, nil
	}
	credential := NewCredential()
	if err := b.options.Store.Put(ctx, credential, b.record(prepared), recordTTL); err != nil {
		return sandbox.Mount{}, err
	}
	b.credential, b.held = credential, prepared
	// The sandbox reaches the proxy alone and sends it every request with the credential.
	proxy := *b.proxy
	proxy.User = url.UserPassword("sbx", credential)
	run := map[string]string{"NODE_EXTRA_CA_CERTS": caPath, "NO_PROXY": noProxy, "no_proxy": noProxy}
	maps.Copy(run, gitEnv)
	for _, name := range bundleEnv {
		run[name] = bundlePath
	}
	mount := sandbox.Mount{Env: env, RunEnv: run, Proxy: &proxy, Files: map[string][]byte{caPath: b.options.CACert}, Setup: []string{bundle}}
	clients := proxy
	if proxy.Scheme == "https" {
		clients = url.URL{Scheme: "http", User: proxy.User, Host: "127.0.0.1:" + tunnelPort}
		start := startTunnel(b.proxy)
		mount.Files[tunnelPath] = forwarder
		mount.Setup = append(mount.Setup, start)
		mount.EachRun = []string{start}
		// Node's fetch and https ignore the proxy env vars without it
		// (https://nodejs.org/api/cli.html#node_use_env_proxy1; live probe, 2026-10-22).
		run["NODE_USE_ENV_PROXY"] = "1"
	}
	for _, name := range proxyEnv {
		run[name] = clients.String()
	}
	return mount, nil
}

// BeforeRun binds each late grant Bind has a connection for, withdraws a grant whose connection
// gives sandbox.ErrNoCredential, and stores the record again, so the proxy sends a bound grant's
// token from the next request on, and the record outlives the run whatever time passed since the
// last one. A newer revision needs nothing: the proxy asks for the token on every request.
func (b *Broker) BeforeRun(ctx context.Context) error {
	b.mu.Lock()
	defer b.mu.Unlock()
	if b.closed {
		return b.release(ctx)
	}
	if b.credential == "" {
		return nil
	}
	var kept []held
	for _, h := range b.held {
		if h.grant.Late {
			bound, ok := b.options.Bind(ctx, h.grant)
			if !ok {
				kept = append(kept, h)
				continue
			}
			h.grant = h.grant.Bound(bound)
			h.entry.Connection, h.entry.Scheme = h.grant.Connection, h.grant.Scheme
		}
		_, err := b.options.Tokens(ctx, h.grant)
		if errors.Is(err, sandbox.ErrNoCredential) {
			b.options.Logger.Warn("a connector's credential is withdrawn from the sandbox", "connector", h.grant.Binding,
				"class", errorClass(err))
			continue
		}
		if err != nil {
			b.options.Logger.Warn("a connector's credential cannot be had just now; the proxy asks again on each request",
				"connector", h.grant.Binding, "class", errorClass(err))
		}
		kept = append(kept, h)
	}
	// The record keeps the allowed domains reachable once no grant is left.
	if len(kept) == 0 && len(b.allowed) == 0 {
		return b.release(ctx)
	}
	if err := b.options.Store.Put(ctx, b.credential, b.record(kept), recordTTL); err != nil {
		return err
	}
	b.held = kept
	return nil
}

func (b *Broker) Close(ctx context.Context) error {
	b.mu.Lock()
	defer b.mu.Unlock()
	b.closed = true
	return b.release(ctx)
}

// release deletes the sandbox's record, keeping its credential for the next try when the
// store refuses. The caller holds mu.
func (b *Broker) release(ctx context.Context) error {
	if b.credential == "" {
		return nil
	}
	if err := b.options.Store.Delete(ctx, b.credential); err != nil {
		return err
	}
	b.credential, b.held = "", nil
	return nil
}

func (b *Broker) record(grants []held) Record {
	record := Record{Session: b.options.Session, Allowed: b.allowed}
	for _, h := range grants {
		record.Grants = append(record.Grants, h.entry)
	}
	return record
}

// certificates are the CERTIFICATE blocks of raw, in PEM.
func certificates(raw []byte) []byte {
	var certs []byte
	for block, rest := pem.Decode(raw); block != nil; block, rest = pem.Decode(rest) {
		if block.Type == "CERTIFICATE" {
			certs = append(certs, pem.EncodeToMemory(block)...)
		}
	}
	return certs
}
