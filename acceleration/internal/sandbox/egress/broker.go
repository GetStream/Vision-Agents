package egress

import (
	"context"
	"crypto/rand"
	"errors"
	"fmt"
	"log/slog"
	"net/url"
	"sync"
	"time"

	"github.com/GetStream/Vision-Agents/acceleration/internal/sandbox"
	"github.com/GetStream/Vision-Agents/acceleration/internal/stack"
)

// BrokerName is the code_sandbox.broker setting that picks Broker.
const BrokerName = "proxy"

// caPath is where the sandbox finds the proxy CA's certificate: /tmp, since every user code
// may run as can write there.
const caPath = "/tmp/va-egress-ca.pem"

// recordTTL is how long a sandbox's grants last after its last run started. BeforeRun renews
// them, so it need only outlast the longest run, and a sandbox left idle loses nothing it
// cannot get back on its next run.
const recordTTL = sandbox.MaxTimeout + time.Minute

// proxyEnv are the env vars HTTP clients read a proxy from: curl reads only the lowercase
// http_proxy (https://curl.se/docs/manpage.html#ENVIRONMENT), Python's urllib and Go either.
var proxyEnv = []string{"HTTPS_PROXY", "HTTP_PROXY", "https_proxy", "http_proxy"}

// caEnv are the env vars that point OpenSSL, Python requests, Node and git at the CA file.
var caEnv = []string{"SSL_CERT_FILE", "REQUESTS_CA_BUNDLE", "NODE_EXTRA_CA_CERTS", "GIT_SSL_CAINFO"}

// BrokerOptions are what Broker needs: the proxy and its store, and the session's tokens.
type BrokerOptions struct {
	Store *Store
	// URL is the proxy as a sandbox reaches it, code_sandbox.proxy_public_url: http, a host
	// and a port.
	URL string
	// CACert is the PEM certificate of the CA the proxy signs each host's certificate with,
	// which the sandbox trusts. Never its key.
	CACert []byte
	// Session is the session the sandbox is for.
	Session string
	Tokens  sandbox.Tokens
	Logger  *slog.Logger
}

// Broker is the sandbox.CredentialBroker over the router's egress proxy: the sandbox holds a
// placeholder per grant and a proxy credential, and the proxy asks for the token on each
// request. Nothing the provider keeps holds a token.
type Broker struct {
	options BrokerOptions
	proxy   *url.URL

	mu sync.Mutex
	// credential is the sandbox's proxy credential while its record is in the store, and held
	// the grants in it.
	credential string
	held       []held
}

// held is a grant in the record, and the sandbox's grant it comes from.
type held struct {
	grant sandbox.Grant
	entry Grant
}

func NewBroker(o BrokerOptions) (*Broker, error) {
	if o.Store == nil || len(o.CACert) == 0 || o.Session == "" || o.Tokens == nil {
		return nil, stack.Wrap(errors.New("egress: the proxy broker needs redis, the proxy's CA certificate, a session and tokens"))
	}
	proxy, err := url.Parse(o.URL)
	if err != nil || proxy.Scheme != "http" || proxy.Hostname() == "" || proxy.User != nil {
		return nil, stack.Wrap(errors.New("egress: the proxy's public URL is http://host:port, with no user"))
	}
	if o.Logger == nil {
		o.Logger = slog.Default()
	}
	return &Broker{options: o, proxy: proxy}, nil
}

func (b *Broker) Prepare(ctx context.Context, grants []sandbox.Grant) (sandbox.Mount, error) {
	b.mu.Lock()
	defer b.mu.Unlock()
	if err := b.release(ctx); err != nil {
		return sandbox.Mount{}, err
	}
	env := map[string]string{}
	var prepared []held
	for _, grant := range grants {
		if _, err := b.options.Tokens(ctx, grant); err != nil {
			if grant.Required {
				return sandbox.Mount{}, fmt.Errorf("egress: required connector %q gives the sandbox no credential: %w", grant.Binding, err)
			}
			// A provider's refresh error may quote its answer, so only its class is logged.
			b.options.Logger.Warn("the sandbox is created without a connector's credential", "connector", grant.Binding,
				"connection", grant.Connection.ConnectionID, "class", errorClass(err))
			continue
		}
		placeholder := "va_ph_" + rand.Text()
		prepared = append(prepared, held{grant: grant, entry: Grant{Connection: grant.Connection, Binding: grant.Binding,
			Scheme: grant.Scheme, Placeholder: placeholder, Hosts: grant.Hosts}})
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
	proxy := *b.proxy
	proxy.User = url.UserPassword("sbx", credential)
	for _, name := range proxyEnv {
		env[name] = proxy.String()
	}
	for _, name := range caEnv {
		env[name] = caPath
	}
	return sandbox.Mount{Env: env, Proxy: &proxy, Files: map[string][]byte{caPath: b.options.CACert}}, nil
}

// BeforeRun withdraws a grant whose connection gives sandbox.ErrNoCredential and stores the
// record again, so it outlives the run whatever time passed since the last one. A newer
// revision needs nothing: the proxy asks for the token on every request.
func (b *Broker) BeforeRun(ctx context.Context) error {
	b.mu.Lock()
	defer b.mu.Unlock()
	if b.credential == "" {
		return nil
	}
	var kept []held
	for _, h := range b.held {
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
	if err := b.options.Store.Put(ctx, b.credential, b.record(kept), recordTTL); err != nil {
		return err
	}
	b.held = kept
	return nil
}

func (b *Broker) Close(ctx context.Context) error {
	b.mu.Lock()
	defer b.mu.Unlock()
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
	record := Record{Session: b.options.Session}
	for _, h := range grants {
		record.Grants = append(record.Grants, h.entry)
	}
	return record
}
