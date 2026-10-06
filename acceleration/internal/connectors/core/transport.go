package core

import (
	"bytes"
	"errors"
	"io"
	"net/http"
	"sync"
	"sync/atomic"
	"time"

	"github.com/GetStream/Vision-Agents/acceleration/internal/egress"
	"github.com/GetStream/Vision-Agents/acceleration/internal/stack"
)

// maxClassifiedBody caps what of a 401's body is read for Scheme.Classify, which looks for an
// error member in it (RFC 6749 section 5.2, RFC 6750 section 3). It is the cap oauth2code
// reads any token response with (maxResponseBytes in schemes/oauth2code/discovery.go), the
// prototype's limit (internal/mcp/oauth.go:124 on codex/connector-support at cf62af0d). The
// cap is a choice, not a measurement. The rest of the body stays readable for the caller.
const maxClassifiedBody = 1 << 20

// maxDrainedBody is how much of a refused answer is read away before the request goes again,
// so its connection can be reused. It is net/http's own limit for the body it drains before
// following a redirect (maxBodySlurpSize, net/http/client.go:725 in go1.27). A longer body
// costs that connection, not the client's Timeout.
const maxDrainedBody = 2 << 10

// maxAttempts is how often one request is sent with a credential the provider may refuse:
// the design's «On a 401 the Router calls Invalidate, refreshes and retries once»
// (docs/connectors/channels.md on connectors/planning, «The proxy»; subtasks.md T44, «On a 401
// it calls Invalidate and retries once»).
const maxAttempts = 2

// idleAfter is how long a connection's client may go unused before Transports drops it. It is
// IdleConnTimeout of net/http's DefaultTransport (90 s, net/http/transport.go:55 in go1.27),
// which egress.NewClient clones: by then the client holds no idle socket, so dropping it costs
// only building a new one on the next use. The same span is how often unused clients are looked
// for, as the resolver sweeps its cache once per maxAge.
const idleAfter = 90 * time.Second

// NewClientFunc builds a connection's outbound client around wrap, as egress.NewClient does.
type NewClientFunc func(timeout time.Duration, wrap func(http.RoundTripper) http.RoundTripper) *http.Client

// TransportsConfig is what Transports builds clients from.
type TransportsConfig struct {
	// Resolver hands out the access credential each request carries.
	Resolver Resolver
	// Timeout bounds one request, redirects included, as http.Client.Timeout does. Zero is
	// none: the request's context bounds it.
	Timeout time.Duration
	// NewClient is egress.NewClient when nil, which is what the router runs. A test sets it
	// to reach a loopback fake, which egress refuses.
	NewClient NewClientFunc
	// Now is the clock unused clients are judged by; nil is time.Now.
	Now func() time.Time
}

// Transports builds and keeps the one outbound client of each connection. The client is
// egress.NewClient's, with the scheme's Wrap as its wrap: per request, the access credential
// comes from the Resolver, the scheme applies it to the final request, and the egress
// transport under it checks the URL again and dials only a checked public IP. The redirect
// policy is the egress client's, so a credential never follows a redirect to another origin.
type Transports struct {
	resolver  Resolver
	timeout   time.Duration
	newClient NewClientFunc
	now       func() time.Time

	mu      sync.Mutex
	clients map[ConnectionRef]*transport
	// swept is when unused clients were last dropped.
	swept time.Time
}

// transport is one connection's client and when it was last asked for or sent with, in Unix
// nanoseconds.
type transport struct {
	client *http.Client
	used   atomic.Int64
}

// NewTransports is the Transports cfg describes.
func NewTransports(cfg TransportsConfig) (*Transports, error) {
	if cfg.Resolver == nil {
		return nil, stack.Wrap(errors.New("core: transports need a resolver"))
	}
	newClient := cfg.NewClient
	if newClient == nil {
		newClient = egress.NewClient
	}
	now := cfg.Now
	if now == nil {
		now = time.Now
	}
	return &Transports{resolver: cfg.Resolver, timeout: cfg.Timeout, newClient: newClient, now: now,
		clients: map[ConnectionRef]*transport{}}, nil
}

// Client is the outbound client of ref's connection, which scheme authenticates. The first
// call for ref builds it and later calls share its connection pool. Each call returns a copy
// of its own, so a caller that sets a field changes only its copy: replacing the copy's
// Transport drops the credential and the egress checks for that caller alone.
//
// Building one also drops, at most once per idleAfter, the clients nothing asked for or sent
// with for idleAfter, so connections no longer used do not keep theirs. A copy a caller still
// holds keeps working on its own pool.
func (t *Transports) Client(ref ConnectionRef, scheme Scheme) *http.Client {
	now := t.now()
	t.mu.Lock()
	defer t.mu.Unlock()
	existing, found := t.clients[ref]
	if !found {
		t.sweep(now)
		existing = &transport{}
		existing.client = t.newClient(t.timeout, func(base http.RoundTripper) http.RoundTripper {
			return &credentialed{base: base, resolver: t.resolver, ref: ref, scheme: scheme,
				used: &existing.used, now: t.now}
		})
		t.clients[ref] = existing
	}
	existing.used.Store(now.UnixNano())
	copied := *existing.client
	return &copied
}

// Close drops ref's client and closes its idle connections, for a connection that was
// deleted or disconnected. A copy a caller still holds keeps working, and each request it
// sends still asks the Resolver first, which refuses a connection that is gone. The next
// Client for ref builds a new one.
func (t *Transports) Close(ref ConnectionRef) {
	t.mu.Lock()
	existing, found := t.clients[ref]
	delete(t.clients, ref)
	t.mu.Unlock()
	if found {
		existing.client.CloseIdleConnections()
	}
}

// sweep drops the clients unused for idleAfter, at most once per idleAfter. t.mu is held.
func (t *Transports) sweep(now time.Time) {
	if now.Sub(t.swept) < idleAfter {
		return
	}
	for ref, existing := range t.clients {
		if now.Sub(time.Unix(0, existing.used.Load())) >= idleAfter {
			delete(t.clients, ref)
			existing.client.CloseIdleConnections()
		}
	}
	t.swept = now
}

// credentialed is the wrap a connection's client is built with. egress.NewClient checks the
// URL before it and hands it, as base, the transport that checks the URL again and dials only
// a checked public IP. So the scheme's Wrap sees the final request, headers and body, and the
// egress check judges what Wrap produced.
type credentialed struct {
	base     http.RoundTripper
	resolver Resolver
	ref      ConnectionRef
	scheme   Scheme
	used     *atomic.Int64
	now      func() time.Time
}

// RoundTrip resolves the access credential for the request, with the request's deadline as
// the call's budget, and sends the request through the scheme's Wrap of it.
//
// A 401 the scheme classifies as invalid_grant or scope_required is the provider refusing
// that credential, which may still be a grant that works: its clock may run ahead of the
// router's, or it may have ended the token early. So the Resolver is first asked for a
// credential renewed past the refused one (CredentialRequest.Refused): a refresh, or the
// credential another router already renewed. With one, the request goes once more, when its
// body can be read again. A refused renewal is the Resolver's to act on, as any failed
// renewal is. Only when nothing renews it (a static key, a token with no refresh token), or
// the renewed one is refused too, is the Resolver told (Invalidate), and the 401 is the answer.
func (c *credentialed) RoundTrip(request *http.Request) (*http.Response, error) {
	c.used.Store(c.now().UnixNano())
	credential, err := c.resolve(request, nil)
	if err != nil {
		if request.Body != nil {
			_ = request.Body.Close()
		}
		return nil, err
	}
	for attempt := 1; ; attempt++ {
		response, err := c.scheme.Wrap(c.base, credential).RoundTrip(request)
		if err != nil || response.StatusCode != http.StatusUnauthorized {
			return response, err
		}
		outcome := c.classify(response)
		if outcome.Kind != OutcomeInvalidGrant && outcome.Kind != OutcomeScopeRequired {
			return response, nil
		}
		if attempt == maxAttempts {
			return c.invalidate(request, response, credential, outcome)
		}
		renewed, err := c.resolve(request, &credential)
		if err != nil {
			// The renewal failed, and the Resolver moved the connection as that failure says:
			// the provider's refusal is the answer.
			return response, nil
		}
		if renewed.Revision == credential.Revision && renewed.ExpiresAt.Equal(credential.ExpiresAt) {
			// Nothing renews it: only a new grant helps.
			return c.invalidate(request, response, credential, outcome)
		}
		again, replayable := replay(request)
		if !replayable {
			// Renewed for the next call; this one cannot be sent again.
			return response, nil
		}
		_, _ = io.CopyN(io.Discard, response.Body, maxDrainedBody)
		_ = response.Body.Close()
		request, credential = again, renewed
	}
}

// resolve asks the Resolver for the request's credential, renewed past refused when it is set.
func (c *credentialed) resolve(request *http.Request, refused *AccessCredential) (AccessCredential, error) {
	deadline, _ := request.Context().Deadline()
	return c.resolver.Resolve(request.Context(), c.ref, CredentialRequest{Deadline: deadline, Refused: refused})
}

// invalidate tells the Resolver the provider refused credential, and answers with response.
func (c *credentialed) invalidate(request *http.Request, response *http.Response, credential AccessCredential, why Outcome) (*http.Response, error) {
	if err := c.resolver.Invalidate(request.Context(), c.ref, credential, why); err != nil {
		_ = response.Body.Close()
		return nil, stack.Wrap(err)
	}
	return response, nil
}

// classify is the scheme's outcome for a 401. It reads at most maxClassifiedBody of the
// body and puts what it read back in front of the rest, so the caller reads the whole body.
func (c *credentialed) classify(response *http.Response) Outcome {
	read, err := io.ReadAll(io.LimitReader(response.Body, maxClassifiedBody))
	response.Body = struct {
		io.Reader
		io.Closer
	}{io.MultiReader(bytes.NewReader(read), response.Body), response.Body}
	return c.scheme.Classify(response, read, err)
}

// replay is a copy of request to send again, with a body of its own from GetBody, which
// http.NewRequest sets for the bodies it can read twice and which works after the first send
// consumed Body, as it does for a redirect. A body without GetBody cannot be sent again.
func replay(request *http.Request) (*http.Request, bool) {
	again := request.Clone(request.Context())
	if request.Body == nil || request.Body == http.NoBody {
		return again, true
	}
	if request.GetBody == nil {
		return nil, false
	}
	body, err := request.GetBody()
	if err != nil {
		return nil, false
	}
	again.Body = body
	return again, true
}
