package core

import (
	"bytes"
	"errors"
	"io"
	"net/http"
	"sync"
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

// maxAttempts is how often one request is sent with a credential the provider may refuse:
// the design's «On a 401 the Router calls Invalidate, refreshes and retries once»
// (docs/connectors/channels.md on connectors/planning, «The proxy»; subtasks.md T44, «On a 401
// it calls Invalidate and retries once»).
const maxAttempts = 2

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

	mu      sync.Mutex
	clients map[ConnectionRef]*http.Client
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
	return &Transports{resolver: cfg.Resolver, timeout: cfg.Timeout, newClient: newClient,
		clients: map[ConnectionRef]*http.Client{}}, nil
}

// Client is the outbound client of ref's connection, which scheme authenticates. The first
// call for ref builds it and later calls share its connection pool. Each call returns a copy
// of its own, so a caller that sets a field changes only its copy: replacing the copy's
// Transport drops the credential and the egress checks for that caller alone.
func (t *Transports) Client(ref ConnectionRef, scheme Scheme) *http.Client {
	t.mu.Lock()
	defer t.mu.Unlock()
	client, found := t.clients[ref]
	if !found {
		client = t.newClient(t.timeout, func(base http.RoundTripper) http.RoundTripper {
			return &credentialed{base: base, resolver: t.resolver, ref: ref, scheme: scheme}
		})
		t.clients[ref] = client
	}
	copied := *client
	return &copied
}

// Close drops ref's client and closes its idle connections, for a connection that was
// deleted or disconnected. A copy a caller still holds keeps working, and each request it
// sends still asks the Resolver first, which refuses a connection that is gone. The next
// Client for ref builds a new one.
func (t *Transports) Close(ref ConnectionRef) {
	t.mu.Lock()
	client, found := t.clients[ref]
	delete(t.clients, ref)
	t.mu.Unlock()
	if found {
		client.CloseIdleConnections()
	}
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
}

// RoundTrip resolves the access credential for the request, with the request's deadline as
// the call's budget, and sends the request through the scheme's Wrap of it.
//
// A 401 is the provider refusing that credential. When the scheme classifies it as
// invalid_grant or scope_required, the Resolver is told (Invalidate), with the credential
// that was refused. The request is then sent once more, only when its body can be sent again
// and the Resolver hands out another credential: one renewed past the refused one, as after
// another router renewed it or it had expired. Otherwise the 401 is the answer.
func (c *credentialed) RoundTrip(request *http.Request) (*http.Response, error) {
	credential, err := c.resolve(request)
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
		if err := c.resolver.Invalidate(request.Context(), c.ref, credential, outcome); err != nil {
			_ = response.Body.Close()
			return nil, stack.Wrap(err)
		}
		if attempt == maxAttempts {
			return response, nil
		}
		again, replayable := replay(request)
		if !replayable {
			return response, nil
		}
		renewed, err := c.resolve(again)
		if err != nil || (renewed.Revision == credential.Revision && renewed.ExpiresAt.Equal(credential.ExpiresAt)) {
			// The Resolver moved the connection to needs_reauthorization, or has nothing
			// newer: the provider's refusal is the answer.
			if again.Body != nil {
				_ = again.Body.Close()
			}
			return response, nil
		}
		_, _ = io.Copy(io.Discard, response.Body)
		_ = response.Body.Close()
		request, credential = again, renewed
	}
}

func (c *credentialed) resolve(request *http.Request) (AccessCredential, error) {
	deadline, _ := request.Context().Deadline()
	return c.resolver.Resolve(request.Context(), c.ref, CredentialRequest{Deadline: deadline})
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
