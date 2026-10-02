package streamapp

import (
	"container/list"
	"context"
	"errors"
	"net/http"
	"sync"
	"time"

	getstream "github.com/GetStream/getstream-go/v5"
)

const (
	// identityTTL is how long a resolved identity is reused before the source is asked
	// again, so a rotated or revoked credential bites within it even when nobody says.
	identityTTL = 30 * time.Second
	// refreshEvery is the least time between two forced refreshes of one credential after
	// Stream refused it, so a key that is simply wrong does not hammer the source.
	refreshEvery = 30 * time.Second
	// defaultMaxClients bounds the clients kept, oldest dropped first.
	defaultMaxClients = 256
	// requestTimeout bounds one request to Stream, as the SDK's own default does.
	requestTimeout = 30 * time.Second
)

// Bound is an identity and the client that acts with it.
type Bound struct {
	Identity Identity
	Client   *getstream.Stream
}

// ClientsOptions configures Clients. All of it is optional.
type ClientsOptions struct {
	// HTTPClient is shared by every Stream client. Empty builds one.
	HTTPClient *http.Client
	// MaxClients bounds how many clients are kept. Zero is a few hundred.
	MaxClients int
	// Now is the clock, for tests.
	Now func() time.Time
}

// Clients resolves customers to identities through a Source and keeps one Stream client
// per credential.
//
// A client is the app's, not the customer's: in deployment mode every customer resolves to
// the same app, and building them a client each would be a transport each for one app.
// getstream-go mints a server token and lazily builds its Chat and Video halves when a
// client is made, without a lock, so a client is handed out only once both exist.
type Clients struct {
	source  Source
	http    *http.Client
	max     int
	now     func() time.Time
	mu      sync.Mutex
	clients map[string]*list.Element
	order   *list.List
	known   map[resolution]resolved
	refresh map[string]time.Time
	checks  map[string]checked
}

// resolution is one question asked of the source: a customer's app for new work, or the
// app a pinned piece of work is finished in.
type resolution struct {
	customer string
	pinned   bool
	reading  bool
	app      int64
}

type kept struct {
	key    string
	client *getstream.Stream
}

type resolved struct {
	identity Identity
	err      error
	at       time.Time
}

// NewClients resolves through source.
func NewClients(source Source, options ClientsOptions) *Clients {
	httpClient := options.HTTPClient
	if httpClient == nil {
		httpClient = &http.Client{Timeout: requestTimeout, Transport: http.DefaultTransport.(*http.Transport).Clone()}
	}
	maxClients := options.MaxClients
	if maxClients <= 0 {
		maxClients = defaultMaxClients
	}
	now := options.Now
	if now == nil {
		now = time.Now
	}
	return &Clients{
		source: source, http: httpClient, max: maxClients, now: now,
		clients: map[string]*list.Element{}, order: list.New(),
		known: map[resolution]resolved{}, refresh: map[string]time.Time{}, checks: map[string]checked{},
	}
}

// Source is what the clients resolve through.
func (c *Clients) Source() Source { return c.source }

// DeploymentApp is the deployment's own Stream app id, zero while it is not known or when
// the source has no deployment app at all.
func (c *Clients) DeploymentApp() int64 {
	if own, ok := c.source.(interface{ App() int64 }); ok {
		return own.App()
	}
	return 0
}

// Pin is the pin new work for a customer is given: the app it acts in, or zero for the
// deployment's own, or for a customer with no app to act in at all.
func (c *Clients) Pin(ctx context.Context, customer string) (int64, error) {
	bound, err := c.For(ctx, customer)
	if errors.Is(err, ErrNoIdentity) {
		return 0, nil
	}
	return bound.Identity.StreamApp, err
}

// HTTPClient is the transport every Stream client shares, for anything else that talks to
// Stream on the router's behalf, such as a voice edge.
func (c *Clients) HTTPClient() *http.Client { return c.http }

// For is the identity and client new work for a customer is done with.
func (c *Clients) For(ctx context.Context, customer string) (Bound, error) {
	identity, err := c.resolve(resolution{customer: customer}, func() (Identity, error) {
		return c.source.For(ctx, customer)
	})
	if err != nil {
		return Bound{}, err
	}
	return c.bind(identity)
}

// ForApp is the identity and client work pinned to an app is finished with.
func (c *Clients) ForApp(ctx context.Context, customer string, app int64) (Bound, error) {
	identity, err := c.resolve(resolution{customer: customer, pinned: true, app: app}, func() (Identity, error) {
		return c.source.ForApp(ctx, customer, app)
	})
	if err != nil {
		return Bound{}, err
	}
	return c.bind(identity)
}

// ForAppReading is the identity and client work pinned to an app is read back with. It
// differs from ForApp only for a source that keeps some work readable and no longer
// writable, which only something reading back what was written asks for.
func (c *Clients) ForAppReading(ctx context.Context, customer string, app int64) (Bound, error) {
	reader, ok := c.source.(interface {
		ForAppReading(context.Context, string, int64) (Identity, error)
	})
	if !ok {
		return c.ForApp(ctx, customer, app)
	}
	identity, err := c.resolve(resolution{customer: customer, pinned: true, reading: true, app: app}, func() (Identity, error) {
		return reader.ForAppReading(ctx, customer, app)
	})
	if err != nil {
		return Bound{}, err
	}
	return c.bind(identity)
}

// PerApp reports whether customers act in apps of their own, which is app mode.
func (c *Clients) PerApp() bool {
	perApp, ok := c.source.(interface{ PerApp() bool })
	return ok && perApp.PerApp()
}

// Invalidate forgets what was resolved for a customer, so the next use asks the source.
// It is called whenever a customer's app is written.
func (c *Clients) Invalidate(customer string) {
	c.mu.Lock()
	defer c.mu.Unlock()
	c.forget(customer)
}

func (c *Clients) forget(customer string) {
	for asked := range c.known {
		if asked.customer == customer {
			delete(c.known, asked)
		}
	}
}

// Rejected is told when Stream refused an identity's credential. It forgets what was
// resolved for the customer, at most once per credential per refreshEvery, and reports
// whether resolving again is worth a retry: a credential that was refused a moment ago and
// is refused again is wrong, not stale.
func (c *Clients) Rejected(identity Identity) bool {
	c.mu.Lock()
	defer c.mu.Unlock()
	fingerprint := identity.Fingerprint()
	if last, ok := c.refresh[fingerprint]; ok && c.now().Sub(last) < refreshEvery {
		return false
	}
	c.refresh[fingerprint] = c.now()
	c.forget(identity.CustomerID)
	return true
}

// resolve asks the source, or reuses an answer younger than identityTTL. Errors are kept
// too, for the same time, so a customer with nothing configured does not ask on every
// write.
func (c *Clients) resolve(asked resolution, ask func() (Identity, error)) (Identity, error) {
	c.mu.Lock()
	if answer, ok := c.known[asked]; ok && c.now().Sub(answer.at) < identityTTL {
		c.mu.Unlock()
		return answer.identity, answer.err
	}
	c.mu.Unlock()

	identity, err := ask()

	c.mu.Lock()
	defer c.mu.Unlock()
	c.known[asked] = resolved{identity: identity, err: err, at: c.now()}
	return identity, err
}

// bind is the client for an identity's credential, built the first time it is needed.
func (c *Clients) bind(identity Identity) (Bound, error) {
	key := identity.Fingerprint()

	c.mu.Lock()
	defer c.mu.Unlock()
	if element, ok := c.clients[key]; ok {
		c.order.MoveToFront(element)
		return Bound{Identity: identity, Client: element.Value.(*kept).client}, nil
	}

	baseURL := identity.BaseURL
	if baseURL == "" {
		baseURL = getstream.DefaultBaseURL
	}
	// The base URL is always passed, because the SDK otherwise reads STREAM_BASE_URL from
	// the process environment, which would make it the deployment's for every app.
	client, err := getstream.NewClient(identity.APIKey, identity.Secret.Reveal(),
		getstream.WithBaseUrl(baseURL), getstream.WithHTTPClient(c.http))
	if err != nil {
		return Bound{}, err
	}
	client.Chat()
	client.Video()

	c.clients[key] = c.order.PushFront(&kept{key: key, client: client})
	for c.order.Len() > c.max {
		oldest := c.order.Back()
		c.order.Remove(oldest)
		delete(c.clients, oldest.Value.(*kept).key)
	}
	return Bound{Identity: identity, Client: client}, nil
}
