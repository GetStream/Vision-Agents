package streamapp

import (
	"cmp"
	"context"
	"errors"
	"log/slog"
	"sync"
	"time"

	"github.com/GetStream/Vision-Agents/acceleration/internal/store"
)

const (
	// fallbackWarnEvery is how often the use of the fallback by one customer is logged.
	fallbackWarnEvery = time.Hour
	// fallbackRecordEvery is how often the use of the fallback by one customer is written.
	fallbackRecordEvery = time.Minute
)

// StoredOptions configures Stored. Store and Deployment are required.
type StoredOptions struct {
	// Store records the uses of the fallback.
	Store *store.Store
	// Deployment is the deployment's own app.
	Deployment *Deployment
	// FallbackToDeployment writes a customer with no token for an app of its own into the
	// deployment's, as deployment mode always did. Without it such a customer is written
	// nowhere.
	FallbackToDeployment bool
	// Forwarded acts in a customer's own app with the token the gateway forwarded for it.
	// Without it every customer but the deployment's own has only the fallback.
	Forwarded *Forwarded
	Logger    *slog.Logger
	// Now is the clock, for tests.
	Now func() time.Time
}

// Stored is app mode: each customer acts in its own app, with the server token the gateway
// forwarded for it.
//
// The deployment's own customer, whose id is the deployment app's, acts in it with the
// deployment's key. A customer with a token acts in its own app. Anybody else acts wherever
// the fallback says.
//
// Work already written is finished in the app it was pinned to. Work written into the
// deployment's own app before a customer had an app of its own can still be read there, and
// is written to only while the fallback allows it.
type Stored struct {
	store      *store.Store
	deployment *Deployment
	fallback   bool
	forwarded  *Forwarded
	logger     *slog.Logger
	now        func() time.Time

	mu        sync.Mutex
	warned    map[string]time.Time
	fallbacks map[string]*tally
	floor     Floor
}

// Floor reports whether a customer must act in a Stream app of its own, which keeps it out
// of the deployment's: never written there for want of one, and what it wrote there before
// only read. An error is taken to require it.
type Floor func(ctx context.Context, customer string) (bool, error)

// SetFloor says where to ask whether a customer must act in an app of its own.
func (s *Stored) SetFloor(floor Floor) {
	s.mu.Lock()
	defer s.mu.Unlock()
	s.floor = floor
}

// fallsBack reports whether a customer may be written into the deployment's app for want of
// an app of its own: only while the fallback allows it, and its policies do not forbid it.
func (s *Stored) fallsBack(ctx context.Context, customer string) bool {
	if !s.fallback {
		return false
	}
	s.mu.Lock()
	floor := s.floor
	s.mu.Unlock()
	if floor == nil {
		return true
	}
	required, err := floor(ctx, customer)
	if err != nil {
		s.logger.Warn("stream: could not read whether a customer must act in its own app, so it may not fall back",
			"customer_id", customer, "error", err)
		return false
	}
	return !required
}

// tally is the fallback uses of one customer not written down yet.
type tally struct {
	written time.Time
	pending int64
}

// NewStored returns app mode's Source.
func NewStored(options StoredOptions) (*Stored, error) {
	if options.Store == nil || options.Deployment == nil {
		return nil, errors.New("streamapp: app mode needs the store and the deployment's own app")
	}
	now := options.Now
	if now == nil {
		now = time.Now
	}
	return &Stored{
		store: options.Store, deployment: options.Deployment,
		fallback: options.FallbackToDeployment,
		logger:   cmp.Or(options.Logger, slog.Default()),
		now:      now,
		warned:   map[string]time.Time{}, fallbacks: map[string]*tally{},
		forwarded: options.Forwarded,
	}, nil
}

// App is the deployment app's id, zero while it is not known.
func (s *Stored) App() int64 { return s.deployment.App() }

func (s *Stored) knowable() bool { return s.deployment.knowable() }

// PerApp says customers act in apps of their own.
func (s *Stored) PerApp() bool { return true }

func (s *Stored) learn(ctx context.Context, clients *Clients) (int64, error) {
	return s.deployment.learn(ctx, clients)
}

// For is the identity new work for a customer is done with.
func (s *Stored) For(ctx context.Context, customer string) (Identity, error) {
	deployment := s.deployment.App()
	if s.ownsDeploymentApp(customer, deployment) {
		return s.inDeploymentApp(ctx, customer, deployment)
	}
	if s.waitsFor(customer) {
		// Until the deployment's own app is known, this may be its customer.
		return Identity{}, ErrDeploymentAppUnknown
	}
	if identity, ok, err := s.inForwardedApp(ctx, customer); ok {
		return identity, err
	}
	if !s.fallsBack(ctx, customer) {
		return Identity{}, ErrNoIdentity
	}
	if deployment == 0 {
		// Falling back names the deployment's app, which has to be known to be named.
		return Identity{}, ErrDeploymentAppUnknown
	}
	s.fellBack(ctx, customer)
	return s.inDeploymentApp(ctx, customer, deployment)
}

// ForApp is the identity work pinned to an app is finished with. Another customer's work
// in the deployment's own app is ErrReadOnly once the fallback no longer allows writing
// there.
func (s *Stored) ForApp(ctx context.Context, customer string, app int64) (Identity, error) {
	identity, readOnly, err := s.forApp(ctx, customer, app)
	if err == nil && readOnly {
		return Identity{}, ErrReadOnly
	}
	return identity, err
}

// ForAppReading is the identity work pinned to an app is read back with, which is the same
// as ForApp except that work kept in the deployment's own app can always be read.
func (s *Stored) ForAppReading(ctx context.Context, customer string, app int64) (Identity, error) {
	identity, _, err := s.forApp(ctx, customer, app)
	return identity, err
}

func (s *Stored) forApp(ctx context.Context, customer string, app int64) (Identity, bool, error) {
	if app < 0 {
		return Identity{}, false, ErrStreamAppMoved
	}
	deployment := s.deployment.App()
	if app == 0 || app == deployment {
		return s.legacy(ctx, customer, app, deployment)
	}
	if customer == CustomerOf(app) {
		if identity, ok, err := s.inForwardedApp(ctx, customer); ok {
			return identity, false, err
		}
	}
	if s.deployment.pending() {
		// The pin may be the deployment's own app, which is not known yet.
		return Identity{}, false, ErrDeploymentAppUnknown
	}
	return Identity{}, false, ErrStreamAppMoved
}

// inForwardedApp is a customer's own app acted in with the token the gateway forwarded for
// it, and false when there is no such token to act with.
func (s *Stored) inForwardedApp(ctx context.Context, customer string) (Identity, bool, error) {
	app, couldBe := ParseAppID(customer)
	if s.forwarded == nil || !couldBe {
		return Identity{}, false, nil
	}
	identity, err := s.forwarded.identity(ctx, customer, app)
	if errors.Is(err, ErrNoIdentity) {
		return Identity{}, false, nil
	}
	return identity, true, err
}

// legacy is work written into the deployment's own app: unpinned, from before apps had
// identities or from deployment mode, or pinned to the deployment app by id. Its customer's
// own work is finished there; anybody else's is written to only while the fallback allows.
func (s *Stored) legacy(ctx context.Context, customer string, app, deployment int64) (Identity, bool, error) {
	if s.ownsDeploymentApp(customer, deployment) || s.fallsBack(ctx, customer) {
		identity, err := s.inDeploymentApp(ctx, customer, app)
		return identity, false, err
	}
	if s.waitsFor(customer) {
		return Identity{}, false, ErrDeploymentAppUnknown
	}
	identity, err := s.inDeploymentApp(ctx, customer, app)
	return identity, true, err
}

// ownsDeploymentApp reports whether a customer is the deployment app's own.
func (s *Stored) ownsDeploymentApp(customer string, deployment int64) bool {
	return deployment != 0 && customer == CustomerOf(deployment)
}

// waitsFor reports whether a customer's work waits for the deployment's own app to be known,
// because the customer may be that app's: only while that app is pending and the
// customer's id could be an app's at all.
func (s *Stored) waitsFor(customer string) bool {
	_, couldBe := ParseAppID(customer)
	return s.deployment.pending() && couldBe
}

func (s *Stored) inDeploymentApp(ctx context.Context, customer string, app int64) (Identity, error) {
	identity, err := s.deployment.For(ctx, customer)
	if err != nil {
		return Identity{}, err
	}
	identity.StreamApp = app
	return identity, nil
}

// fellBack notes that a customer was written into the deployment's app for want of one of
// its own: logged once an hour, and written down at most once a minute.
func (s *Stored) fellBack(ctx context.Context, customer string) {
	now := s.now()
	s.mu.Lock()
	warn := now.Sub(s.warned[customer]) >= fallbackWarnEvery
	if warn {
		s.warned[customer] = now
	}
	counted := s.fallbacks[customer]
	if counted == nil {
		counted = &tally{}
		s.fallbacks[customer] = counted
	}
	counted.pending++
	var uses int64
	if now.Sub(counted.written) >= fallbackRecordEvery {
		uses, counted.pending, counted.written = counted.pending, 0, now
	}
	s.mu.Unlock()

	if warn {
		s.logger.Warn("stream: a customer with no Stream app of its own is written into the deployment's app",
			"customer_id", customer)
	}
	if uses > 0 {
		if err := s.store.RecordStreamFallbackUses(ctx, customer, uses, now); err != nil {
			s.logger.Warn("stream: could not record a use of the fallback", "customer_id", customer, "error", err)
		}
	}
}
