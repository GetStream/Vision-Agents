package streamapp

import (
	"cmp"
	"context"
	"errors"
	"fmt"
	"log/slog"
	"strconv"
	"sync"
	"time"

	"github.com/GetStream/Vision-Agents/acceleration/internal/auth"
	"github.com/GetStream/Vision-Agents/acceleration/internal/store"
)

const (
	// fallbackWarnEvery is how often the use of the fallback by one customer is logged.
	fallbackWarnEvery = time.Hour
	// fallbackRecordEvery is how often the use of the fallback by one customer is written.
	fallbackRecordEvery = time.Minute
)

// StoredOptions configures Stored. Store, Sealer and Deployment are required.
type StoredOptions struct {
	// Store holds the apps customers registered and their sealed keys.
	Store *store.Store
	// Sealer opens those keys.
	Sealer *auth.Sealer
	// Deployment is the deployment's own app.
	Deployment *Deployment
	// FallbackToDeployment writes a customer that registered no app into the deployment's
	// own, as deployment mode always did. Without it such a customer is written nowhere.
	FallbackToDeployment bool
	Logger               *slog.Logger
	// Now is the clock, for tests.
	Now func() time.Time
}

// Stored is app mode: each customer acts in the app it registered, with its own keys.
//
// A customer with a registered app acts there whatever state the app is in, and never in
// the deployment's: a disconnected or blocked app is written nowhere. The deployment's own
// customer, whose id is the deployment app's, acts in it with the deployment's key. Anybody
// else acts wherever the fallback says.
//
// Work already written is finished in the app it was pinned to. Work written into the
// deployment's own app before a customer had an app of its own can still be read there, and
// is written to only while the fallback allows it.
type Stored struct {
	store      *store.Store
	sealer     *auth.Sealer
	deployment *Deployment
	fallback   bool
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
	if options.Store == nil || options.Sealer == nil || options.Deployment == nil {
		return nil, errors.New("streamapp: app mode needs the store, the keyring and the deployment's own app")
	}
	now := options.Now
	if now == nil {
		now = time.Now
	}
	return &Stored{
		store: options.Store, sealer: options.Sealer, deployment: options.Deployment,
		fallback: options.FallbackToDeployment,
		logger:   cmp.Or(options.Logger, slog.Default()),
		now:      now,
		warned:   map[string]time.Time{}, fallbacks: map[string]*tally{},
	}, nil
}

// App is the deployment app's id, zero while it is not known.
func (s *Stored) App() int64 { return s.deployment.App() }

// PerApp says customers act in apps of their own.
func (s *Stored) PerApp() bool { return true }

func (s *Stored) learn(ctx context.Context, clients *Clients) (int64, error) {
	return s.deployment.learn(ctx, clients)
}

// For is the identity new work for a customer is done with.
func (s *Stored) For(ctx context.Context, customer string) (Identity, error) {
	registered, err := s.store.StreamApp(ctx, customer)
	if err == nil {
		return s.registered(ctx, registered)
	}
	if !errors.Is(err, store.ErrNoStreamApp) {
		return Identity{}, err
	}
	deployment := s.deployment.App()
	if s.ownsDeploymentApp(customer, deployment) {
		return s.inDeploymentApp(ctx, customer, deployment)
	}
	if deployment == 0 && couldBeAnApp(customer) {
		// Until the deployment's own app is known, this may be its customer.
		return Identity{}, ErrDeploymentAppUnknown
	}
	if !s.fallsBack(ctx, customer) {
		return Identity{}, ErrNoIdentity
	}
	if deployment == 0 {
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
	registered, err := s.store.StreamApp(ctx, customer)
	switch {
	case err == nil && registered.StreamAppPK == app:
		identity, err := s.registered(ctx, registered)
		return identity, false, err
	case err != nil && !errors.Is(err, store.ErrNoStreamApp):
		return Identity{}, false, err
	case deployment == 0:
		// The pin may be the deployment's own app, which is not known yet.
		return Identity{}, false, ErrDeploymentAppUnknown
	}
	return Identity{}, false, ErrStreamAppMoved
}

// legacy is work written into the deployment's own app: unpinned, from before apps had
// identities or from deployment mode, or pinned to the deployment app by id. Its customer's
// own work is finished there; anybody else's is written to only while the fallback allows.
func (s *Stored) legacy(ctx context.Context, customer string, app, deployment int64) (Identity, bool, error) {
	if s.ownsDeploymentApp(customer, deployment) || s.fallsBack(ctx, customer) {
		identity, err := s.inDeploymentApp(ctx, customer, app)
		return identity, false, err
	}
	if deployment == 0 && couldBeAnApp(customer) {
		return Identity{}, false, ErrDeploymentAppUnknown
	}
	identity, err := s.inDeploymentApp(ctx, customer, app)
	return identity, true, err
}

// ownsDeploymentApp reports whether a customer is the deployment app's own, which is the
// app's id written exactly as Stream writes it.
func (s *Stored) ownsDeploymentApp(customer string, deployment int64) bool {
	return deployment != 0 && customer == strconv.FormatInt(deployment, 10)
}

// couldBeAnApp reports whether a customer id could be a Stream app's id at all.
func couldBeAnApp(customer string) bool {
	id, err := strconv.ParseInt(customer, 10, 64)
	return err == nil && id > 0 && strconv.FormatInt(id, 10) == customer
}

func (s *Stored) inDeploymentApp(ctx context.Context, customer string, app int64) (Identity, error) {
	identity, err := s.deployment.For(ctx, customer)
	if err != nil {
		return Identity{}, err
	}
	identity.StreamApp = app
	return identity, nil
}

// registered is the identity a registered app is acted in with: its primary key while
// Stream accepts it, its oldest accepted key otherwise.
func (s *Stored) registered(ctx context.Context, app store.StreamApp) (Identity, error) {
	if app.State != store.StreamAppConnected {
		return Identity{}, ErrStreamAppDisconnected
	}
	key, ok := activeKey(app)
	if !ok {
		return Identity{}, ErrStreamAppDisconnected
	}
	secret, stale, err := OpenKey(s.sealer, app, key)
	if err != nil {
		return Identity{}, fmt.Errorf("streamapp: opening key %s of the app registered for %s: %w", key.APIKey, app.CustomerID, err)
	}
	if stale {
		s.rewrap(ctx, app, key, secret)
	}
	return Identity{
		CustomerID: app.CustomerID, StreamApp: app.StreamAppPK, APIKey: key.APIKey, Secret: secret,
		BaseURL: s.deployment.identity.BaseURL, Registered: true, AllowGuests: app.AllowGuests,
	}, nil
}

func activeKey(app store.StreamApp) (store.StreamAppKey, bool) {
	if primary, ok := app.Key(app.PrimaryKey); ok && primary.Status == store.StreamAppKeyActive {
		return primary, true
	}
	for _, key := range app.Keys {
		if key.Status == store.StreamAppKeyActive {
			return key, true
		}
	}
	return store.StreamAppKey{}, false
}

// rewrap seals a key again under the keyring's current version. It is a chore, not a
// condition of using the key: a failure leaves the old seal, which still opens.
func (s *Stored) rewrap(ctx context.Context, app store.StreamApp, key store.StreamAppKey, secret Secret) {
	sealed, err := SealKey(s.sealer, app.CustomerID, app.StreamAppPK, key.APIKey, secret.Reveal())
	if err == nil {
		_, err = s.store.RewrapStreamAppKey(ctx, app.CustomerID, key.APIKey, key.Sealed, sealed.Sealed, sealed.KEKVersion)
	}
	if err != nil {
		s.logger.Warn("stream: could not seal a key again under the current key version",
			"customer_id", app.CustomerID, "api_key", key.APIKey, "error", err)
	}
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
