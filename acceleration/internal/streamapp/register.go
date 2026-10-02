package streamapp

import (
	"context"
	"errors"
	"fmt"
	"slices"
	"strconv"
	"time"

	getstream "github.com/GetStream/getstream-go/v5"

	"github.com/GetStream/Vision-Agents/acceleration/internal/store"
)

// ErrKeyRefused is a key Stream does not accept, or accepts for an app that may not be
// registered. Its message names the key and never its secret.
var ErrKeyRefused = errors.New("streamapp: Stream refused the key")

// Key is a key and its secret, as a registration hands them over.
type Key struct {
	APIKey string
	Secret Secret
	// CreatedAt is when Stream says the key was made, which orders keys by age. Zero is
	// not known.
	CreatedAt time.Time
}

// Registration is a customer's app and every key it now holds.
type Registration struct {
	CustomerID     string
	OrganizationID string
	Keys           []Key
	// PrimaryKey mints tokens. Empty is the first key.
	PrimaryKey  string
	AllowGuests bool
	// ExpectedRevision is the revision the writer read, zero for an app never registered,
	// nil to write whatever is there.
	ExpectedRevision *int64
	UpdatedBy        string
	// StreamApp is the app's id, which only the operator's command line may name for an
	// app whose id is not its customer's. Zero takes the id from Stream, which must then
	// be the customer's.
	StreamApp int64
}

// Registered is what a registration changed.
type Registered struct {
	App store.StreamApp
	// Dropped are the keys the app held before and no longer does.
	Dropped []string
	// Previous is the app's id before, zero for one never registered.
	Previous int64
}

// Verify asks Stream which app a key belongs to and how that app stands. The client it
// asks with is made for the question and kept nowhere.
func (s *Stored) Verify(ctx context.Context, key Key) (Readiness, error) {
	baseURL := s.deployment.identity.BaseURL
	if baseURL == "" {
		baseURL = getstream.DefaultBaseURL
	}
	client, err := getstream.NewClient(key.APIKey, key.Secret.Reveal(), getstream.WithBaseUrl(baseURL))
	if err != nil {
		return Readiness{}, fmt.Errorf("%w: key %s", ErrKeyRefused, key.APIKey)
	}
	readiness, err := ReadReadiness(ctx, client, s.now())
	if err != nil {
		var refused *getstream.StreamError
		if errors.As(err, &refused) && refused.StatusCode >= 400 && refused.StatusCode < 500 {
			return Readiness{}, fmt.Errorf("%w: Stream answered %d to key %s", ErrKeyRefused, refused.StatusCode, key.APIKey)
		}
		return Readiness{}, fmt.Errorf("streamapp: could not ask Stream about key %s: %w", key.APIKey, err)
	}
	return readiness, nil
}

// Register verifies every key with Stream and keeps them, sealed, as the customer's app.
// Each must belong to the same app, the customer's own unless the operator named one, and
// that app must neither be suspended nor take requests without checking their tokens. The
// deployment's own app is only ever its own customer's.
func (s *Stored) Register(ctx context.Context, registration Registration) (Registered, error) {
	if len(registration.Keys) == 0 {
		return Registered{}, errors.New("streamapp: registering needs at least one key; disconnect to hold none")
	}
	app := registration.StreamApp
	for _, key := range registration.Keys {
		readiness, err := s.Verify(ctx, key)
		if err != nil {
			return Registered{}, err
		}
		if err := s.admissible(registration.CustomerID, app, key.APIKey, readiness); err != nil {
			return Registered{}, err
		}
		if app == 0 {
			app = readiness.App
		}
	}

	sealed := make([]store.StreamAppKey, 0, len(registration.Keys))
	for _, key := range registration.Keys {
		one, err := SealKey(s.sealer, registration.CustomerID, app, key.APIKey, key.Secret.Reveal())
		if err != nil {
			return Registered{}, err
		}
		if !key.CreatedAt.IsZero() {
			created := key.CreatedAt.UTC()
			one.KeyCreatedAt = &created
		}
		sealed = append(sealed, one)
	}
	primary := registration.PrimaryKey
	if primary == "" {
		primary = registration.Keys[0].APIKey
	}

	before, err := s.store.StreamApp(ctx, registration.CustomerID)
	if err != nil && !errors.Is(err, store.ErrNoStreamApp) {
		return Registered{}, err
	}
	stored, err := s.store.PutStreamApp(ctx, store.StreamAppRegistration{
		CustomerID: registration.CustomerID, OrganizationID: registration.OrganizationID, StreamAppPK: app,
		Keys: sealed, PrimaryKey: primary, AllowGuests: registration.AllowGuests,
		ExpectedRevision: registration.ExpectedRevision, UpdatedBy: registration.UpdatedBy, VerifiedAt: s.now(),
	})
	if err != nil {
		return Registered{}, err
	}
	return Registered{App: stored, Dropped: dropped(before, stored), Previous: before.StreamAppPK}, nil
}

// admissible refuses a key whose app may not be registered by this customer.
func (s *Stored) admissible(customer string, named int64, apiKey string, readiness Readiness) error {
	switch {
	case named == 0 && strconv.FormatInt(readiness.App, 10) != customer:
		return fmt.Errorf("%w: key %s belongs to another Stream app than this one", ErrKeyRefused, apiKey)
	case named != 0 && readiness.App != named:
		return fmt.Errorf("%w: key %s belongs to another Stream app than the one named", ErrKeyRefused, apiKey)
	case readiness.Suspended:
		return fmt.Errorf("%w: the Stream app key %s belongs to is suspended", ErrKeyRefused, apiKey)
	case readiness.AuthChecksOff:
		return fmt.Errorf("%w: the Stream app key %s belongs to does not check the tokens it is sent", ErrKeyRefused, apiKey)
	}
	deployment := s.deployment.App()
	if deployment != 0 && readiness.App == deployment && customer != strconv.FormatInt(deployment, 10) {
		return fmt.Errorf("%w: key %s belongs to the router's own Stream app, which is only its own customer's", ErrKeyRefused, apiKey)
	}
	return nil
}

// Disconnect deletes every key of a customer's app once a key of that app proves the
// caller holds it, and leaves the app behind with none. It reports the app's id.
func (s *Stored) Disconnect(ctx context.Context, customer string, proof Key, expected *int64, updatedBy string) (store.StreamApp, error) {
	current, err := s.store.StreamApp(ctx, customer)
	if err != nil {
		return store.StreamApp{}, err
	}
	readiness, err := s.Verify(ctx, proof)
	if err != nil {
		return store.StreamApp{}, err
	}
	if readiness.App != current.StreamAppPK {
		return store.StreamApp{}, fmt.Errorf("%w: key %s does not belong to the app being disconnected", ErrKeyRefused, proof.APIKey)
	}
	return s.store.DisconnectStreamApp(ctx, customer, expected, updatedBy)
}

// ForKey is the identity a customer's app is acted in with one of its keys rather than its
// primary, when that key is the app's and Stream still accepts it. Anything else is the
// identity given.
func (s *Stored) ForKey(ctx context.Context, identity Identity, apiKey string) Identity {
	if !identity.Registered || apiKey == "" || apiKey == identity.APIKey {
		return identity
	}
	app, err := s.store.StreamApp(ctx, identity.CustomerID)
	if err != nil || app.State != store.StreamAppConnected || app.StreamAppPK != identity.StreamApp {
		return identity
	}
	key, ok := app.Key(apiKey)
	if !ok || key.Status != store.StreamAppKeyActive {
		return identity
	}
	secret, _, err := OpenKey(s.sealer, app, key)
	if err != nil {
		return identity
	}
	identity.APIKey, identity.Secret = key.APIKey, secret
	return identity
}

// Rewrap seals every key under an older key version again under the current one, and
// reports how many it sealed again.
func (s *Stored) Rewrap(ctx context.Context) (int, error) {
	apps, err := s.store.ConnectedStreamApps(ctx)
	if err != nil {
		return 0, err
	}
	rewrapped := 0
	for _, listed := range apps {
		app, err := s.store.StreamApp(ctx, listed.CustomerID)
		if err != nil {
			return rewrapped, err
		}
		for _, key := range app.Keys {
			secret, stale, err := OpenKey(s.sealer, app, key)
			if err != nil {
				return rewrapped, fmt.Errorf("streamapp: opening key %s of %s: %w", key.APIKey, app.CustomerID, err)
			}
			if !stale {
				continue
			}
			sealed, err := SealKey(s.sealer, app.CustomerID, app.StreamAppPK, key.APIKey, secret.Reveal())
			if err != nil {
				return rewrapped, err
			}
			applied, err := s.store.RewrapStreamAppKey(ctx, app.CustomerID, key.APIKey, key.Sealed, sealed.Sealed, sealed.KEKVersion)
			if err != nil {
				return rewrapped, err
			}
			if applied {
				rewrapped++
			}
		}
	}
	return rewrapped, nil
}

// CheckApp checks one customer's app as the periodic check does.
func (s *Stored) CheckApp(ctx context.Context, clients *Clients, customer string, ended Ended) error {
	app, err := s.store.StreamApp(ctx, customer)
	if err != nil {
		return err
	}
	if app.State != store.StreamAppConnected {
		return ErrStreamAppDisconnected
	}
	s.check(ctx, clients, app, ended)
	return nil
}

func dropped(before, after store.StreamApp) []string {
	var gone []string
	for _, key := range before.Keys {
		if !slices.ContainsFunc(after.Keys, func(kept store.StreamAppKey) bool { return kept.APIKey == key.APIKey }) {
			gone = append(gone, key.APIKey)
		}
	}
	return gone
}
