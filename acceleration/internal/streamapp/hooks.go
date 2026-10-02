package streamapp

import (
	"context"
	"errors"
	"time"

	"github.com/GetStream/Vision-Agents/acceleration/internal/store"
)

const (
	// unknownKeyTTL is how long an api key no app holds is remembered as no app's, so a
	// stream of hooks naming one does not ask the database each time.
	unknownKeyTTL = 60 * time.Second
	// maxUnknownKeys bounds how many such keys are remembered.
	maxUnknownKeys = 1024
)

// Verifier is a secret a hook may be signed with, and the app a hook it verifies is from.
type Verifier struct {
	// CustomerID is the registered app's customer, empty for the deployment's own secret.
	CustomerID string
	// StreamApp is the app's id: the registered app's, or the deployment's, zero while
	// that is not known in deployment mode.
	StreamApp int64
	APIKey    string
	Secret    Secret
	// Deployment is the deployment's own secret, whose hooks are about work in its app.
	Deployment bool
}

// Verifiers are the secrets a hook may be checked against, in app mode. A hook naming an
// app is checked only against that app's keys, and the deployment's secret only when the
// app is the deployment's own. A hook naming no app is checked against the key it says it
// was signed with, when an app holds that key, and the deployment's secret otherwise. An
// api key never widens what a hook is checked against, and the deployment's secret is
// tried only for work in the deployment's app.
func (c *Clients) Verifiers(ctx context.Context, apiKey string, pathApp int64) ([]Verifier, error) {
	stored, ok := c.Stored()
	if !ok {
		return nil, errors.New("streamapp: hooks are checked per app only in app mode")
	}
	deployment := stored.deployment
	own := deployment.App()
	ours := func() []Verifier {
		if apiKey != "" && apiKey != deployment.identity.APIKey {
			return nil
		}
		return []Verifier{{StreamApp: own, APIKey: deployment.identity.APIKey, Secret: deployment.identity.Secret, Deployment: true}}
	}

	if pathApp != 0 {
		var verifiers []Verifier
		app, err := stored.store.StreamAppByPK(ctx, pathApp)
		switch {
		case err == nil:
			verifiers = stored.keysOf(app, apiKey)
		case !errors.Is(err, store.ErrNoStreamApp):
			return nil, err
		}
		if own == 0 && len(verifiers) == 0 {
			// The app named may be the deployment's own, which is not known yet.
			return nil, ErrDeploymentAppUnknown
		}
		if pathApp == own {
			verifiers = append(verifiers, ours()...)
		}
		return verifiers, nil
	}

	if apiKey != "" && !c.unknownKey(apiKey) {
		app, err := stored.store.StreamAppByAPIKey(ctx, apiKey)
		switch {
		case err == nil:
			return stored.keysOf(app, apiKey), nil
		case !errors.Is(err, store.ErrNoStreamApp):
			return nil, err
		}
		c.rememberUnknownKey(apiKey)
	}
	if own == 0 {
		return nil, ErrDeploymentAppUnknown
	}
	return []Verifier{{StreamApp: own, APIKey: deployment.identity.APIKey, Secret: deployment.identity.Secret, Deployment: true}}, nil
}

// keysOf are a connected app's keys Stream still accepts, as verifiers, narrowed to the
// one named when a key is named.
func (s *Stored) keysOf(app store.StreamApp, apiKey string) []Verifier {
	if app.State != store.StreamAppConnected {
		return nil
	}
	var verifiers []Verifier
	for _, key := range app.Keys {
		if key.Status != store.StreamAppKeyActive || (apiKey != "" && key.APIKey != apiKey) {
			continue
		}
		secret, _, err := OpenKey(s.sealer, app, key)
		if err != nil {
			s.logger.Warn("stream: could not open a key to check a hook with", "customer_id", app.CustomerID,
				"api_key", key.APIKey, "error", err)
			continue
		}
		verifiers = append(verifiers, Verifier{
			CustomerID: app.CustomerID, StreamApp: app.StreamAppPK, APIKey: key.APIKey, Secret: secret,
		})
	}
	return verifiers
}

func (c *Clients) unknownKey(apiKey string) bool {
	c.mu.Lock()
	defer c.mu.Unlock()
	at, ok := c.unknown[apiKey]
	return ok && c.now().Sub(at) < unknownKeyTTL
}

func (c *Clients) rememberUnknownKey(apiKey string) {
	c.mu.Lock()
	defer c.mu.Unlock()
	if len(c.unknown) >= maxUnknownKeys {
		for key, at := range c.unknown {
			if c.now().Sub(at) >= unknownKeyTTL {
				delete(c.unknown, key)
			}
		}
		if len(c.unknown) >= maxUnknownKeys {
			clear(c.unknown)
		}
	}
	c.unknown[apiKey] = c.now()
}
