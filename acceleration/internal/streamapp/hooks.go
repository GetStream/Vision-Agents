package streamapp

import (
	"context"
	"errors"
	"time"

	"github.com/GetStream/Vision-Agents/acceleration/internal/store"
)

// maxHookAsks bounds how many answers about hooks are kept.
const maxHookAsks = 1024

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
//
// What is found is kept as an identity is, and forgotten whenever an app is written, so a
// busy app's hooks do not each read its keys and open them again.
func (c *Clients) Verifiers(ctx context.Context, apiKey string, pathApp int64) ([]Verifier, error) {
	asked := hookAsk{apiKey: apiKey, pathApp: pathApp}
	c.mu.Lock()
	held, ok := c.hookKeys[asked]
	before := c.hookGeneration
	c.mu.Unlock()
	if ok && c.now().Sub(held.at) < identityTTL {
		return held.verifiers, nil
	}
	verifiers, err := c.verifiers(ctx, apiKey, pathApp)
	if err != nil {
		return nil, err
	}
	c.mu.Lock()
	defer c.mu.Unlock()
	// An app written while its keys were being read may have dropped one of them, which is
	// then read again next time rather than kept.
	if c.hookGeneration != before {
		return verifiers, nil
	}
	// Anybody can name a key or an app on a hook, so what is kept is bounded.
	if len(c.hookKeys) >= maxHookAsks {
		clear(c.hookKeys)
	}
	c.hookKeys[asked] = heldVerifiers{verifiers: verifiers, at: c.now()}
	return verifiers, nil
}

// hookAsk is one question about a hook: the key it named and the app its path named.
type hookAsk struct {
	apiKey  string
	pathApp int64
}

// heldVerifiers is an answer to one, as of when it was found.
type heldVerifiers struct {
	verifiers []Verifier
	at        time.Time
}

func (c *Clients) verifiers(ctx context.Context, apiKey string, pathApp int64) ([]Verifier, error) {
	stored, ok := c.Stored()
	if !ok {
		return nil, errors.New("streamapp: hooks are checked per app only in app mode")
	}
	deployment := stored.deployment
	own := deployment.App()
	// The deployment's own secret checks only the deployment app's hooks, and only when
	// there is one: an empty secret is one anybody can sign with.
	ours := func(named string) []Verifier {
		if own == 0 || !deployment.Configured() || (named != "" && named != deployment.identity.APIKey) {
			return nil
		}
		return []Verifier{{StreamApp: own, APIKey: deployment.identity.APIKey, Secret: deployment.identity.Secret, Deployment: true}}
	}

	if pathApp != 0 {
		var verifiers []Verifier
		app, err := stored.store.StreamAppByPK(ctx, pathApp)
		switch {
		case err == nil:
			verifiers = stored.keysOf(app, apiKey, own)
		case !errors.Is(err, store.ErrNoStreamApp):
			return nil, err
		}
		if len(verifiers) == 0 && deployment.pending() {
			// The app named may be the deployment's own, which is not known yet.
			return nil, ErrDeploymentAppUnknown
		}
		if pathApp == own {
			verifiers = append(verifiers, ours(apiKey)...)
		}
		return verifiers, nil
	}

	if apiKey != "" {
		app, err := stored.store.StreamAppByAPIKey(ctx, apiKey)
		switch {
		case err == nil:
			return stored.keysOf(app, apiKey, own), nil
		case !errors.Is(err, store.ErrNoStreamApp):
			return nil, err
		}
	}
	if deployment.pending() {
		return nil, ErrDeploymentAppUnknown
	}
	return ours(""), nil
}

// keysOf are a connected app's keys Stream still accepts, as verifiers, narrowed to the
// one named when a key is named. The deployment's own app, registered by its own customer,
// is still the deployment's: its hooks are about everything written there, which is more
// than that customer's.
func (s *Stored) keysOf(app store.StreamApp, apiKey string, own int64) []Verifier {
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
		verifier := Verifier{CustomerID: app.CustomerID, StreamApp: app.StreamAppPK, APIKey: key.APIKey, Secret: secret}
		if own != 0 && app.StreamAppPK == own {
			verifier.CustomerID, verifier.Deployment = "", true
		}
		verifiers = append(verifiers, verifier)
	}
	return verifiers
}
