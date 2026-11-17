package streamapp

import (
	"context"
	"errors"
	"time"
)

// maxHookAsks bounds how many answers about hooks are kept.
const maxHookAsks = 1024

// Verifier is a secret a hook may be signed with, and the app a hook it verifies is from.
type Verifier struct {
	// StreamApp is the deployment app's id, zero while that is not known in deployment mode.
	StreamApp int64
	APIKey    string
	Secret    Secret
	// Deployment is the deployment's own secret, whose hooks are about work in its app.
	Deployment bool
}

// Verifiers are the secrets a hook may be checked against, in app mode: only ever the
// deployment's own, for a hook about work in the deployment's app. A customer's own app
// signs its hooks with a secret the router does not hold, so those are vouched for by the
// gateway in front instead. A hook naming the deployment's app is checked only against the
// key it names, when it names one.
//
// What is found is kept as an identity is, so a busy app's hooks do not each ask again.
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

	if deployment.pending() {
		// The app named may be the deployment's own, which is not known yet.
		return nil, ErrDeploymentAppUnknown
	}
	switch pathApp {
	case 0:
		return ours(""), nil
	case own:
		return ours(apiKey), nil
	}
	return nil, nil
}
