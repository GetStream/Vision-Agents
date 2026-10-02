package streamapp

import (
	"context"
	"fmt"
	"sync/atomic"
)

// DeploymentOptions is the router's own Stream app, as its environment names it.
type DeploymentOptions struct {
	APIKey    string
	Secret    string
	UserToken string
	BaseURL   string
	// App is the app's id when the deployment says what it is. Zero is learned later,
	// with SetApp, or never. One that is set is checked against Stream when it can be.
	App int64
	// Strict stops the deployment answering at all once Stream says its key belongs to
	// another app than App, which is what app mode needs: every pin it writes names an
	// app, and one named wrongly would be finished with the wrong key. Without it only
	// the pins stop matching, and new work carries on as it always has.
	Strict bool
}

// Deployment is the router's own Stream app: the pair in its environment, which is the
// app every Stream action was taken in before customers had apps of their own. In
// deployment mode it answers for every customer.
type Deployment struct {
	identity   Identity
	app        atomic.Int64
	configured int64
	strict     bool
	verified   atomic.Bool
	wrong      atomic.Bool
}

// NewDeployment returns the deployment's own app as a Source.
func NewDeployment(options DeploymentOptions) *Deployment {
	d := &Deployment{identity: Identity{
		APIKey:    options.APIKey,
		Secret:    NewSecret(options.Secret),
		UserToken: options.UserToken,
		BaseURL:   options.BaseURL,
	}, configured: options.App, strict: options.Strict}
	d.app.Store(options.App)
	return d
}

// Configured reports whether the environment names an app at all. A router without one
// still starts; it just has nowhere in Stream to write.
func (d *Deployment) Configured() bool {
	return d.identity.APIKey != "" && !d.identity.Secret.Empty()
}

// App is the deployment app's id, zero while it is not known, or once the id it was
// given turned out to be another app's.
func (d *Deployment) App() int64 {
	if d.wrong.Load() {
		return 0
	}
	return d.app.Load()
}

// SetApp records the deployment app's id once it has been learned.
func (d *Deployment) SetApp(app int64) { d.app.Store(app) }

// For answers every customer with the deployment's own app. The pin is zero: in
// deployment mode nothing records an app, which is how everything written before apps had
// identities reads too.
func (d *Deployment) For(_ context.Context, customer string) (Identity, error) {
	if !d.Configured() {
		return Identity{}, ErrNoIdentity
	}
	if d.strict && d.wrong.Load() {
		return Identity{}, ErrDeploymentAppMismatch
	}
	identity := d.identity
	identity.CustomerID = customer
	return identity, nil
}

// ForApp finishes work in the deployment app when that is where it was written: no pin,
// or the deployment app's own id. A pin naming any other app is somebody else's to finish,
// and a pin that cannot be compared yet waits.
func (d *Deployment) ForApp(ctx context.Context, customer string, app int64) (Identity, error) {
	if app == 0 {
		return d.For(ctx, customer)
	}
	// A negative pin is one an import could not vouch for, and names no app at all.
	if app < 0 {
		return Identity{}, ErrStreamAppMoved
	}
	known := d.App()
	switch {
	case known == 0:
		return Identity{}, ErrDeploymentAppUnknown
	case known != app:
		return Identity{}, ErrStreamAppMoved
	}
	return d.For(ctx, customer)
}

// verifyOrLearn checks the configured id against the app Stream says the key belongs to,
// or learns the id when none was configured. A mismatch is remembered: the configured id
// names no pin from then on, and in strict mode the deployment stops answering.
func (d *Deployment) verifyOrLearn(actual int64) (int64, error) {
	if d.configured == 0 {
		d.SetApp(actual)
		return actual, nil
	}
	if actual != d.configured {
		d.wrong.Store(true)
		return 0, fmt.Errorf("%w: stream.app_id is %d, and the deployment's key belongs to app %d",
			ErrDeploymentAppMismatch, d.configured, actual)
	}
	d.verified.Store(true)
	return actual, nil
}

// settled reports the deployment app's id when nothing is left to ask Stream about it.
func (d *Deployment) settled() (int64, bool, error) {
	switch {
	case d.wrong.Load():
		return 0, true, ErrDeploymentAppMismatch
	case d.configured != 0 && d.verified.Load():
		return d.configured, true, nil
	case d.configured == 0 && d.app.Load() != 0:
		return d.app.Load(), true, nil
	}
	return 0, false, nil
}

// knowable reports whether the deployment app's id is known, or can still be learned: the
// deployment has a key to ask with, and the id it was given has not turned out wrong.
func (d *Deployment) knowable() bool {
	return d.Configured() && !d.wrong.Load()
}
