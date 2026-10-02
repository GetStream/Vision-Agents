package streamapp

import (
	"context"
	"sync/atomic"
)

// DeploymentOptions is the router's own Stream app, as its environment names it.
type DeploymentOptions struct {
	APIKey    string
	Secret    string
	UserToken string
	BaseURL   string
	// App is the app's id when the deployment says what it is. Zero is learned later,
	// with SetApp, or never.
	App int64
}

// Deployment is the router's own Stream app: the pair in its environment, which is the
// app every Stream action was taken in before customers had apps of their own. In
// deployment mode it answers for every customer.
type Deployment struct {
	identity Identity
	app      atomic.Int64
}

// NewDeployment returns the deployment's own app as a Source.
func NewDeployment(options DeploymentOptions) *Deployment {
	d := &Deployment{identity: Identity{
		APIKey:    options.APIKey,
		Secret:    NewSecret(options.Secret),
		UserToken: options.UserToken,
		BaseURL:   options.BaseURL,
	}}
	d.app.Store(options.App)
	return d
}

// Configured reports whether the environment names an app at all. A router without one
// still starts; it just has nowhere in Stream to write.
func (d *Deployment) Configured() bool {
	return d.identity.APIKey != "" && !d.identity.Secret.Empty()
}

// App is the deployment app's id, zero while it is not known.
func (d *Deployment) App() int64 { return d.app.Load() }

// SetApp records the deployment app's id once it has been learned.
func (d *Deployment) SetApp(app int64) { d.app.Store(app) }

// For answers every customer with the deployment's own app. The pin is zero: in
// deployment mode nothing records an app, which is how everything written before apps had
// identities reads too.
func (d *Deployment) For(_ context.Context, customer string) (Identity, error) {
	if !d.Configured() {
		return Identity{}, ErrNoIdentity
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
	known := d.app.Load()
	switch {
	case known == 0:
		return Identity{}, ErrDeploymentAppUnknown
	case known != app:
		return Identity{}, ErrStreamAppMoved
	}
	return d.For(ctx, customer)
}
