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
	identity Identity
	// configured is the id the deployment was given, zero for none.
	configured int64
	// learned is the id Stream says the deployment's key belongs to, zero until asked.
	learned atomic.Int64
	strict  bool
}

// NewDeployment returns the deployment's own app as a Source.
func NewDeployment(options DeploymentOptions) *Deployment {
	return &Deployment{identity: Identity{
		APIKey:    options.APIKey,
		Secret:    NewSecret(options.Secret),
		UserToken: options.UserToken,
		BaseURL:   options.BaseURL,
	}, configured: options.App, strict: options.Strict}
}

// Configured reports whether the environment names an app at all. A router without one
// still starts; it just has nowhere in Stream to write.
func (d *Deployment) Configured() bool {
	return d.identity.APIKey != "" && !d.identity.Secret.Empty()
}

// App is the deployment app's id: the one it was given, until Stream says otherwise, or
// the one Stream gave. Zero is not known, or given wrongly.
func (d *Deployment) App() int64 {
	switch learned := d.learned.Load(); {
	case d.wrong():
		return 0
	case d.configured != 0:
		return d.configured
	default:
		return learned
	}
}

// SetApp records the deployment app's id, as Stream gave it.
func (d *Deployment) SetApp(app int64) { d.learned.Store(app) }

// wrong reports whether the id the deployment was given is not the app its key belongs to.
func (d *Deployment) wrong() bool {
	learned := d.learned.Load()
	return d.configured != 0 && learned != 0 && learned != d.configured
}

// pending reports whether the deployment app's id is not known yet and can still be
// learned: there is a key to ask Stream with, and no id has turned out wrong. Work that may
// be that app's waits for it then, and only then.
func (d *Deployment) pending() bool {
	return d.App() == 0 && d.Configured() && !d.wrong()
}

// knowable reports whether the deployment app's id is known, or can still be learned.
func (d *Deployment) knowable() bool {
	return d.Configured() && !d.wrong()
}

// For answers every customer with the deployment's own app. The pin is zero: in
// deployment mode nothing records an app, which is how everything written before apps had
// identities reads too.
func (d *Deployment) For(_ context.Context, customer string) (Identity, error) {
	if !d.Configured() {
		return Identity{}, ErrNoIdentity
	}
	if d.strict && d.wrong() {
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
	switch known := d.App(); {
	case known == 0 && d.pending():
		return Identity{}, ErrDeploymentAppUnknown
	case known != app:
		return Identity{}, ErrStreamAppMoved
	}
	return d.For(ctx, customer)
}

// learnt records the app Stream says the deployment's key belongs to. One that is not the
// configured id is remembered as wrong: it names no pin from then on, and in strict mode the
// deployment stops answering.
func (d *Deployment) learnt(actual int64) (int64, error) {
	d.learned.Store(actual)
	if d.wrong() {
		return 0, fmt.Errorf("%w: stream.app_id is %d, and the deployment's key belongs to app %d",
			ErrDeploymentAppMismatch, d.configured, actual)
	}
	return actual, nil
}
