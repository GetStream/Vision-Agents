package streamapp

import "context"

// Static answers from a fixed list of customers' apps, for tests and for a deployment
// embedding the router that knows its customers' apps itself. A customer it does not list
// is the fallback's, when there is one.
type Static struct {
	apps     map[string]Identity
	fallback Source
}

// NewStatic returns a Source over the identities given, keyed by customer. Each one's
// StreamApp is its pin.
func NewStatic(apps map[string]Identity, fallback Source) *Static {
	listed := make(map[string]Identity, len(apps))
	for customer, identity := range apps {
		identity.CustomerID = customer
		listed[customer] = identity
	}
	return &Static{apps: listed, fallback: fallback}
}

// For is the customer's own app, or the fallback's answer for a customer not listed.
func (s *Static) For(ctx context.Context, customer string) (Identity, error) {
	if identity, ok := s.apps[customer]; ok {
		return identity, nil
	}
	if s.fallback != nil {
		return s.fallback.For(ctx, customer)
	}
	return Identity{}, ErrNoIdentity
}

// ForApp finishes work in the app it was pinned to. A listed customer's work pinned
// anywhere but its app is parked; anything else is the fallback's.
func (s *Static) ForApp(ctx context.Context, customer string, app int64) (Identity, error) {
	if identity, ok := s.apps[customer]; ok && identity.StreamApp == app {
		return identity, nil
	}
	if s.fallback != nil {
		return s.fallback.ForApp(ctx, customer, app)
	}
	return Identity{}, ErrStreamAppMoved
}
