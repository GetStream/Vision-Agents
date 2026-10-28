package appconfig

import (
	"context"

	"github.com/GetStream/Vision-Agents/acceleration/internal/store"
)

// ProviderKeys returns the keys a customer keeps for their own accounts at the vendors. It is read
// every time a request of theirs is routed, so it is cached, and the keys stay sealed in the cache
// as they are in Postgres.
func (s *Store) ProviderKeys(ctx context.Context, customerID string) ([]store.ProviderKey, error) {
	if customerID == "" {
		return s.db.ProviderKeys(ctx, customerID)
	}
	return read(ctx, s, key("provider-keys", customerID), func(ctx context.Context) ([]store.ProviderKey, error) {
		return s.db.ProviderKeys(ctx, customerID)
	})
}

// PutProviderKey stores a key, replacing the one the customer kept for its provider and region,
// and answers the row it replaced, and false when there was none. Every replica routes on it from
// its next request.
func (s *Store) PutProviderKey(ctx context.Context, providerKey *store.ProviderKey) (store.ProviderKey, bool, error) {
	was, replaced, err := s.db.PutProviderKey(ctx, providerKey)
	if err != nil {
		return store.ProviderKey{}, false, err
	}
	s.forget(ctx, key("provider-keys", providerKey.CustomerID))
	return was, replaced, nil
}

// DeleteProviderKey forgets a customer's key for a provider in a region, and answers the row it
// deleted. Every replica routes on the deployment's own key from its next request.
func (s *Store) DeleteProviderKey(ctx context.Context, customerID, provider, region string) (store.ProviderKey, error) {
	deleted, err := s.db.DeleteProviderKey(ctx, customerID, provider, region)
	if err != nil {
		return store.ProviderKey{}, err
	}
	s.forget(ctx, key("provider-keys", customerID))
	return deleted, nil
}
