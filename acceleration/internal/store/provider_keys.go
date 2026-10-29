package store

import (
	"context"
	"database/sql"
	"errors"
	"fmt"
	"time"

	"github.com/uptrace/bun"

	"github.com/GetStream/Vision-Agents/acceleration/internal/stack"
)

// ErrNoProviderKey says a customer keeps no key for that provider in that region.
var ErrNoProviderKey = errors.New("store: no such provider key")

const (
	defaultProviderKeyLimit = 25
	maxProviderKeyLimit     = 200
)

// ProviderKeyLimit is the page size a provider key list uses for the limit asked for.
// ProviderKeyPage returns one row more than this.
func ProviderKeyLimit(asked int) int {
	return clampLimit(asked, defaultProviderKeyLimit, maxProviderKeyLimit)
}

// ProviderKeyPosition is where a page of a customer's provider keys ends: the provider and the
// region of its last key, which are the key's own and so a total order.
type ProviderKeyPosition struct {
	Provider string `json:"p"`
	Region   string `json:"r"`
}

// ProviderKey is a key a customer keeps with the router for their own account at a vendor, in
// one region (20261202130000_provider_keys.sql). Work routed on it is billed to them by the
// vendor, and by the router as BYOK.
type ProviderKey struct {
	bun.BaseModel `bun:"table:provider_keys,alias:pk"`

	CustomerID string `bun:"customer_id,pk"`
	Provider   string `bun:"provider,pk"`
	Region     string `bun:"region,pk"`
	// APIKeySealed is the key, sealed under KEKVersion and bound to the customer, the provider
	// and the region, so it opens on no other row.
	APIKeySealed []byte `bun:"api_key_sealed,notnull"`
	KEKVersion   int    `bun:"kek_version,notnull"`
	// KeyHint is the key's last four characters, empty for one too short to show any of.
	KeyHint   string    `bun:"key_hint,notnull"`
	CreatedAt time.Time `bun:"created_at,notnull"`
	UpdatedAt time.Time `bun:"updated_at,notnull"`
}

// PutProviderKey stores a key, replacing the one the customer kept for its provider and region,
// and fills in its timestamps. It answers the row it replaced, and false when there was none.
func (s *Store) PutProviderKey(ctx context.Context, key *ProviderKey) (ProviderKey, bool, error) {
	if key.CustomerID == "" || key.Provider == "" || key.Region == "" || len(key.APIKeySealed) == 0 {
		return ProviderKey{}, false, stack.Wrap(errors.New("store: a provider key needs a customer, a provider, a region and a key"))
	}

	var was ProviderKey
	var replaced bool
	err := s.db.RunInTx(ctx, nil, func(ctx context.Context, tx bun.Tx) error {
		err := tx.NewSelect().Model(&was).
			Where("customer_id = ? AND provider = ? AND region = ?", key.CustomerID, key.Provider, key.Region).
			For("UPDATE").
			Scan(ctx)
		switch {
		case err == nil:
			replaced = true
		case !errors.Is(err, sql.ErrNoRows):
			return err
		}
		now := time.Now().UTC().Truncate(time.Microsecond)
		key.CreatedAt, key.UpdatedAt = now, now
		if replaced {
			key.CreatedAt = was.CreatedAt
		}
		_, err = tx.NewInsert().Model(key).
			On("CONFLICT (customer_id, provider, region) DO UPDATE").
			Set("api_key_sealed = EXCLUDED.api_key_sealed").
			Set("kek_version = EXCLUDED.kek_version").
			Set("key_hint = EXCLUDED.key_hint").
			Set("updated_at = EXCLUDED.updated_at").
			Exec(ctx)
		return err
	})
	if err != nil {
		return ProviderKey{}, false, stack.Wrap(fmt.Errorf("store: put provider key: %w", err))
	}
	return was, replaced, nil
}

// ProviderKeys returns every key a customer keeps, by provider and then region. A customer keeps
// at most one per provider and region, so there is no page.
func (s *Store) ProviderKeys(ctx context.Context, customerID string) ([]ProviderKey, error) {
	if customerID == "" {
		return nil, stack.Wrap(errors.New("store: customer id is required"))
	}
	keys := []ProviderKey{}
	err := s.db.NewSelect().Model(&keys).
		Where("customer_id = ?", customerID).
		Order("provider", "region").
		Scan(ctx)
	if err != nil {
		return nil, stack.Wrap(fmt.Errorf("store: provider keys: %w", err))
	}
	return keys, nil
}

// ProviderKeyPage returns a page of a customer's keys, by provider and then region, from after,
// one more than the limit so a caller can tell whether there is another.
func (s *Store) ProviderKeyPage(ctx context.Context, customerID string, limit int, after *ProviderKeyPosition) ([]ProviderKey, error) {
	if customerID == "" {
		return nil, stack.Wrap(errors.New("store: customer id is required"))
	}
	var keys []ProviderKey
	query := s.db.NewSelect().Model(&keys).Where("customer_id = ?", customerID)
	if after != nil {
		query = query.Where("(provider, region) > (?, ?)", after.Provider, after.Region)
	}
	err := query.Order("provider", "region").Limit(ProviderKeyLimit(limit) + 1).Scan(ctx)
	if err != nil {
		return nil, stack.Wrap(fmt.Errorf("store: provider key page: %w", err))
	}
	return keys, nil
}

// ProviderKeysOf returns every customer's keys for a provider, for a job that reaches each
// customer's own account at it, by customer and then region.
func (s *Store) ProviderKeysOf(ctx context.Context, provider string) ([]ProviderKey, error) {
	if provider == "" {
		return nil, stack.Wrap(errors.New("store: provider is required"))
	}
	keys := []ProviderKey{}
	err := s.db.NewSelect().Model(&keys).
		Where("provider = ?", provider).
		Order("customer_id", "region").
		Scan(ctx)
	if err != nil {
		return nil, stack.Wrap(fmt.Errorf("store: provider keys of %s: %w", provider, err))
	}
	return keys, nil
}

// DeleteProviderKey forgets a customer's key for a provider in a region, and answers the row it
// deleted. Work is routed on the deployment's own key from then on.
func (s *Store) DeleteProviderKey(ctx context.Context, customerID, provider, region string) (ProviderKey, error) {
	if customerID == "" || provider == "" || region == "" {
		return ProviderKey{}, stack.Wrap(errors.New("store: a customer, a provider and a region are required"))
	}
	var deleted ProviderKey
	err := s.db.RunInTx(ctx, nil, func(ctx context.Context, tx bun.Tx) error {
		err := tx.NewSelect().Model(&deleted).
			Where("customer_id = ? AND provider = ? AND region = ?", customerID, provider, region).
			For("UPDATE").
			Scan(ctx)
		if err != nil {
			return err
		}
		_, err = tx.NewDelete().Model(&deleted).WherePK().Exec(ctx)
		return err
	})
	if errors.Is(err, sql.ErrNoRows) {
		return ProviderKey{}, stack.Wrap(fmt.Errorf("%w: %s in %s", ErrNoProviderKey, provider, region))
	}
	if err != nil {
		return ProviderKey{}, stack.Wrap(fmt.Errorf("store: delete provider key: %w", err))
	}
	return deleted, nil
}
