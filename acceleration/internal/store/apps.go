package store

import (
	"context"
	"database/sql"
	"errors"
	"fmt"
	"time"
)

// ErrNoAPIKey says no live key has that id. It is one error for every reason a key might
// not be usable — unknown, revoked, expired — because a caller that can tell those apart
// can find out which keys exist.
var ErrNoAPIKey = errors.New("store: no such api key")

// CreateOrganization stores a new organization and fills in its id and timestamp.
func (s *Store) CreateOrganization(ctx context.Context, org *Organization) error {
	if org.Name == "" {
		return errors.New("store: an organization needs a name")
	}

	org.ID = newID()
	org.CreatedAt = time.Now().UTC()

	if _, err := s.db.NewInsert().Model(org).Exec(ctx); err != nil {
		return fmt.Errorf("store: create organization: %w", err)
	}
	return nil
}

// CreateApp stores a new app and fills in its id and timestamp.
func (s *Store) CreateApp(ctx context.Context, app *App) error {
	if app.OrganizationID == "" {
		return errors.New("store: an app needs an organization")
	}
	if app.Name == "" {
		return errors.New("store: an app needs a name")
	}

	app.ID = newID()
	app.CreatedAt = time.Now().UTC()

	if _, err := s.db.NewInsert().Model(app).Exec(ctx); err != nil {
		return fmt.Errorf("store: create app: %w", err)
	}
	return nil
}

// CreateAPIKey stores a credential. The id is minted by the caller rather than here,
// because the key is what the caller is handed and it has to be well formed.
func (s *Store) CreateAPIKey(ctx context.Context, key *APIKey) error {
	if key.ID == "" || key.AppID == "" {
		return errors.New("store: a key needs an id and an app")
	}
	if len(key.Sealed) == 0 {
		return errors.New("store: a key needs a sealed secret")
	}

	key.CreatedAt = time.Now().UTC()

	if _, err := s.db.NewInsert().Model(key).Exec(ctx); err != nil {
		return fmt.Errorf("store: create api key: %w", err)
	}
	return nil
}

// APIKeyOwner is the app a live key belongs to, plus the sealed secret to verify the
// caller's token with. It is one row rather than two lookups because it is read on every
// authenticated request.
type APIKeyOwner struct {
	AppID          string `bun:"app_id"`
	OrganizationID string `bun:"organization_id"`
	Sealed         []byte `bun:"secret_sealed"`
	KEKVersion     int    `bun:"kek_version"`
}

// LiveAPIKey returns the owner of the key with that id, provided it has not been revoked
// and has not expired. Every other outcome is ErrNoAPIKey.
func (s *Store) LiveAPIKey(ctx context.Context, id string) (APIKeyOwner, error) {
	var owner APIKeyOwner

	err := s.db.NewSelect().Model((*APIKey)(nil)).
		ColumnExpr("k.app_id, k.secret_sealed, k.kek_version").
		ColumnExpr("ap.organization_id").
		Join("JOIN apps AS ap ON ap.id = k.app_id").
		Where("k.id = ?", id).
		Where("k.revoked_at IS NULL").
		Where("k.expires_at IS NULL OR k.expires_at > ?", time.Now().UTC()).
		Scan(ctx, &owner)
	if errors.Is(err, sql.ErrNoRows) {
		return APIKeyOwner{}, ErrNoAPIKey
	}
	if err != nil {
		return APIKeyOwner{}, fmt.Errorf("store: live api key: %w", err)
	}

	return owner, nil
}

// TouchAPIKey records that a key was used, but only when the last record is older than
// the interval. A synchronous update on every request would double the writes of a busy
// key, and without any record at all nobody can answer whether a key is still in use, so
// nobody ever revokes one.
func (s *Store) TouchAPIKey(ctx context.Context, id string, interval time.Duration) error {
	now := time.Now().UTC()
	_, err := s.db.NewUpdate().Model((*APIKey)(nil)).
		Set("last_used_at = ?", now).
		Where("id = ?", id).
		Where("last_used_at IS NULL OR last_used_at < ?", now.Add(-interval)).
		Exec(ctx)
	if err != nil {
		return fmt.Errorf("store: touch api key: %w", err)
	}
	return nil
}

// RevokeAPIKey stops a key working. The row survives, because the audit trail of which key
// made which call has to outlive the key.
func (s *Store) RevokeAPIKey(ctx context.Context, id, by string) error {
	now := time.Now().UTC()
	result, err := s.db.NewUpdate().Model((*APIKey)(nil)).
		Set("revoked_at = ?", now).
		Set("revoked_by = ?", by).
		Where("id = ?", id).
		Where("revoked_at IS NULL").
		Exec(ctx)
	if err != nil {
		return fmt.Errorf("store: revoke api key: %w", err)
	}
	affected, err := result.RowsAffected()
	if err != nil {
		return fmt.Errorf("store: revoke api key: %w", err)
	}
	if affected == 0 {
		return ErrNoAPIKey
	}
	return nil
}

// ListAPIKeys returns an app's keys, revoked ones included, newest first.
func (s *Store) ListAPIKeys(ctx context.Context, appID string) ([]APIKey, error) {
	var keys []APIKey
	err := s.db.NewSelect().Model(&keys).
		Where("app_id = ?", appID).
		Order("created_at DESC").
		Scan(ctx)
	if err != nil {
		return nil, fmt.Errorf("store: list api keys: %w", err)
	}
	return keys, nil
}
