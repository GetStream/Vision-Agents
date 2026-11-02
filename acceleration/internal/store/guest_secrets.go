package store

import (
	"context"
	"crypto/rand"
	"crypto/sha256"
	"database/sql"
	"encoding/base64"
	"errors"
	"fmt"

	"github.com/GetStream/Vision-Agents/acceleration/internal/stack"
)

// ErrGuestSecretRefused is a refresh secret that is not the one the guest was last handed:
// wrong, already used, or for a guest that has since been claimed.
var ErrGuestSecretRefused = errors.New("store: that refresh secret does not belong to an unclaimed guest")

// NewGuestSecret returns a refresh secret to hand a guest and the hash to keep of it.
//
// 256 random bits, so the hash needs no salt or stretching: there is nothing to guess.
func NewGuestSecret() (secret string, hash []byte, err error) {
	raw := make([]byte, 32)
	if _, err := rand.Read(raw); err != nil {
		return "", nil, stack.Wrap(err)
	}
	secret = base64.RawURLEncoding.EncodeToString(raw)
	return secret, GuestSecretHash(secret), nil
}

// GuestSecretHash is what is kept of a refresh secret.
func GuestSecretHash(secret string) []byte {
	sum := sha256.Sum256([]byte(secret))
	return sum[:]
}

// RotateGuestSecret spends a guest's refresh secret and records the next one in its place,
// returning the guest.
//
// One statement, so a secret is spent once: two requests replaying the same secret cannot both
// see it match. A guest that has been claimed is refused, since it has an account to sign in
// with now.
func (s *Store) RotateGuestSecret(ctx context.Context, customerID, id string, spent, next []byte) (User, error) {
	if customerID == "" || id == "" || len(spent) == 0 || len(next) == 0 {
		return User{}, stack.Wrap(errors.New("store: a customer, a guest id and both secret hashes are required"))
	}

	var guest User
	err := s.db.NewUpdate().Model(&guest).
		Set("refresh_secret_hash = ?", next).
		Where("customer_id = ?", customerID).
		Where("id = ?", id).
		Where("kind = ?", UserKindGuest).
		Where("claimed_by IS NULL").
		Where("refresh_secret_hash = ?", spent).
		Returning("*").
		Scan(ctx)
	if errors.Is(err, sql.ErrNoRows) {
		return User{}, stack.Wrap(ErrGuestSecretRefused)
	}
	if err != nil {
		return User{}, stack.Wrap(fmt.Errorf("store: rotate guest secret: %w", err))
	}
	return guest, nil
}

// SetGuestSecret replaces a guest's refresh secret without the old one, for a caller that
// already proved it may: the guest's own still-valid token, or the app's backend.
func (s *Store) SetGuestSecret(ctx context.Context, customerID, id string, hash []byte) error {
	if customerID == "" || id == "" || len(hash) == 0 {
		return stack.Wrap(errors.New("store: a customer, a guest id and a secret hash are required"))
	}

	result, err := s.db.NewUpdate().Model((*User)(nil)).
		Set("refresh_secret_hash = ?", hash).
		Where("customer_id = ?", customerID).
		Where("id = ?", id).
		Where("kind = ?", UserKindGuest).
		Where("claimed_by IS NULL").
		Exec(ctx)
	if err != nil {
		return stack.Wrap(fmt.Errorf("store: set guest secret: %w", err))
	}
	if n, _ := result.RowsAffected(); n == 0 {
		return notAGuest(id, customerID)
	}
	return nil
}

// SetGuestProfile records what the app's backend says a guest is called, so the name a
// transcript shows and the one a returning guest is handed agree with what Stream holds.
func (s *Store) SetGuestProfile(ctx context.Context, customerID, id, name string, custom map[string]any) error {
	if customerID == "" || id == "" {
		return stack.Wrap(errors.New("store: a customer and a guest id are required"))
	}
	if custom == nil {
		custom = map[string]any{}
	}

	result, err := s.db.NewUpdate().Model((*User)(nil)).
		Set("name = ?", name).
		Set("custom = ?", custom).
		Where("customer_id = ?", customerID).
		Where("id = ?", id).
		Where("kind = ?", UserKindGuest).
		Where("claimed_by IS NULL").
		Exec(ctx)
	if err != nil {
		return stack.Wrap(fmt.Errorf("store: set guest profile: %w", err))
	}
	if n, _ := result.RowsAffected(); n == 0 {
		return notAGuest(id, customerID)
	}
	return nil
}
