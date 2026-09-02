package main

import (
	"context"
	"fmt"
	"log/slog"
	"time"

	"github.com/GetStream/Vision-Agents/acceleration/internal/auth"
	"github.com/GetStream/Vision-Agents/acceleration/internal/store"
)

const (
	postgresEnvVar = "GATEWAY_POSTGRES_DSN"
	kekEnvVar      = "GATEWAY_AUTH_KEK"
	// lastUsedInterval throttles the record of a key's use. Writing on every
	// request would double the writes of a busy key; writing nothing means
	// nobody can answer whether a key is still in use, so nobody revokes one.
	lastUsedInterval = time.Minute
)

// newLookup resolves an API key to the app holding it, reading the same tables
// the router would in a deployment with no proxy in front of it. There is no
// second implementation of the check: this is the one the router runs.
func newLookup(dsn, kek string, logger *slog.Logger) (auth.Lookup, func() error, error) {
	if dsn == "" {
		return nil, nil, fmt.Errorf("%s is required: it is where the keys are", postgresEnvVar)
	}
	sealer, err := auth.NewSealer(kek)
	if err != nil {
		return nil, nil, fmt.Errorf("%s: %w", kekEnvVar, err)
	}
	pgStore, err := store.Open(dsn)
	if err != nil {
		return nil, nil, fmt.Errorf("open postgres: %w", err)
	}

	lookup := func(ctx context.Context, key string) (auth.App, error) {
		// Rejecting a malformed key costs nothing and saves a round trip.
		if !auth.ValidKey(key) {
			return auth.App{}, auth.ErrUnauthenticated
		}
		owner, err := pgStore.LiveAPIKey(ctx, key)
		if err != nil {
			return auth.App{}, auth.ErrUnauthenticated
		}
		secret, err := sealer.Open(owner.Sealed)
		if err != nil {
			return auth.App{}, fmt.Errorf("unseal key %s: %w", key, err)
		}
		if err := pgStore.TouchAPIKey(ctx, key, lastUsedInterval); err != nil {
			logger.Debug("could not record key use", "key", key, "error", err)
		}
		return auth.App{
			OrganizationID: owner.OrganizationID,
			AppID:          owner.AppID,
			Secret:         secret,
		}, nil
	}
	return lookup, pgStore.Close, nil
}
