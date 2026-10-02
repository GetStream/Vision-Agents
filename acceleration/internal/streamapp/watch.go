package streamapp

import (
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"net/http"
	"time"

	getstream "github.com/GetStream/getstream-go/v5"

	"github.com/GetStream/Vision-Agents/acceleration/internal/store"
)

// WatchEvery is how often app mode checks on every connected app.
const WatchEvery = 15 * time.Minute

// Ended is told when the router stops acting in an app, so the work pinned to it ends.
type Ended func(customer string, app int64)

// Watch checks every connected app on an interval until ctx ends.
func (s *Stored) Watch(ctx context.Context, clients *Clients, every time.Duration, ended Ended) {
	ticker := time.NewTicker(every)
	defer ticker.Stop()
	for {
		s.CheckApps(ctx, clients, ended)
		select {
		case <-ctx.Done():
			return
		case <-ticker.C:
		}
	}
}

// CheckApps asks Stream about each connected app once. One Stream suspended, one that stopped
// checking the tokens it is sent, and one whose key turns out to be another app's is
// blocked, and what was pinned to it ends. A key Stream refuses is marked rejected, so the
// app's next key is used from then on.
func (s *Stored) CheckApps(ctx context.Context, clients *Clients, ended Ended) {
	apps, err := s.store.StreamApps(ctx, true)
	if err != nil {
		s.logger.Warn("stream: could not list the apps to check", "error", err)
		return
	}
	for _, app := range apps {
		if ctx.Err() != nil {
			return
		}
		s.check(ctx, clients, app, ended)
	}
}

func (s *Stored) check(ctx context.Context, clients *Clients, app store.StreamApp, ended Ended) {
	identity, err := s.registered(ctx, app)
	if err != nil {
		s.logger.Warn("stream: could not act in an app to check it", "customer_id", app.CustomerID, "error", err)
		return
	}
	// A client of the check's own, so checking every app does not push the ones in use out
	// of the cache.
	client, err := newStreamClient(identity, clients.http)
	if err != nil {
		s.logger.Warn("stream: could not act in an app to check it", "customer_id", app.CustomerID, "error", err)
		return
	}
	readiness, err := ReadReadiness(ctx, client, s.now())
	var refused *getstream.StreamError
	if errors.As(err, &refused) && (refused.StatusCode == http.StatusUnauthorized || refused.StatusCode == http.StatusForbidden) {
		reason := fmt.Sprintf("Stream answered %d to the key", refused.StatusCode)
		if err := s.store.RejectStreamAppKey(ctx, app.CustomerID, identity.APIKey, reason, s.now()); err != nil {
			s.logger.Warn("stream: could not mark a refused key", "customer_id", app.CustomerID, "error", err)
		}
		s.logger.Warn("stream: Stream refuses one of an app's keys", "customer_id", app.CustomerID, "api_key", identity.APIKey)
		clients.Invalidate(app.CustomerID)
		return
	}
	if err != nil {
		s.logger.Warn("stream: could not check an app", "customer_id", app.CustomerID, "error", err)
		return
	}

	checks, _ := json.Marshal(map[string]any{
		"channel_type": readiness.ChannelType, "call_type": readiness.CallType,
		"suspended": readiness.Suspended, "auth_checks_off": readiness.AuthChecksOff,
	})
	if err := s.store.RecordStreamAppChecks(ctx, app.CustomerID, checks, readiness.CheckedAt); err != nil {
		s.logger.Warn("stream: could not record an app's checks", "customer_id", app.CustomerID, "error", err)
	}
	reason := readiness.standing()
	if readiness.App != app.StreamAppPK {
		reason = "the app's key belongs to another Stream app"
	}
	if reason == "" {
		return
	}
	blocked, err := s.store.BlockStreamApp(ctx, app.CustomerID, reason)
	if err != nil {
		s.logger.Warn("stream: could not block an app", "customer_id", app.CustomerID, "error", err)
		return
	}
	clients.Invalidate(app.CustomerID)
	if blocked {
		s.logger.Warn("stream: the router stopped acting in an app", "customer_id", app.CustomerID, "reason", reason)
		if ended != nil {
			ended(app.CustomerID, app.StreamAppPK)
		}
	}
}
