package main

import (
	"context"
	"errors"
	"fmt"
	"log/slog"
	"time"

	"github.com/GetStream/Vision-Agents/acceleration/internal/auth"
	"github.com/GetStream/Vision-Agents/acceleration/internal/config"
	"github.com/GetStream/Vision-Agents/acceleration/internal/store"
	"github.com/GetStream/Vision-Agents/acceleration/internal/streamapp"
)

// newStreamClients builds what every Stream action the router takes resolves through. In
// deployment mode that is the deployment's own app, for every customer. In app mode it is
// the app each customer registered, which needs the store the apps are kept in and the
// keyring their keys are sealed under.
func newStreamClients(settings config.Config, pgStore *store.Store, sealer *auth.Sealer, logger *slog.Logger) (*streamapp.Clients, error) {
	perApp := settings.Stream.Tenancy == config.TenancyApp
	deployment := streamapp.NewDeployment(streamapp.DeploymentOptions{
		APIKey:    settings.Stream.APIKey,
		Secret:    settings.Stream.APISecret,
		UserToken: settings.Stream.UserToken,
		BaseURL:   settings.Stream.BaseURL,
		App:       settings.Stream.AppID,
		Strict:    perApp,
	})
	if !perApp {
		return streamapp.NewClients(deployment, streamapp.ClientsOptions{}), nil
	}
	if pgStore == nil {
		return nil, fmt.Errorf("stream.tenancy=%s keeps every app's keys in Postgres: set postgres.dsn", config.TenancyApp)
	}
	stored, err := streamapp.NewStored(streamapp.StoredOptions{
		Store: pgStore, Sealer: sealer, Deployment: deployment,
		FallbackToDeployment: settings.Stream.EffectiveFallback() == config.FallbackDeployment,
		Logger:               logger,
	})
	if err != nil {
		return nil, err
	}
	return streamapp.NewClients(stored, streamapp.ClientsOptions{}), nil
}

// streamPins tells the store how this deployment's Stream app pins cross to another one.
func streamPins(clients *streamapp.Clients) store.StreamPins {
	return store.StreamPins{Deployment: clients.DeploymentApp, For: clients.Pin}
}

// How long a deployment that could not learn its own app waits to ask again, doubling up
// to the most, and how long one attempt may take.
var (
	learnRetry    = 30 * time.Second
	learnRetryMax = 15 * time.Minute
	learnTimeout  = 30 * time.Second
)

// checkDeploymentApp is what app mode asks before it starts: whether the deployment's key
// belongs to the app stream.app_id names. A key that belongs to another app refuses the
// start, since every pin app mode writes would name the wrong app. Stream being out of
// reach does not: the check carries on beside the router, and the deployment's work waits
// until it is done.
func checkDeploymentApp(ctx context.Context, settings config.Config, clients *streamapp.Clients) error {
	if settings.Stream.Tenancy != config.TenancyApp {
		return nil
	}
	attempt, cancel := context.WithTimeout(ctx, learnTimeout)
	defer cancel()
	_, err := clients.LearnDeploymentApp(attempt)
	if errors.Is(err, streamapp.ErrDeploymentAppMismatch) {
		return err
	}
	return nil
}

// learnDeploymentApp asks Stream which app the deployment's own credential belongs to, so
// work pinned to that app can be finished here and an export can say what an unpinned row
// meant, and logs it once. Nothing waits on it: a deployment that cannot reach Stream
// starts all the same, writes as it always has, and asks again later. A configured id is
// checked rather than learned, and one that turns out to be another app's is said loudly.
func learnDeploymentApp(ctx context.Context, clients *streamapp.Clients, logger *slog.Logger) {
	wait := learnRetry
	for {
		attempt, cancel := context.WithTimeout(ctx, learnTimeout)
		app, err := clients.LearnDeploymentApp(attempt)
		cancel()
		switch {
		case err == nil:
			logger.Info("stream: the deployment acts in its own Stream app", "stream_app", app)
			return
		case errors.Is(err, streamapp.ErrDeploymentAppMismatch):
			logger.Error("stream: the deployment's key belongs to another Stream app than "+
				"stream.app_id names, so nothing pinned to that id is finished here", "error", err)
			return
		case errors.Is(err, streamapp.ErrNoIdentity), ctx.Err() != nil:
			return
		}
		logger.Warn("stream: could not learn which Stream app is the deployment's own, asking again later",
			"error", err, "retry_in", wait)
		select {
		case <-ctx.Done():
			return
		case <-time.After(wait):
		}
		wait = min(wait*2, learnRetryMax)
	}
}
