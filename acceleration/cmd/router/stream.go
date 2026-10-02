package main

import (
	"context"
	"errors"
	"log/slog"
	"time"

	"github.com/GetStream/Vision-Agents/acceleration/internal/config"
	"github.com/GetStream/Vision-Agents/acceleration/internal/store"
	"github.com/GetStream/Vision-Agents/acceleration/internal/streamapp"
)

// newStreamClients builds what every Stream action the router takes resolves through. In
// deployment mode that is the deployment's own app, for every customer.
func newStreamClients(settings config.Config) *streamapp.Clients {
	deployment := streamapp.NewDeployment(streamapp.DeploymentOptions{
		APIKey:    settings.Stream.APIKey,
		Secret:    settings.Stream.APISecret,
		UserToken: settings.Stream.UserToken,
		BaseURL:   settings.Stream.BaseURL,
	})
	return streamapp.NewClients(deployment, streamapp.ClientsOptions{})
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

// learnDeploymentApp asks Stream which app the deployment's own credential belongs to, so
// work pinned to that app can be finished here and an export can say what an unpinned row
// meant, and logs it once. Nothing waits on it: a deployment that cannot reach Stream
// starts all the same, writes as it always has, and asks again later.
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
