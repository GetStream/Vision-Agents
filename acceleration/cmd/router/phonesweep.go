package main

import (
	"context"
	"log/slog"
	"time"

	"github.com/GetStream/Vision-Agents/acceleration/internal/phone"
)

// phoneSweepEvery is a variable so a test of the router's start can sweep sooner.
var phoneSweepEvery = 10 * time.Minute

// A call longer than this is not a campaign call, and the session's own limits end it far
// sooner, so its trunk is one whose end hook never came.
const phoneCallMaxAge = 6 * time.Hour

// startPhoneSweeper deletes the per-call trunks of calls that never reported their end, and
// retires the status urls of old placed calls, until ctx is done.
func startPhoneSweeper(ctx context.Context, service *phone.Service, logger *slog.Logger) {
	// Read before the goroutine, so a test that sets it back after run returns does not race it.
	every := phoneSweepEvery
	go func() {
		ticker := time.NewTicker(every)
		defer ticker.Stop()
		for {
			select {
			case <-ctx.Done():
				return
			case <-ticker.C:
				swept, err := service.SweepCallResources(ctx, time.Now().Add(-phoneCallMaxAge))
				if err != nil {
					logger.Error("could not sweep the trunks of calls that never ended", "error", err)
				} else if swept > 0 {
					logger.Info("swept the trunks of calls that never ended", "calls", swept)
				}
				retired, err := service.SweepStatusTokens(ctx)
				if err != nil {
					logger.Error("could not retire the status urls of old placed calls", "error", err)
				} else if retired > 0 {
					logger.Info("retired the status urls of old placed calls", "calls", retired)
				}
			}
		}
	}()
}
