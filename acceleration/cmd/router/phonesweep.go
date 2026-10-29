package main

import (
	"context"
	"log/slog"
	"time"

	"github.com/GetStream/Vision-Agents/acceleration/internal/phone"
)

const (
	phoneSweepEvery = 10 * time.Minute
	// A call longer than this is not a campaign call, and the session's own limits end it far
	// sooner, so its trunk is one whose end hook never came.
	phoneCallMaxAge = 6 * time.Hour
)

// startPhoneSweeper deletes the per-call trunks of calls that never reported their end, and
// retires the status urls of old placed calls, until ctx is done.
func startPhoneSweeper(ctx context.Context, service *phone.Service, logger *slog.Logger) {
	go func() {
		ticker := time.NewTicker(phoneSweepEvery)
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
