package main

import (
	"context"
	"log/slog"
	"slices"
	"time"

	"github.com/GetStream/Vision-Agents/acceleration/internal/alerts"
	"github.com/GetStream/Vision-Agents/acceleration/internal/keypool"
	"github.com/GetStream/Vision-Agents/acceleration/internal/memory/mem0"
	"github.com/GetStream/Vision-Agents/acceleration/internal/routing"
)

// meteringAlertEvery is how often one item's metering alert is sent at most: each is an
// adapter or a price to fix, not an outage, and the rows are written meanwhile.
const meteringAlertEvery = time.Hour

// startMeteringAlerts sends an alert for each item whose rows this process recorded with no
// usage, on an estimate, with no price or with tokens that do not add up, and for each mem0
// refusal and refused provider key, until the returned function stops it, after one last
// check. They are a loop of their own because each item is held back for an hour, where the
// usage alerts are for a minute.
func startMeteringAlerts(ctx context.Context, source, shard string, logger *slog.Logger) (stop func()) {
	return startAlerts(ctx, alerts.New(alerts.Options{
		Logger: logger, Source: source, Tags: map[string]string{"shard": shard}, Every: usageAlertEvery, Limit: meteringAlertEvery,
	}, slices.Concat(routing.MeteringAlerts(meteringAlertEvery), mem0.Alerts(meteringAlertEvery), keypool.Alerts(meteringAlertEvery))...))
}

// startAlerts runs the loop until the returned function stops it, which waits for its last
// check. Shutdown does not end it: what the rows recorded while the writer drains still alerts.
func startAlerts(ctx context.Context, loop *alerts.Loop) (stop func()) {
	background, cancel := context.WithCancel(context.WithoutCancel(ctx))
	done := make(chan struct{})
	go func() {
		defer close(done)
		loop.Run(background)
	}()
	return func() {
		cancel()
		<-done
	}
}
