package main

import (
	"context"
	"errors"
	"fmt"
	"log/slog"
	"math"
	"os"
	"path/filepath"
	"strconv"
	"strings"
	"time"

	"github.com/GetStream/Vision-Agents/acceleration/internal/agent"
	"github.com/GetStream/Vision-Agents/acceleration/internal/eotdefaults"
	"github.com/joho/godotenv"
)

const (
	demoEOTDefaultEndpoint = eotdefaults.HostedDemoEndpoint
	demoEOTURLVar          = "ROUTER_EOT_URL"
	demoEOTModeVar         = "ROUTER_EOT_MODE"
	demoEOTThresholdVar    = "ROUTER_EOT_THRESHOLD"
	demoEOTTokenFileVar    = "ROUTER_EOT_ID_TOKEN_FILE"
	demoEOTPreflightLimit  = 5 * time.Second
)

type demoEOTSettings struct {
	endpoint  string
	mode      agent.EOTMode
	threshold float64
	tokenFile string
}

// loadDemoDotEnv loads the nearest checkout .env before flags and environment-backed
// defaults are read. godotenv preserves values already present in the process environment.
func loadDemoDotEnv() error {
	path := findDemoDotEnv()
	if path == "" {
		return nil
	}
	if err := godotenv.Load(path); err != nil {
		// Parser errors may include the offending line, which can contain credentials.
		return errors.New("could not load the nearest .env file")
	}
	return nil
}

func findDemoDotEnv() string {
	dir, err := os.Getwd()
	if err != nil {
		return ""
	}
	for {
		path := filepath.Join(dir, ".env")
		if info, err := os.Stat(path); err == nil && !info.IsDir() {
			return path
		}
		parent := filepath.Dir(dir)
		if parent == dir {
			return ""
		}
		dir = parent
	}
}

func demoEOTSettingsFrom(lookup func(string) (string, bool)) (demoEOTSettings, error) {
	settings := demoEOTSettings{
		endpoint:  demoEOTDefaultEndpoint,
		mode:      agent.EOTModePrimary,
		threshold: 0.5,
	}
	modeOverrode := false
	if value, ok := lookup(demoEOTURLVar); ok {
		settings.endpoint = strings.TrimSpace(value)
	}
	if value, ok := lookup(demoEOTModeVar); ok {
		settings.mode = agent.EOTMode(strings.TrimSpace(value))
		modeOverrode = true
	}
	if settings.mode != agent.EOTModeGate && settings.mode != agent.EOTModePrimary {
		return demoEOTSettings{}, fmt.Errorf("%s must be gate or primary", demoEOTModeVar)
	}
	if !modeOverrode && !eotdefaults.IsHostedDemoEndpoint(settings.endpoint) {
		settings.mode = agent.EOTModeGate
	}
	if value, ok := lookup(demoEOTThresholdVar); ok {
		threshold, err := strconv.ParseFloat(strings.TrimSpace(value), 64)
		if err != nil || math.IsNaN(threshold) || math.IsInf(threshold, 0) || threshold < 0 || threshold > 1 {
			return demoEOTSettings{}, fmt.Errorf("%s must be a finite number between 0 and 1", demoEOTThresholdVar)
		}
		settings.threshold = threshold
	}
	if value, ok := lookup(demoEOTTokenFileVar); ok {
		settings.tokenFile = strings.TrimSpace(value)
	}
	if eotdefaults.IsHostedDemoOrigin(settings.endpoint) && settings.tokenFile != "" {
		return demoEOTSettings{}, fmt.Errorf("%s requires a private EOT URL, not the hosted demo endpoint", demoEOTTokenFileVar)
	}
	return settings, nil
}

func (settings demoEOTSettings) usesHostedDemoClient() bool {
	return eotdefaults.IsHostedDemoEndpoint(settings.endpoint) && settings.tokenFile == ""
}

func newDemoEOTClient(settings demoEOTSettings) (*agent.EOTClient, error) {
	if settings.endpoint == "" {
		return nil, nil
	}
	if settings.usesHostedDemoClient() {
		return agent.NewHostedDemoEOTClient()
	}
	return agent.NewEOTClient(settings.endpoint, settings.tokenFile)
}

func preflightDemoEOT(ctx context.Context, client *agent.EOTClient, hosted bool, logger *slog.Logger) error {
	if client == nil {
		return nil
	}
	preflight, cancel := context.WithTimeout(ctx, demoEOTPreflightLimit)
	defer cancel()
	// The minimum accepted window is enough to verify auth, routing and model readiness
	// without uploading caller audio or inventing pause metadata.
	_, err := client.Score(preflight, "demo-preflight", make([]byte, 320*2))
	if hosted && agent.IsTransientEOTError(err) {
		logger.Warn("hosted EOT preflight is temporarily unavailable; runtime retries remain enabled")
		return nil
	}
	if err != nil {
		return fmt.Errorf("EOT preflight failed; check endpoint availability and configuration: %w", err)
	}
	return nil
}
