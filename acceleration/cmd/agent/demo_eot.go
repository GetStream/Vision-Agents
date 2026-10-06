package main

import (
	"context"
	"errors"
	"fmt"
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
	if settings.endpoint == "" && !modeOverrode {
		settings.mode = agent.EOTModeGate
	}
	return settings, nil
}

func (settings demoEOTSettings) usesHostedDemoClient() bool {
	return eotdefaults.IsHostedDemoEndpoint(settings.endpoint) && settings.tokenFile == ""
}

func newDemoEOTClient(_ context.Context, settings demoEOTSettings) (*agent.EOTClient, func(), error) {
	if settings.endpoint == "" {
		return nil, func() {}, nil
	}
	if eotdefaults.IsHostedDemoOrigin(settings.endpoint) {
		if settings.tokenFile != "" {
			return nil, nil, fmt.Errorf("%s requires a private EOT URL, not the hosted demo endpoint", demoEOTTokenFileVar)
		}
		if !eotdefaults.IsHostedDemoEndpoint(settings.endpoint) {
			return nil, nil, errors.New("the hosted demo endpoint path must be /v1/eot")
		}
		client, err := agent.NewHostedDemoEOTClient()
		if err != nil {
			return nil, nil, err
		}
		return client, func() {}, nil
	}
	client, err := agent.NewEOTClient(settings.endpoint, settings.tokenFile)
	if err != nil {
		return nil, nil, err
	}
	return client, func() {}, nil
}

func preflightDemoEOT(ctx context.Context, client *agent.EOTClient) error {
	if client == nil {
		return nil
	}
	preflight, cancel := context.WithTimeout(ctx, demoEOTPreflightLimit)
	defer cancel()
	// The minimum accepted window is enough to verify auth, routing and model readiness
	// without uploading caller audio or inventing pause metadata.
	_, err := client.Score(preflight, "demo-preflight", make([]byte, 320*2))
	if err != nil {
		return fmt.Errorf("EOT preflight failed; check endpoint availability and configuration: %w", err)
	}
	return nil
}

func preflightDemoEOTWithPolicy(ctx context.Context, client *agent.EOTClient, hosted bool, warn func()) error {
	err := preflightDemoEOT(ctx, client)
	return handleDemoEOTPreflightError(hosted, err, warn)
}

func handleDemoEOTPreflightError(hosted bool, err error, warn func()) error {
	if err == nil {
		return nil
	}
	if hosted && agent.IsTransientEOTError(err) {
		if warn != nil {
			warn()
		}
		return nil
	}
	return err
}
