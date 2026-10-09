package main

import (
	"bytes"
	"context"
	"io"
	"log/slog"
	"net/http"
	"net/http/httptest"
	"os"
	"path/filepath"
	"testing"

	"github.com/GetStream/Vision-Agents/acceleration/internal/agent"
	"github.com/stretchr/testify/suite"
)

type DemoEOTSuite struct{ suite.Suite }

func TestDemoEOTSuite(t *testing.T) { suite.Run(t, new(DemoEOTSuite)) }

func lookupMap(values map[string]string) func(string) (string, bool) {
	return func(key string) (string, bool) {
		value, ok := values[key]
		return value, ok
	}
}

func (s *DemoEOTSuite) TestExplicitSettingsOverrideDefaults() {
	settings, err := demoEOTSettingsFrom(lookupMap(map[string]string{
		demoEOTURLVar:       "http://127.0.0.1:8080/v1/eot",
		demoEOTModeVar:      "primary",
		demoEOTThresholdVar: "0.72",
		demoEOTTokenFileVar: "/tmp/eot-token",
	}))
	s.Require().NoError(err)
	s.Equal(demoEOTSettings{
		endpoint: "http://127.0.0.1:8080/v1/eot", mode: agent.EOTModePrimary,
		threshold: 0.72, tokenFile: "/tmp/eot-token",
	}, settings)
}

func (s *DemoEOTSuite) TestAnEmptyEndpointDisablesScoring() {
	settings, err := demoEOTSettingsFrom(lookupMap(map[string]string{demoEOTURLVar: ""}))
	s.Require().NoError(err)
	s.Equal(agent.EOTModeGate, settings.mode)
	client, err := newDemoEOTClient(settings)
	s.NoError(err)
	s.Nil(client)
	s.NoError(preflightDemoEOT(s.T().Context(), client, false, slog.Default()))
}

func (s *DemoEOTSuite) TestPrivateEndpointsDefaultToGate() {
	settings, err := demoEOTSettingsFrom(lookupMap(map[string]string{demoEOTURLVar: "https://private.example/v1/eot"}))
	s.Require().NoError(err)
	s.Equal(agent.EOTModeGate, settings.mode)
}

func (s *DemoEOTSuite) TestTheHostedEndpointRejectsCredentialsAndOtherPaths() {
	_, err := demoEOTSettingsFrom(lookupMap(map[string]string{demoEOTTokenFileVar: "/tmp/token"}))
	s.Error(err)
	_, err = newDemoEOTClient(demoEOTSettings{endpoint: demoEOTDefaultEndpoint + "/other"})
	s.Error(err)
}

func (s *DemoEOTSuite) TestInvalidModesAndThresholdsAreRejected() {
	_, err := demoEOTSettingsFrom(lookupMap(map[string]string{demoEOTModeVar: "automatic"}))
	s.Error(err)
	_, err = demoEOTSettingsFrom(lookupMap(map[string]string{demoEOTThresholdVar: "NaN"}))
	s.Error(err)
	_, err = demoEOTSettingsFrom(lookupMap(map[string]string{demoEOTThresholdVar: "+Inf"}))
	s.Error(err)
	_, err = demoEOTSettingsFrom(lookupMap(map[string]string{demoEOTThresholdVar: "-0.01"}))
	s.Error(err)
	_, err = demoEOTSettingsFrom(lookupMap(map[string]string{demoEOTThresholdVar: "1.01"}))
	s.Error(err)
	_, err = demoEOTSettingsFrom(lookupMap(map[string]string{demoEOTThresholdVar: "not-a-number"}))
	s.Error(err)
}

func (s *DemoEOTSuite) preflight(status int, body string, hosted bool) (error, string) {
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		_, _ = io.Copy(io.Discard, r.Body)
		w.WriteHeader(status)
		_, _ = io.WriteString(w, body)
	}))
	s.T().Cleanup(server.Close)
	client, err := agent.NewEOTClient(server.URL, "")
	s.Require().NoError(err)
	var logs bytes.Buffer
	err = preflightDemoEOT(context.Background(), client, hosted, slog.New(slog.NewTextHandler(&logs, nil)))
	return err, logs.String()
}

func (s *DemoEOTSuite) TestHostedTransientFailuresWarnAndAllowStartup() {
	err, logs := s.preflight(http.StatusServiceUnavailable, "", true)
	s.NoError(err)
	s.Contains(logs, "temporarily unavailable")
}

func (s *DemoEOTSuite) TestPrivateEndpointFailuresPreventStartup() {
	err, _ := s.preflight(http.StatusServiceUnavailable, "", false)
	s.ErrorContains(err, "EOT preflight failed")
}

func (s *DemoEOTSuite) TestHostedAuthErrorsPreventStartup() {
	err, _ := s.preflight(http.StatusForbidden, "", true)
	s.ErrorContains(err, "EOT preflight failed")
}

func (s *DemoEOTSuite) TestHostedMalformedResponsesPreventStartup() {
	err, _ := s.preflight(http.StatusOK, "not-json", true)
	s.ErrorContains(err, "EOT preflight failed")
}

func (s *DemoEOTSuite) TestDotEnvLoadsTheNearestFileAndPreservesTheEnvironment() {
	root := s.T().TempDir()
	working := filepath.Join(root, "nested", "cmd")
	s.Require().NoError(os.MkdirAll(working, 0o700))
	s.Require().NoError(os.WriteFile(filepath.Join(root, ".env"), []byte("DEMO_PARENT_VALUE=parent\n"), 0o600))
	s.Require().NoError(os.WriteFile(filepath.Join(root, "nested", ".env"), []byte("DEMO_NEAREST_VALUE=nearest\nDEMO_ENV_OVERRIDE=file\n"), 0o600))
	s.T().Setenv("DEMO_ENV_OVERRIDE", "environment")
	s.T().Setenv("DEMO_NEAREST_VALUE", "")
	s.Require().NoError(os.Unsetenv("DEMO_NEAREST_VALUE"))
	s.T().Setenv("DEMO_PARENT_VALUE", "")
	s.Require().NoError(os.Unsetenv("DEMO_PARENT_VALUE"))
	s.T().Chdir(working)

	s.Require().NoError(loadDemoDotEnv())
	s.Equal("nearest", os.Getenv("DEMO_NEAREST_VALUE"))
	s.Equal("environment", os.Getenv("DEMO_ENV_OVERRIDE"))
	s.Empty(os.Getenv("DEMO_PARENT_VALUE"))
}

func (s *DemoEOTSuite) TestDotEnvParseErrorsDoNotExposeCredentials() {
	dir := s.T().TempDir()
	s.Require().NoError(os.WriteFile(filepath.Join(dir, ".env"), []byte("TOKEN=\"super-secret-token\n"), 0o600))
	s.T().Chdir(dir)
	err := loadDemoDotEnv()
	s.Require().Error(err)
	s.NotContains(err.Error(), "super-secret-token")
}
