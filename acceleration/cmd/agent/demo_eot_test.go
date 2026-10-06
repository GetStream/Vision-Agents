package main

import (
	"os"
	"path/filepath"
	"strings"
	"testing"

	"github.com/GetStream/Vision-Agents/acceleration/internal/agent"
)

func lookupMap(values map[string]string) func(string) (string, bool) {
	return func(key string) (string, bool) {
		value, ok := values[key]
		return value, ok
	}
}

func TestDemoEOTSettingsDefaultsToPrimaryCloudRunScorer(t *testing.T) {
	settings, err := demoEOTSettingsFrom(lookupMap(nil))
	if err != nil {
		t.Fatal(err)
	}
	if settings.endpoint != demoEOTDefaultEndpoint || settings.mode != agent.EOTModePrimary || settings.threshold != 0.5 {
		t.Fatalf("unexpected demo defaults: %+v", settings)
	}
	if !settings.usesGCloudTokenSource() {
		t.Fatal("the trusted default endpoint should reuse the local gcloud login")
	}
}

func TestDemoEOTSettingsPreserveExplicitOverrides(t *testing.T) {
	settings, err := demoEOTSettingsFrom(lookupMap(map[string]string{
		demoEOTURLVar:       "http://127.0.0.1:8080/v1/eot",
		demoEOTModeVar:      "gate",
		demoEOTThresholdVar: "0.72",
		demoEOTTokenFileVar: "/tmp/eot-token",
	}))
	if err != nil {
		t.Fatal(err)
	}
	if settings.endpoint != "http://127.0.0.1:8080/v1/eot" || settings.mode != agent.EOTModeGate || settings.threshold != 0.72 || settings.tokenFile != "/tmp/eot-token" {
		t.Fatalf("explicit settings were not applied: %+v", settings)
	}
	if settings.usesGCloudTokenSource() {
		t.Fatal("an overridden endpoint/token file must not receive the demo's gcloud credential")
	}
}

func TestDemoEOTGCloudTokenSourceIsRestrictedToCanonicalEndpoint(t *testing.T) {
	for _, test := range []struct {
		endpoint  string
		tokenFile string
		want      bool
	}{
		{endpoint: demoEOTDefaultEndpoint, want: true},
		{endpoint: demoEOTDefaultEndpoint, tokenFile: "/tmp/token", want: false},
		{endpoint: "https://other.example/v1/eot", want: false},
		{endpoint: "https://audioturn-edge-5gdhza7snq-wn.a.run.app.attacker.example/v1/eot", want: false},
	} {
		settings := demoEOTSettings{endpoint: test.endpoint, tokenFile: test.tokenFile}
		if got := settings.usesGCloudTokenSource(); got != test.want {
			t.Errorf("usesGCloudTokenSource(%q, tokenFile=%t) = %t, want %t",
				test.endpoint, test.tokenFile != "", got, test.want)
		}
	}

	explicitDefault, err := demoEOTSettingsFrom(lookupMap(map[string]string{demoEOTURLVar: demoEOTDefaultEndpoint}))
	if err != nil {
		t.Fatal(err)
	}
	if !explicitDefault.usesGCloudTokenSource() {
		t.Fatal("an explicit .env copy of the canonical endpoint should retain the demo auth path")
	}
}

func TestDemoEOTSettingsAllowAnExplicitDisable(t *testing.T) {
	settings, err := demoEOTSettingsFrom(lookupMap(map[string]string{demoEOTURLVar: ""}))
	if err != nil {
		t.Fatal(err)
	}
	if settings.endpoint != "" || settings.mode != agent.EOTModeGate {
		t.Fatalf("empty endpoint should restore the existing transcript cadence: %+v", settings)
	}
	settings, err = demoEOTSettingsFrom(lookupMap(map[string]string{
		demoEOTURLVar:  "",
		demoEOTModeVar: "primary",
	}))
	if err != nil {
		t.Fatal(err)
	}
	if settings.endpoint != "" || settings.mode != agent.EOTModePrimary || settings.usesGCloudTokenSource() {
		t.Fatalf("explicit primary mode without a scorer should fall back to semantic cadence: %+v", settings)
	}
}

func TestDemoEOTSettingsRejectInvalidModeAndThreshold(t *testing.T) {
	for _, values := range []map[string]string{
		{demoEOTModeVar: "automatic"},
		{demoEOTThresholdVar: "NaN"},
		{demoEOTThresholdVar: "+Inf"},
		{demoEOTThresholdVar: "-0.01"},
		{demoEOTThresholdVar: "1.01"},
		{demoEOTThresholdVar: "not-a-number"},
	} {
		if _, err := demoEOTSettingsFrom(lookupMap(values)); err == nil {
			t.Fatalf("expected settings to be rejected: %v", values)
		}
	}
}

func TestDemoDotEnvLoadsNearestFileWithoutReplacingEnvironment(t *testing.T) {
	root := t.TempDir()
	nested := filepath.Join(root, "nested")
	working := filepath.Join(nested, "cmd", "agent")
	if err := os.MkdirAll(working, 0o700); err != nil {
		t.Fatal(err)
	}
	if err := os.WriteFile(filepath.Join(root, ".env"), []byte("DEMO_PARENT_VALUE=parent\n"), 0o600); err != nil {
		t.Fatal(err)
	}
	if err := os.WriteFile(filepath.Join(nested, ".env"), []byte("DEMO_NEAREST_VALUE=nearest\nDEMO_ENV_OVERRIDE=file\n"), 0o600); err != nil {
		t.Fatal(err)
	}
	t.Setenv("DEMO_ENV_OVERRIDE", "environment")
	t.Chdir(working)

	if err := loadDemoDotEnv(); err != nil {
		t.Fatal(err)
	}
	if got := os.Getenv("DEMO_NEAREST_VALUE"); got != "nearest" {
		t.Fatalf("nearest .env value = %q, want nearest", got)
	}
	if got := os.Getenv("DEMO_ENV_OVERRIDE"); got != "environment" {
		t.Fatalf("environment value was overwritten: %q", got)
	}
	if got := os.Getenv("DEMO_PARENT_VALUE"); got != "" {
		t.Fatalf("a parent .env was loaded despite a nearer file: %q", got)
	}
}

func TestDemoDotEnvParseErrorDoesNotExposeFileContents(t *testing.T) {
	dir := t.TempDir()
	secret := "super-secret-token-value"
	if err := os.WriteFile(filepath.Join(dir, ".env"), []byte("TOKEN=\""+secret+"\n"), 0o600); err != nil {
		t.Fatal(err)
	}
	t.Chdir(dir)
	err := loadDemoDotEnv()
	if err == nil {
		t.Fatal("expected malformed .env to fail")
	}
	if strings.Contains(err.Error(), secret) {
		t.Fatalf(".env parse error exposed file contents: %v", err)
	}
}
