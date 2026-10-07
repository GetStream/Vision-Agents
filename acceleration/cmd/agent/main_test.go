package main

import (
	"io"
	"log/slog"
	"net/http"
	"net/http/httptest"
	"os"
	"path/filepath"
	"strings"
	"sync/atomic"
	"testing"
	"time"

	"github.com/GetStream/Vision-Agents/acceleration/internal/agent"
	"github.com/GetStream/Vision-Agents/acceleration/internal/config"
)

func TestParseOptionsLoadsDotEnvBeforeEnvironmentFlagDefaults(t *testing.T) {
	keys := []string{
		skillsEnvVar,
		toolsEnvVar,
		demoEOTURLVar,
		demoEOTModeVar,
		demoEOTThresholdVar,
		demoEOTTokenFileVar,
	}
	clearEnvironmentForTest(t, keys...)
	t.Setenv(skillsEnvVar, "process-skills.yaml")

	workingDir := t.TempDir()
	dotEnv := "\n" +
		"HARNESS_SKILLS=file-skills.yaml\n" +
		"HARNESS_TOOLS=file-tools.yaml\n" +
		"ROUTER_EOT_URL=https://custom.example/v1/eot\n" +
		"ROUTER_EOT_MODE=gate\n" +
		"ROUTER_EOT_THRESHOLD=0.37\n" +
		"ROUTER_EOT_ID_TOKEN_FILE=/tmp/demo-id-token\n"
	if err := os.WriteFile(filepath.Join(workingDir, ".env"), []byte(dotEnv), 0o600); err != nil {
		t.Fatal(err)
	}
	t.Chdir(workingDir)

	parsed, verbose, err := parseOptions([]string{"-call", "from-flag"})
	if err != nil {
		t.Fatal(err)
	}
	if verbose || parsed.callID != "from-flag" {
		t.Fatalf("parsed options: call=%q verbose=%t", parsed.callID, verbose)
	}
	if parsed.skillsFile != "process-skills.yaml" {
		t.Fatalf("process environment should override .env for skills, got %q", parsed.skillsFile)
	}
	if parsed.toolsFile != "file-tools.yaml" {
		t.Fatalf(".env should be loaded before tools flag defaults, got %q", parsed.toolsFile)
	}

	settings, err := demoEOTSettingsFrom(os.LookupEnv)
	if err != nil {
		t.Fatal(err)
	}
	if settings.endpoint != "https://custom.example/v1/eot" || settings.mode != agent.EOTModeGate || settings.threshold != 0.37 || settings.tokenFile != "/tmp/demo-id-token" {
		t.Fatalf("EOT environment settings were not loaded before run: %+v", settings)
	}
	if settings.usesHostedDemoClient() {
		t.Fatal("a .env custom endpoint must not use the fixed hosted client")
	}

	parsed, _, err = parseOptions([]string{"-call", "from-flag", "-skills", "flag-skills.yaml"})
	if err != nil {
		t.Fatal(err)
	}
	if parsed.skillsFile != "flag-skills.yaml" {
		t.Fatalf("command-line flag should override environment default, got %q", parsed.skillsFile)
	}
}

func TestParseOptionsKeepsZeroConfigEOTDefaults(t *testing.T) {
	keys := []string{demoEOTURLVar, demoEOTModeVar, demoEOTThresholdVar, demoEOTTokenFileVar}
	clearEnvironmentForTest(t, keys...)
	t.Chdir(t.TempDir())

	if _, _, err := parseOptions([]string{"-call", "demo-call"}); err != nil {
		t.Fatal(err)
	}
	settings, err := demoEOTSettingsFrom(os.LookupEnv)
	if err != nil {
		t.Fatal(err)
	}
	if settings.endpoint != demoEOTDefaultEndpoint || settings.mode != agent.EOTModePrimary || settings.threshold != 0.5 {
		t.Fatalf("unexpected local demo defaults: %+v", settings)
	}
	if !settings.usesHostedDemoClient() {
		t.Fatal("the unmodified canonical endpoint should use the anonymous hosted client")
	}
}

func TestParseOptionsRequiresBackchannelOptIn(t *testing.T) {
	t.Chdir(t.TempDir())
	keys := []string{demoEOTURLVar, demoEOTModeVar, demoEOTThresholdVar, demoEOTTokenFileVar}
	clearEnvironmentForTest(t, keys...)

	parsed, _, err := parseOptions([]string{"-call", "demo-call"})
	if err != nil {
		t.Fatal(err)
	}
	if parsed.backchannel {
		t.Fatal("backchannel should be off by default")
	}

	parsed, _, err = parseOptions([]string{"-call", "demo-call", "-backchannel=true"})
	if err != nil {
		t.Fatal(err)
	}
	if !parsed.backchannel {
		t.Fatal("-backchannel=true should opt in to backchannel speech")
	}
}

func TestParseOptionsRequiresIdleCheckInOptIn(t *testing.T) {
	t.Chdir(t.TempDir())
	keys := []string{demoEOTURLVar, demoEOTModeVar, demoEOTThresholdVar, demoEOTTokenFileVar}
	clearEnvironmentForTest(t, keys...)

	parsed, _, err := parseOptions([]string{"-call", "demo-call"})
	if err != nil {
		t.Fatal(err)
	}
	if parsed.checkIn || !parsed.duplex().DisableIdleCheckIn {
		t.Fatal("idle check-in should be disabled by default in the standalone CLI")
	}

	parsed, _, err = parseOptions([]string{"-call", "demo-call", "-check-in=true"})
	if err != nil {
		t.Fatal(err)
	}
	if !parsed.checkIn || parsed.duplex().DisableIdleCheckIn {
		t.Fatal("-check-in=true should opt in to idle prompts")
	}
}

func TestParseOptionsExposesTheReplyTimingSettings(t *testing.T) {
	t.Chdir(t.TempDir())
	clearEnvironmentForTest(t, demoEOTURLVar, demoEOTModeVar, demoEOTThresholdVar, demoEOTTokenFileVar)

	parsed, _, err := parseOptions([]string{"-call", "demo-call"})
	if err != nil {
		t.Fatal(err)
	}
	defaults := config.Defaults().Agent
	if parsed.replySilence != defaults.ReplySilence || parsed.replySilenceMax != defaults.ReplySilenceMax ||
		parsed.replySilenceConfident != defaults.ReplySilenceConfident ||
		parsed.replyConfidentScore != defaults.ReplyConfidentScore ||
		parsed.previewDebounce != defaults.PreviewDebounce || parsed.previewQuiet != defaults.PreviewQuiet {
		t.Fatalf("the CLI should start from the router's defaults, got %+v", parsed)
	}

	parsed, _, err = parseOptions([]string{
		"-call", "demo-call",
		"-reply-silence=400ms", "-reply-silence-max=2s", "-reply-silence-confident=150ms",
		"-reply-confident-score=0.8", "-preview-debounce=0s", "-preview-quiet=0s",
	})
	if err != nil {
		t.Fatal(err)
	}
	if parsed.replySilence != 400*time.Millisecond || parsed.replySilenceMax != 2*time.Second ||
		parsed.replySilenceConfident != 150*time.Millisecond || parsed.replyConfidentScore != 0.8 ||
		parsed.previewDebounce != 0 || parsed.previewQuiet != 0 {
		t.Fatalf("the flags were not read: %+v", parsed)
	}
}

func TestRunPreflightsEOTBeforeBuildingRouters(t *testing.T) {
	var requests atomic.Int32
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		requests.Add(1)
		if r.Method != http.MethodPost || r.Header.Get("X-Request-ID") != "demo-preflight" {
			t.Errorf("unexpected preflight request: method=%q request-id=%q", r.Method, r.Header.Get("X-Request-ID"))
		}
		pcm, err := io.ReadAll(r.Body)
		if err != nil {
			t.Errorf("read preflight body: %v", err)
		} else if len(pcm) != 320*2 {
			t.Errorf("preflight body has %d bytes, want 640", len(pcm))
		}
		_, _ = io.WriteString(w, `{"request_id":"demo-preflight","model":"audioturn-stack16k-blend","release":"c4497ce3ba47","probability":0.5,"wait_probability":0.5,"sample_rate":16000,"samples":320,"window_samples":320}`)
	}))
	t.Cleanup(server.Close)
	t.Setenv(demoEOTURLVar, server.URL)
	t.Setenv(demoEOTModeVar, "gate")
	t.Setenv(demoEOTThresholdVar, "0.5")
	t.Setenv(demoEOTTokenFileVar, "")
	t.Setenv(configEnvVar, filepath.Join(t.TempDir(), "missing-router-config.yaml"))

	logger := slog.New(slog.NewTextHandler(io.Discard, nil))
	err := run(options{callID: "preflight-check"}, logger)
	if err == nil || !strings.Contains(err.Error(), "routing: read config") {
		t.Fatalf("run error = %v, want router config load error after successful EOT preflight", err)
	}
	if got := requests.Load(); got != 1 {
		t.Fatalf("EOT preflight requests = %d, want 1 before router construction", got)
	}
}

func clearEnvironmentForTest(t *testing.T, keys ...string) {
	t.Helper()
	for _, key := range keys {
		value, wasSet := os.LookupEnv(key)
		if err := os.Unsetenv(key); err != nil {
			t.Fatal(err)
		}
		t.Cleanup(func() {
			if wasSet {
				_ = os.Setenv(key, value)
			} else {
				_ = os.Unsetenv(key)
			}
		})
	}
}
