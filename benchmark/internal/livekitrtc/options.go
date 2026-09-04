package livekitrtc

import (
	"context"
	"errors"
	"fmt"
	"log/slog"
	"net/url"
	"os"
	"strings"

	"github.com/GetStream/Vision-Agents/benchmark/internal/transport"
)

const (
	defaultIdentity = "voicebench-caller"
	urlEnvVar       = "LIVEKIT_URL"
	apiKeyEnvVar    = "LIVEKIT_API_KEY"
	apiSecretEnvVar = "LIVEKIT_API_SECRET"
	regionEnvVar    = "LIVEKIT_REGION"
	defaultRegion   = "us"
	autoRegion      = "auto"
	livekitCloud    = ".livekit.cloud"
)

// Options join a LiveKit room as the scripted caller.
type Options struct {
	URL       string
	APIKey    string
	APISecret string
	Region    string
	Room      string
	Identity  string
	Logger    *slog.Logger
}

// Resolve fills the connection details from the LIVEKIT_* environment and validates them.
func (o *Options) Resolve() error {
	if o.URL == "" {
		o.URL = os.Getenv(urlEnvVar)
	}
	if o.APIKey == "" {
		o.APIKey = os.Getenv(apiKeyEnvVar)
	}
	if o.APISecret == "" {
		o.APISecret = os.Getenv(apiSecretEnvVar)
	}
	if o.Region == "" {
		o.Region = os.Getenv(regionEnvVar)
	}
	for _, missing := range []struct {
		value string
		name  string
	}{
		{o.URL, urlEnvVar},
		{o.APIKey, apiKeyEnvVar},
		{o.APISecret, apiSecretEnvVar},
	} {
		if missing.value == "" {
			return fmt.Errorf("livekitrtc: %s is not set", missing.name)
		}
	}
	o.Region = normalizeRegion(o.Region)
	rewritten, err := RegionalURL(o.URL, o.Region)
	if err != nil {
		return err
	}
	o.URL = rewritten
	return nil
}

// RegionalURL pins a LiveKit Cloud project URL to a region group.
// LIVEKIT_REGION defaults to us. Use auto to keep LiveKit's geo-DNS.
// us-east, us-west, and us-central map to the us group; LiveKit has no
// Ashburn-only realtime hostname.
func RegionalURL(raw, region string) (string, error) {
	region = normalizeRegion(region)
	if region == autoRegion {
		return raw, nil
	}
	parsed, err := url.Parse(raw)
	if err != nil {
		return "", fmt.Errorf("livekitrtc: %s: %w", urlEnvVar, err)
	}
	host := parsed.Hostname()
	if !strings.HasSuffix(host, livekitCloud) {
		return raw, nil
	}
	project, ok := livekitCloudProject(host)
	if !ok {
		return "", fmt.Errorf("livekitrtc: cannot pin region %s on %s", region, host)
	}
	pinned := project + "." + region + ".rtc.livekit.cloud"
	if port := parsed.Port(); port != "" {
		pinned += ":" + port
	}
	parsed.Host = pinned
	return parsed.String(), nil
}

func normalizeRegion(region string) string {
	switch strings.ToLower(strings.TrimSpace(region)) {
	case "":
		return defaultRegion
	case "us-east", "us-east-1", "us-west", "us-west-2", "us-central":
		return defaultRegion
	default:
		return strings.ToLower(strings.TrimSpace(region))
	}
}

func livekitCloudProject(host string) (string, bool) {
	rest := strings.TrimSuffix(host, livekitCloud)
	rest = strings.TrimSuffix(rest, ".rtc")
	if i := strings.LastIndex(rest, "."); i >= 0 {
		rest = rest[:i]
	}
	if rest == "" || strings.Contains(rest, ".") {
		return "", false
	}
	return rest, true
}

// ErrDisabled is returned when this binary was built without `-tags webrtc` (and cgo/libopus).
var ErrDisabled = errors.New("livekitrtc: rebuild with -tags webrtc, CGO_ENABLED=1, and libopus")

// Join connects to a LiveKit room as the benchmark caller.
func Join(ctx context.Context, options Options) (transport.Media, error) {
	return join(ctx, options)
}
